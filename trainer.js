// In-browser PPO trainer, run as a Web Worker. No libraries: a small MLP with
// hand-written backprop and Adam. Same world and reward as training/train.py,
// smaller networks so it runs at a useful speed in JavaScript.
//
// Messages in:  {cmd: "start"} | {cmd: "pause"} | {cmd: "reset", seed}
// Messages out: {type: "progress", steps, updates, epReturn, std, sps, layers}
"use strict";
if (typeof importScripts === "function") importScripts("physics.js");
const PD = (typeof self !== "undefined" ? self : globalThis).Pendulum;
const P = PD.P;

// ---------- random ----------
let seedState = 12345;
function rand() { // mulberry32
  seedState |= 0; seedState = (seedState + 0x6D2B79F5) | 0;
  let t = Math.imul(seedState ^ (seedState >>> 15), 1 | seedState);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function uni(a, b) { return a + (b - a) * rand(); }
function gauss() {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- MLP ----------
// sizes e.g. [6, 32, 32, 1]; tanh on hidden layers; output tanh if outTanh.
class MLP {
  constructor(sizes, outTanh, outScale) {
    this.sizes = sizes; this.outTanh = outTanh;
    this.L = sizes.length - 1;
    this.offW = []; this.offB = [];
    let n = 0;
    for (let k = 0; k < this.L; k++) {
      this.offW.push(n); n += sizes[k] * sizes[k + 1];
      this.offB.push(n); n += sizes[k + 1];
    }
    this.n = n;
    this.p = new Float32Array(n);
    this.g = new Float32Array(n);
    for (let k = 0; k < this.L; k++) {
      const fanIn = sizes[k], fanOut = sizes[k + 1];
      const scale = (k === this.L - 1 ? outScale : 1) * Math.sqrt(1 / fanIn);
      for (let i = 0; i < fanIn * fanOut; i++) this.p[this.offW[k] + i] = gauss() * scale;
    }
    this.acts = sizes.map(s => new Float32Array(s));
    this.deltas = sizes.map(s => new Float32Array(s));
  }
  forward(x) {
    const a0 = this.acts[0];
    for (let i = 0; i < this.sizes[0]; i++) a0[i] = x[i];
    for (let k = 0; k < this.L; k++) {
      const nin = this.sizes[k], nout = this.sizes[k + 1];
      const W = this.offW[k], B = this.offB[k], inp = this.acts[k], out = this.acts[k + 1];
      const tanh = k < this.L - 1 || this.outTanh;
      for (let o = 0; o < nout; o++) {
        let s = this.p[B + o];
        const row = W + o * nin;
        for (let i = 0; i < nin; i++) s += this.p[row + i] * inp[i];
        out[o] = tanh ? Math.tanh(s) : s;
      }
    }
    return this.acts[this.L];
  }
  // dOut: gradient of loss w.r.t. network output (post-activation). Accumulates into g.
  backward(dOut) {
    const L = this.L;
    const dl = this.deltas[L];
    for (let o = 0; o < this.sizes[L]; o++) {
      const y = this.acts[L][o];
      dl[o] = this.outTanh ? dOut[o] * (1 - y * y) : dOut[o];
    }
    for (let k = L - 1; k >= 0; k--) {
      const nin = this.sizes[k], nout = this.sizes[k + 1];
      const W = this.offW[k], B = this.offB[k], inp = this.acts[k];
      const d = this.deltas[k + 1], dprev = this.deltas[k];
      if (k > 0) dprev.fill(0);
      for (let o = 0; o < nout; o++) {
        const dd = d[o];
        if (dd === 0) continue;
        this.g[B + o] += dd;
        const row = W + o * nin;
        for (let i = 0; i < nin; i++) {
          this.g[row + i] += dd * inp[i];
          if (k > 0) dprev[i] += dd * this.p[row + i];
        }
      }
      if (k > 0) for (let i = 0; i < nin; i++) dprev[i] *= 1 - inp[i] * inp[i];
    }
  }
  exportLayers() {
    const layers = [];
    for (let k = 0; k < this.L; k++) {
      const nin = this.sizes[k], nout = this.sizes[k + 1], W = [];
      for (let o = 0; o < nout; o++)
        W.push(Array.from(this.p.subarray(this.offW[k] + o * nin, this.offW[k] + (o + 1) * nin)));
      layers.push({ W, b: Array.from(this.p.subarray(this.offB[k], this.offB[k] + nout)) });
    }
    return layers;
  }
}

class Adam {
  constructor(n, lr) {
    this.m = new Float32Array(n); this.v = new Float32Array(n);
    this.t = 0; this.lr = lr;
  }
  step(p, g, scale) {
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-5;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (let i = 0; i < p.length; i++) {
      const gi = g[i] * scale;
      this.m[i] = b1 * this.m[i] + (1 - b1) * gi;
      this.v[i] = b2 * this.v[i] + (1 - b2) * gi * gi;
      p[i] -= this.lr * (this.m[i] / c1) / (Math.sqrt(this.v[i] / c2) + eps);
    }
  }
}

// ---------- environment copies ----------
const N_ENVS = 16, HORIZON = 512, BATCH = N_ENVS * HORIZON;
const EPOCHS = 4, MINIBATCH = 512, GAMMA = 0.99, LAM = 0.95, CLIP = 0.2, LR = 6e-4;
const D = P.OBS_DIM;

function makeEnv() {
  return { s: new Float64Array(4), p: {}, t: 0, lastU: 0, push: 0, pushLeft: 0, ret: 0 };
}
function resetEnv(e) {
  e.p = { mc: P.CART_MASS * uni(0.8, 1.2), mp: P.POLE_MASS * uni(0.7, 1.3),
          l: P.POLE_LENGTH / 2 * uni(0.85, 1.15),
          bc: P.CART_FRICTION * uni(0.5, 2), bp: P.POLE_FRICTION * uni(0.5, 2) };
  e.s[0] = uni(-1.5, 1.5); e.s[1] = uni(-1, 1);
  if (rand() < 0.25) { e.s[2] = uni(-0.5, 0.5); e.s[3] = uni(-2, 2); }
  else { e.s[2] = uni(-Math.PI, Math.PI); e.s[3] = uni(-8, 8); }
  e.t = 0; e.lastU = 0; e.push = 0; e.pushLeft = 0; e.ret = 0;
}

let actor, critic, logStd, optA, optC, optS, envs, steps, updates, recent, running = false;
const obsBuf = new Float32Array(BATCH * D), actBuf = new Float32Array(BATCH),
      logpBuf = new Float32Array(BATCH), rewBuf = new Float32Array(BATCH),
      valBuf = new Float32Array(BATCH), doneBuf = new Uint8Array(BATCH),
      advBuf = new Float32Array(BATCH), retBuf = new Float32Array(BATCH);
const tmpObs = new Float32Array(D);

function init(seed) {
  seedState = seed >>> 0;
  actor = new MLP([D, 32, 32, 1], true, 0.01);
  critic = new MLP([D, 64, 64, 1], false, 1.0);
  logStd = [-0.5];
  optA = new Adam(actor.n, LR); optC = new Adam(critic.n, LR); optS = new Adam(1, LR);
  envs = Array.from({ length: N_ENVS }, () => { const e = makeEnv(); resetEnv(e); return e; });
  steps = 0; updates = 0; recent = [];
}

const LOG2PI = Math.log(2 * Math.PI);
function logProb(a, mean, ls) {
  const z = (a - mean) / Math.exp(ls);
  return -0.5 * z * z - ls - 0.5 * LOG2PI;
}

function rollout() {
  const std = Math.exp(logStd[0]);
  for (let t = 0; t < HORIZON; t++) {
    for (let n = 0; n < N_ENVS; n++) {
      const e = envs[n], idx = t * N_ENVS + n;
      const o = obsBuf.subarray(idx * D, idx * D + D);
      PD.observe(e.s, e.lastU, o);
      const mean = actor.forward(o)[0];
      const a = mean + std * gauss();
      actBuf[idx] = a;
      logpBuf[idx] = logProb(a, mean, logStd[0]);
      valBuf[idx] = critic.forward(o)[0];

      const u = Math.max(-1, Math.min(1, a));
      let F = u * P.MAX_FORCE;
      if (e.pushLeft === 0 && rand() < 0.004) {
        e.push = uni(-25, 25); e.pushLeft = 3 + Math.floor(rand() * 7);
      }
      if (e.pushLeft > 0) { F += e.push; e.pushLeft--; }
      PD.stepControl(e.s, F, e.p);
      e.t++;
      let r = PD.reward(e.s, u);
      e.lastU = u;
      const crashed = Math.abs(e.s[0]) > P.RAIL_HALF;
      if (crashed) r = -20;
      const trunc = !crashed && e.t >= P.EPISODE_STEPS;
      e.ret += r;
      if (trunc) {
        PD.observe(e.s, e.lastU, tmpObs);
        r += GAMMA * critic.forward(tmpObs)[0];
      }
      rewBuf[idx] = r;
      doneBuf[idx] = crashed || trunc ? 1 : 0;
      if (crashed || trunc) {
        recent.push(e.ret); if (recent.length > 100) recent.shift();
        resetEnv(e);
      }
    }
  }
  // bootstrap + GAE
  const lastVal = new Float32Array(N_ENVS);
  for (let n = 0; n < N_ENVS; n++) {
    PD.observe(envs[n].s, envs[n].lastU, tmpObs);
    lastVal[n] = critic.forward(tmpObs)[0];
  }
  for (let n = 0; n < N_ENVS; n++) {
    let gae = 0;
    for (let t = HORIZON - 1; t >= 0; t--) {
      const idx = t * N_ENVS + n;
      const nv = t === HORIZON - 1 ? lastVal[n] : valBuf[idx + N_ENVS];
      const nonterm = 1 - doneBuf[idx];
      const delta = rewBuf[idx] + GAMMA * nv * nonterm - valBuf[idx];
      gae = delta + GAMMA * LAM * nonterm * gae;
      advBuf[idx] = gae;
      retBuf[idx] = gae + valBuf[idx];
    }
  }
  steps += BATCH;
}

const perm = new Uint32Array(BATCH);
const dA = new Float32Array(1), dC = new Float32Array(1);
function learn(frac) {
  const lr = LR * (1 - 0.8 * Math.min(1, frac));
  optA.lr = optC.lr = optS.lr = lr;
  for (let ep = 0; ep < EPOCHS; ep++) {
    for (let i = 0; i < BATCH; i++) perm[i] = i;
    for (let i = BATCH - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1)); const t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (let s0 = 0; s0 < BATCH; s0 += MINIBATCH) {
      let mean = 0, sq = 0;
      for (let k = 0; k < MINIBATCH; k++) { const a = advBuf[perm[s0 + k]]; mean += a; sq += a * a; }
      mean /= MINIBATCH;
      const sd = Math.sqrt(Math.max(sq / MINIBATCH - mean * mean, 0)) + 1e-8;
      actor.g.fill(0); critic.g.fill(0);
      let gLogStd = 0;
      const ls = logStd[0], std = Math.exp(ls);
      for (let k = 0; k < MINIBATCH; k++) {
        const idx = perm[s0 + k];
        const o = obsBuf.subarray(idx * D, idx * D + D);
        const adv = (advBuf[idx] - mean) / sd;
        const mu = actor.forward(o)[0];
        const a = actBuf[idx];
        const lp = logProb(a, mu, ls);
        const ratio = Math.exp(lp - logpBuf[idx]);
        // gradient of -min(r*A, clip(r)*A) w.r.t. logp is -r*A when unclipped term is active
        const clipped = (adv > 0 && ratio > 1 + CLIP) || (adv < 0 && ratio < 1 - CLIP);
        const dLdlogp = clipped ? 0 : -ratio * adv;
        if (dLdlogp !== 0) {
          const z = (a - mu) / std;
          dA[0] = dLdlogp * z / std;           // d logp / d mu = z / std
          actor.backward(dA);
          gLogStd += dLdlogp * (z * z - 1);    // d logp / d logstd
        }
        const v = critic.forward(o)[0];
        dC[0] = (v - retBuf[idx]) * 0.5;       // 0.5 * 0.5 * (v-R)^2 -> 0.5(v-R)
        critic.backward(dC);
      }
      const sc = 1 / MINIBATCH;
      clipGrad(actor.g, sc); clipGrad(critic.g, sc);
      optA.step(actor.p, actor.g, sc);
      optC.step(critic.p, critic.g, sc);
      const gs = new Float32Array([gLogStd]);
      optS.step(logStd, gs, sc);
      logStd[0] = Math.max(-3, Math.min(0.5, logStd[0]));
    }
  }
  updates++;
}
function clipGrad(g, sc) {
  let n = 0;
  for (let i = 0; i < g.length; i++) n += g[i] * g[i];
  n = Math.sqrt(n) * sc;
  if (n > 0.5) { const f = 0.5 / n; for (let i = 0; i < g.length; i++) g[i] *= f; }
}

function oneUpdate() {
  rollout();
  learn(steps / 3e7);
}

// Exposed for node testing
const api = { init, oneUpdate, get steps() { return steps; }, get recent() { return recent; },
              get std() { return Math.exp(logStd[0]); }, actorLayers: () => actor.exportLayers() };
if (typeof module !== "undefined") module.exports = api;

if (typeof self !== "undefined" && typeof importScripts === "function") {
  let t0 = 0, s0 = 0;
  init(Date.now());
  function loop() {
    if (!running) return;
    const tStart = performance.now();
    oneUpdate();
    const avg = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : null;
    const now = performance.now();
    const sps = (steps - s0) / ((now - t0) / 1000);
    self.postMessage({ type: "progress", steps, updates, epReturn: avg,
                       std: Math.exp(logStd[0]), sps, ms: now - tStart,
                       layers: actor.exportLayers() });
    setTimeout(loop, 0);
  }
  self.onmessage = (ev) => {
    const m = ev.data;
    if (m.cmd === "start" && !running) { running = true; t0 = performance.now(); s0 = steps; loop(); }
    else if (m.cmd === "pause") running = false;
    else if (m.cmd === "reset") { running = false; init(m.seed || Date.now());
      self.postMessage({ type: "progress", steps: 0, updates: 0, epReturn: null, std: Math.exp(logStd[0]),
                         sps: 0, layers: actor.exportLayers() }); }
  };
}
