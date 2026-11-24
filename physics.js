// Cart-and-pole world. The same equations, constants, observation and reward
// as training/env.py, so the trained network sees exactly what it learned on.
// Used by the page (index.html) and by the in-browser trainer (trainer.js).
(function (root) {
  "use strict";

  const P = {
    G: 9.81,
    CART_MASS: 1.0,      // kg
    POLE_MASS: 0.2,      // kg
    POLE_LENGTH: 1.0,    // m, pivot to tip
    CART_FRICTION: 0.5,  // N per m/s
    POLE_FRICTION: 0.004,// N m per rad/s
    MAX_FORCE: 15.0,     // N
    RAIL_HALF: 2.4,      // m
    PHYS_DT: 0.0025,     // s
    CTRL_DT: 0.02,       // s
    EPISODE_STEPS: 1000,
    OBS_DIM: 6,
  };
  P.SUBSTEPS = Math.round(P.CTRL_DT / P.PHYS_DT);

  // params: {mc, mp, l (HALF length), bc, bp}
  function nominalParams() {
    return { mc: P.CART_MASS, mp: P.POLE_MASS, l: P.POLE_LENGTH / 2,
             bc: P.CART_FRICTION, bp: P.POLE_FRICTION };
  }

  // s = [x, xd, th, thd]; writes derivative into out.
  // w: wind load on the pole (N per metre, + right); ag: rail acceleration.
  function derivs(s, F, p, out, w, ag) {
    w = w || 0; ag = ag || 0;
    const xd = s[1], th = s[2], thd = s[3];
    const total = p.mc + p.mp;
    const sin = Math.sin(th), cos = Math.cos(th);
    const fw = w * 2 * p.l;
    const qx = fw - total * ag, qth = (fw - p.mp * ag) * p.l * cos;
    const tmp = (F + qx + p.mp * p.l * thd * thd * sin - p.bc * xd) / total;
    const thdd = (P.G * sin + qth / (p.mp * p.l) - cos * tmp - p.bp * thd / (p.mp * p.l)) /
                 (p.l * (4 / 3 - p.mp * cos * cos / total));
    out[0] = xd;
    out[1] = tmp - p.mp * p.l * thdd * cos / total;
    out[2] = thd;
    out[3] = thdd;
  }

  const k1 = new Float64Array(4), k2 = new Float64Array(4),
        k3 = new Float64Array(4), k4 = new Float64Array(4), tmpS = new Float64Array(4);

  function rk4(s, F, p, dt, w, ag) {
    derivs(s, F, p, k1, w, ag);
    for (let i = 0; i < 4; i++) tmpS[i] = s[i] + 0.5 * dt * k1[i];
    derivs(tmpS, F, p, k2, w, ag);
    for (let i = 0; i < 4; i++) tmpS[i] = s[i] + 0.5 * dt * k2[i];
    derivs(tmpS, F, p, k3, w, ag);
    for (let i = 0; i < 4; i++) tmpS[i] = s[i] + dt * k3[i];
    derivs(tmpS, F, p, k4, w, ag);
    for (let i = 0; i < 4; i++) s[i] += dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  }

  function wrap(a) {
    a = (a + Math.PI) % (2 * Math.PI);
    if (a < 0) a += 2 * Math.PI;
    return a - Math.PI;
  }

  function observe(s, lastU, out) {
    out = out || new Float32Array(P.OBS_DIM);
    out[0] = s[0] / P.RAIL_HALF;
    out[1] = s[1] / 3;
    out[2] = Math.cos(s[2]);
    out[3] = Math.sin(s[2]);
    out[4] = s[3] / 10;
    out[5] = lastU;
    return out;
  }

  function reward(s, u) {
    const x = s[0], xd = s[1], th = s[2], thd = s[3];
    let up = (1 + Math.cos(th)) / 2;
    up = up * up;
    const centred = 0.5 + 0.3 * Math.exp(-(x / 0.8) * (x / 0.8)) + 0.2 * Math.exp(-(x / 0.1) * (x / 0.1));
    const gentle = 0.8 + 0.2 * (1 - u * u);
    const still = 0.5 + 0.5 * Math.exp(-(thd / 2) * (thd / 2) - xd * xd);
    return up * centred * gentle * still;
  }

  // Advance one control period with force F (newtons). Mutates s.
  // agAt(k): rail acceleration at substep k (optional).
  function stepControl(s, F, p, w, agAt) {
    for (let i = 0; i < P.SUBSTEPS; i++) rk4(s, F, p, P.PHYS_DT, w, agAt ? agAt(i) : 0);
    s[2] = wrap(s[2]);
  }

  // Small MLP forward pass with tanh on every layer (incl. output).
  // layers: [{W: [[...]], b: [...]}], W is (out x in) as exported by PyTorch.
  function makePolicy(layers) {
    const L = layers.map(l => ({
      out: l.W.length, inp: l.W[0].length,
      W: Float32Array.from(l.W.flat()), b: Float32Array.from(l.b),
    }));
    const bufs = L.map(l => new Float32Array(l.out));
    return {
      layers: L,
      act(obs) {
        let x = obs;
        for (let k = 0; k < L.length; k++) {
          const l = L[k], y = bufs[k];
          for (let o = 0; o < l.out; o++) {
            let a = l.b[o];
            const row = o * l.inp;
            for (let i = 0; i < l.inp; i++) a += l.W[row + i] * x[i];
            y[o] = Math.tanh(a);
          }
          x = y;
        }
        return x[0];
      },
      // also return hidden activations for visualisation
      activations() { return bufs; },
    };
  }

  // ---------------------------------------------------------------------------
  // Double pendulum on a cart: two uniform rods hinged end to end.
  // Same equations as training/env_double.py (Lagrangian, 3x3 mass matrix).
  const D = {
    G: 9.81, CART_MASS: 1.0, POLE1_MASS: 0.2, POLE2_MASS: 0.2,
    POLE1_LENGTH: 0.6, POLE2_LENGTH: 0.6, CART_FRICTION: 0.5, JOINT_FRICTION: 0.002,
    MAX_FORCE: 25.0, RAIL_HALF: 2.4, PHYS_DT: 0.0025, CTRL_DT: 0.02, EPISODE_STEPS: 1000, OBS_DIM: 9,
  };
  D.SUBSTEPS = Math.round(D.CTRL_DT / D.PHYS_DT);

  function dNominal() {
    return { mc: D.CART_MASS, m1: D.POLE1_MASS, m2: D.POLE2_MASS, L1: D.POLE1_LENGTH,
             L2: D.POLE2_LENGTH, bc: D.CART_FRICTION, bj: D.JOINT_FRICTION };
  }

  // s = [x, th1, th2, xd, t1d, t2d]
  function dDerivs(s, F, p, out, w, ag) {
    w = w || 0; ag = ag || 0;
    const th1 = s[1], th2 = s[2], xd = s[3], t1d = s[4], t2d = s[5];
    const l1 = p.L1 / 2, l2 = p.L2 / 2;
    const a = p.m1 * l1 + p.m2 * p.L1, b = p.m2 * l2, c = p.m2 * p.L1 * l2;
    const J1 = p.m1 * l1 * l1 + p.m1 * p.L1 * p.L1 / 12 + p.m2 * p.L1 * p.L1;
    const J2 = p.m2 * l2 * l2 + p.m2 * p.L2 * p.L2 / 12;
    const s1 = Math.sin(th1), c1 = Math.cos(th1), s2 = Math.sin(th2), c2 = Math.cos(th2);
    const s12 = Math.sin(th1 - th2), c12 = Math.cos(th1 - th2);
    const M11 = p.mc + p.m1 + p.m2, M12 = a * c1, M13 = b * c2, M22 = J1, M23 = c * c12, M33 = J2;
    const rel = t2d - t1d;
    const h1 = w * p.L1 - p.m1 * ag, h2 = w * p.L2 - p.m2 * ag;   // wind + rail pseudo-force
    const g1 = h1 + h2 - p.mc * ag, g2 = (h1 * l1 + h2 * p.L1) * c1, g3 = h2 * l2 * c2;
    const f1 = F + g1 + a * s1 * t1d * t1d + b * s2 * t2d * t2d - p.bc * xd;
    const f2 = g2 - c * s12 * t2d * t2d + a * D.G * s1 - p.bj * t1d + p.bj * rel;
    const f3 = g3 + c * s12 * t1d * t1d + b * D.G * s2 - p.bj * rel;
    const C11 = M22 * M33 - M23 * M23, C12 = M13 * M23 - M12 * M33, C13 = M12 * M23 - M13 * M22;
    const C22 = M11 * M33 - M13 * M13, C23 = M12 * M13 - M11 * M23, C33 = M11 * M22 - M12 * M12;
    const det = M11 * C11 + M12 * C12 + M13 * C13;
    out[0] = xd; out[1] = t1d; out[2] = t2d;
    out[3] = (C11 * f1 + C12 * f2 + C13 * f3) / det;
    out[4] = (C12 * f1 + C22 * f2 + C23 * f3) / det;
    out[5] = (C13 * f1 + C23 * f2 + C33 * f3) / det;
  }
  const q1 = new Float64Array(6), q2 = new Float64Array(6), q3 = new Float64Array(6),
        q4 = new Float64Array(6), qs = new Float64Array(6);
  function dRk4(s, F, p, dt, w, ag) {
    dDerivs(s, F, p, q1, w, ag);
    for (let i = 0; i < 6; i++) qs[i] = s[i] + 0.5 * dt * q1[i];
    dDerivs(qs, F, p, q2, w, ag);
    for (let i = 0; i < 6; i++) qs[i] = s[i] + 0.5 * dt * q2[i];
    dDerivs(qs, F, p, q3, w, ag);
    for (let i = 0; i < 6; i++) qs[i] = s[i] + dt * q3[i];
    dDerivs(qs, F, p, q4, w, ag);
    for (let i = 0; i < 6; i++) s[i] += dt / 6 * (q1[i] + 2 * q2[i] + 2 * q3[i] + q4[i]);
  }
  function dObserve(s, lastU, out) {
    out = out || new Float32Array(D.OBS_DIM);
    out[0] = s[0] / D.RAIL_HALF; out[1] = s[3] / 3;
    out[2] = Math.cos(s[1]); out[3] = Math.sin(s[1]);
    out[4] = Math.cos(s[2]); out[5] = Math.sin(s[2]);
    out[6] = s[4] / 10; out[7] = s[5] / 15; out[8] = lastU;
    return out;
  }
  function dReward(s, u) {
    const x = s[0], xd = s[3], t1d = s[4], t2d = s[5];
    const up = ((1 + Math.cos(s[1])) / 2) * ((1 + Math.cos(s[2])) / 2);
    const centred = 0.5 + 0.3 * Math.exp(-(x / 0.8) * (x / 0.8)) + 0.2 * Math.exp(-(x / 0.1) * (x / 0.1));
    const gentle = 0.8 + 0.2 * (1 - u * u);
    const still = 0.5 + 0.5 * Math.exp(-(t1d / 2) * (t1d / 2) - (t2d / 2) * (t2d / 2) - xd * xd);
    return up * up * centred * gentle * still;
  }
  function dStepControl(s, F, p, w, agAt) {
    for (let i = 0; i < D.SUBSTEPS; i++) dRk4(s, F, p, D.PHYS_DT, w, agAt ? agAt(i) : 0);
    s[1] = wrap(s[1]); s[2] = wrap(s[2]);
  }

  root.PendulumDouble = { P: D, nominalParams: dNominal, derivs: dDerivs, rk4: dRk4,
                          observe: dObserve, reward: dReward, stepControl: dStepControl, wrap };

  // ---------------------------------------------------------------------------
  // Random wind and ground shake, same statistics as training/disturb.py.
  // windScale: N per metre at strength 1; groundScale: peak m/s^2 at strength 1.
  function gauss() {
    let u = 0; while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }
  function makeDisturbance(windScale, groundScale) {
    const d = { wind: false, ground: false, strength: 0.5, w: 0, wMean: 0,
                freq: [], amp: [], phase: [], windScale, groundScale };
    d.reset = function () {
      const W = windScale * d.strength;
      d.wMean = (Math.random() * 1.2 - 0.6) * W;
      d.w = d.wMean + 0.5 * W * gauss();
      const a = [0, 1, 2, 3].map(() => Math.random()), sum = a.reduce((x, y) => x + y, 0);
      d.freq = a.map(() => 0.3 + 1.2 * Math.random());
      d.phase = a.map(() => 2 * Math.PI * Math.random());
      d.amp = a.map(v => v / sum * groundScale);          // scaled by strength when used
    };
    d.stepWind = function (dt) {
      const W = windScale * d.strength;
      if (Math.random() < dt / 8) d.wMean = (Math.random() * 1.2 - 0.6) * W;
      d.w += (d.wMean - d.w) * dt / 1.5 + 0.5 * W * Math.sqrt(2 * dt / 1.5) * gauss();
      d.w = Math.max(-2.5 * W, Math.min(2.5 * W, d.w));
      return d.wind ? d.w : 0;
    };
    // rail acceleration and displacement at time t
    d.acc = function (t) {
      if (!d.ground) return 0;
      let a = 0;
      for (let k = 0; k < 4; k++) a -= d.strength * d.amp[k] * Math.sin(2 * Math.PI * d.freq[k] * t + d.phase[k]);
      return a;
    };
    d.disp = function (t) {
      if (!d.ground) return 0;
      let g = 0;
      for (let k = 0; k < 4; k++) {
        const om = 2 * Math.PI * d.freq[k];
        g += d.strength * d.amp[k] / (om * om) * Math.sin(om * t + d.phase[k]);
      }
      return g;
    };
    d.reset();
    return d;
  }

  root.Pendulum = { P, nominalParams, derivs, rk4, wrap, observe, reward, stepControl, makePolicy, makeDisturbance };
})(typeof self !== "undefined" ? self : globalThis);
