"""Check that physics.js and env.py produce the same trajectory under the same
policy (so the browser shows exactly what was trained). Needs node.

    python parity_test.py ../weights.json
"""
import json, subprocess, sys, os
import numpy as np
from env import CartPoleVec, observe, reward

here = os.path.dirname(os.path.abspath(__file__))
wpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "weights.json")
layers = json.load(open(wpath))["snapshots"][-1]["layers"]
start = [0.3, -0.2, 3.0, 1.5]
STEPS = 300

def act(o):
    x = o
    for l in layers:
        x = np.tanh(np.array(l["W"]) @ x + np.array(l["b"]))
    return x[0]

env = CartPoleVec(1, randomise=False, pushes=False)
env.s[:, 0] = start
env.reset_idx = lambda idx: None
py = []
for t in range(STEPS):
    u = act(env.obs()[0].astype(np.float64))
    env.step(np.array([u]))
    py.append(env.s[:, 0].copy())

js_src = f"""
require({json.dumps(os.path.join(here, '..', 'physics.js'))});
const PD = globalThis.Pendulum, w = require({json.dumps(os.path.abspath(wpath))});
const pol = PD.makePolicy(w.snapshots[w.snapshots.length - 1].layers);
const p = PD.nominalParams(), s = Float64Array.from({start});
let lastU = 0; const out = [];
for (let t = 0; t < {STEPS}; t++) {{
  const u = Math.max(-1, Math.min(1, pol.act(PD.observe(s, lastU))));
  PD.stepControl(s, u * PD.P.MAX_FORCE, p); lastU = u; out.push(Array.from(s));
}}
console.log(JSON.stringify(out));
"""
js = np.array(json.loads(subprocess.check_output(["node", "-e", js_src])))
py = np.array(py)
err = np.abs(js - py).max(0)
print("max abs difference over", STEPS, "steps (x, xd, th, thd):", err)
print("final state py:", py[-1].round(4), " js:", js[-1].round(4))
assert err.max() < 1e-3, "JS and Python diverged"
print("OK")
