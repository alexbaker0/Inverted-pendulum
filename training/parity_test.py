"""Check that physics.js and the Python env produce the same closed-loop
trajectory under the same policy (so the browser shows exactly what was
trained). Needs node.

    python parity_test.py                 # single, ../weights.json
    python parity_test.py --system double # double, ../weights_double.json
"""
import argparse, json, os, subprocess
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--system", choices=["single", "double"], default="single")
ap.add_argument("weights", nargs="?")
args = ap.parse_args()

if args.system == "single":
    from env import CartPoleVec as Env
    wpath = args.weights or os.path.join(here, "..", "weights.json")
    start, js_obj = [0.3, -0.2, 3.0, 1.5], "Pendulum"
else:
    from env_double import DoublePoleVec as Env
    wpath = args.weights or os.path.join(here, "..", "weights_double.json")
    start, js_obj = [0.3, 3.0, -2.5, -0.2, 1.5, -1.0], "PendulumDouble"
layers = json.load(open(wpath))["snapshots"][-1]["layers"]
STEPS = 300

def act(o):
    x = o
    for l in layers:
        x = np.tanh(np.array(l["W"]) @ x + np.array(l["b"]))
    return x[0]

env = Env(1, randomise=False, pushes=False)
env.s[:, 0] = start
env.reset_idx = lambda idx: None
py = []
for t in range(STEPS):
    u = act(env.obs()[0].astype(np.float64))
    env.step(np.array([u]))
    py.append(env.s[:, 0].copy())

js_src = f"""
require({json.dumps(os.path.join(here, '..', 'physics.js'))});
const PD = globalThis.{js_obj}, w = require({json.dumps(os.path.abspath(wpath))});
const pol = globalThis.Pendulum.makePolicy(w.snapshots[w.snapshots.length - 1].layers);
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
print(args.system, "- max abs difference over", STEPS, "steps per state variable:", err)
assert err.max() < 1e-3, "JS and Python diverged"
print("OK")
