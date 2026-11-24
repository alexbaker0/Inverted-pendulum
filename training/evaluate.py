"""Stress-test a trained policy.

    python evaluate.py runs/main/ckpt.pt
    python evaluate.py runs/double/ckpt.pt --system double

Runs the deterministic policy from random starts (any angle, any spin) under
nominal physics and under physics changed beyond the training range, and
reports how often it ends up balanced and how still it is when it gets there.
"""


import numpy as np
import torch

import argparse

from env import CTRL_DT, EPISODE_STEPS
from train import Policy, SYSTEMS, angle_error


@torch.no_grad()
def run(policy, n=256, seed=7, scale=None, pushes=False, system="single", dist=None):
    env = SYSTEMS[system][1](n, seed=seed, randomise=False, pushes=pushes)
    if dist:                                   # (wind_on, ground_on, strength)
        env.dist.fixed = dist
        env.dist.reset_idx(np.arange(n))
    if scale:
        rows = ({"cart_mass": [0], "pole_mass": [1], "pole_length": [2], "friction": [3]} if system == "single" else
                {"cart_mass": [0], "pole_mass": [1, 2], "pole_length": [3, 4], "friction": [5]})
        for k, idx in rows.items():
            for i in idx:
                env.p[i] *= scale.get(k, 1)
    env.reset_idx = lambda idx: None   # keep one episode per copy, no auto-reset
    obs = env.obs()
    crashed = np.zeros(n, bool)
    up_at = np.full(n, np.nan)
    th_hist, x_hist, u_hist = [], [], []
    for t in range(EPISODE_STEPS):
        u = policy.net(torch.as_tensor(obs)).numpy()
        obs, r, c, tr, _ = env.step(u)
        crashed |= c
        th = angle_error(obs, system)
        newly = np.isnan(up_at) & (np.abs(th) < np.radians(10))
        up_at[newly] = (t + 1) * CTRL_DT
        th_hist.append(th); x_hist.append(env.s[0].copy()); u_hist.append(u[:, 0])
    force_scale = SYSTEMS[system][0].MAX_FORCE
    th = np.abs(np.array(th_hist)); x = np.array(x_hist); u = np.abs(np.array(u_hist))
    last = slice(-250, None)          # final 5 seconds
    held = (~crashed) & (th[last].max(0) < np.radians(5))
    up = (~crashed) & (th[last].max(0) < np.radians(30))
    return {
        "balanced_final_5s": held.mean(),
        "up_final_5s": up.mean(),
        "crashed": crashed.mean(),
        "median_swing_up_s": np.nanmedian(up_at),
        "worst_swing_up_s": np.nanmax(up_at),
        # averaged over the final 5 s of every run that stayed up
        "final_angle_deg_rms": np.degrees(np.sqrt((th[last][:, up] ** 2).mean())) if up.any() else np.nan,
        "final_cart_drift_m": np.abs(x[last][:, up]).mean() if up.any() else np.nan,
        "final_force_N": force_scale * u[last][:, up].mean() if up.any() else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="runs/main/ckpt.pt")
    ap.add_argument("--system", choices=list(SYSTEMS), default="single")
    ap.add_argument("--quick", action="store_true", help="only the nominal and disturbance cases")
    args = ap.parse_args()
    mod, _, hidden, _ = SYSTEMS[args.system]
    policy = Policy(hidden, mod.OBS_DIM)
    policy.load_state_dict(torch.load(args.ckpt)["policy"])
    system = args.system
    cases = [
        ("nominal", None, False),
        ("nominal + random shoves", None, True),
        ("wind, strength 0.5", None, False, (True, False, 0.5)),
        ("ground shake, strength 0.5", None, False, (False, True, 0.5)),
        ("wind + shake, strength 0.5", None, False, (True, True, 0.5)),
        ("wind, strength 1", None, False, (True, False, 1.0)),
        ("ground shake, strength 1", None, False, (False, True, 1.0)),
        ("wind + shake, strength 1", None, False, (True, True, 1.0)),
        ("wind + shake, strength 1.5", None, False, (True, True, 1.5)),
        ("pole 30% longer", {"pole_length": 1.3}, False),
        ("pole 30% shorter", {"pole_length": 0.7}, False),
        ("pole twice as heavy", {"pole_mass": 2.0}, False),
        ("cart twice as heavy", {"cart_mass": 2.0}, False),
        ("friction x3", {"friction": 3.0}, False),
    ]
    print(f"{'case':28s} {'stays up':>9s} {'within 5°':>9s} {'crashed':>8s} {'swing-up':>9s} {'worst':>6s} {'angle':>7s} {'drift':>6s} {'force':>6s}")
    if args.quick:
        cases = cases[:9]
    for name, sc, pushes, *dist in cases:
        m = run(policy, scale=sc, pushes=pushes, system=system, dist=dist[0] if dist else None)
        print(f"{name:28s} {m['up_final_5s']:9.1%} {m['balanced_final_5s']:9.1%} {m['crashed']:8.1%} "
              f"{m['median_swing_up_s']:8.2f}s {m['worst_swing_up_s']:5.1f}s "
              f"{m['final_angle_deg_rms']:6.3f}° {m['final_cart_drift_m']:5.3f}m {m['final_force_N']:5.2f}N")


if __name__ == "__main__":
    main()
