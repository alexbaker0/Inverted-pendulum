"""Stress-test a trained policy.

    python evaluate.py runs/main/ckpt.pt

Runs the deterministic policy from random starts (any angle, any spin) under
nominal physics and under physics changed beyond the training range, and
reports how often it ends up balanced and how still it is when it gets there.
"""

import sys

import numpy as np
import torch

from env import CTRL_DT, EPISODE_STEPS, CartPoleVec, POLE_LENGTH, POLE_MASS, CART_MASS
from train import Policy


@torch.no_grad()
def run(policy, n=256, seed=7, scale=None, pushes=False):
    env = CartPoleVec(n, seed=seed, randomise=False, pushes=pushes)
    if scale:
        env.p[0] *= scale.get("cart_mass", 1)
        env.p[1] *= scale.get("pole_mass", 1)
        env.p[2] *= scale.get("pole_length", 1)
        env.p[3] *= scale.get("friction", 1)
    env.reset_idx = lambda idx: None   # keep one episode per copy, no auto-reset
    obs = env.obs()
    crashed = np.zeros(n, bool)
    up_at = np.full(n, np.nan)
    th_hist, x_hist, u_hist = [], [], []
    for t in range(EPISODE_STEPS):
        u = policy.net(torch.as_tensor(obs)).numpy()
        obs, r, c, tr, _ = env.step(u)
        crashed |= c
        th = np.arctan2(obs[:, 3], obs[:, 2])
        newly = np.isnan(up_at) & (np.abs(th) < np.radians(10))
        up_at[newly] = (t + 1) * CTRL_DT
        th_hist.append(th); x_hist.append(env.s[0].copy()); u_hist.append(u[:, 0])
    th = np.abs(np.array(th_hist)); x = np.array(x_hist); u = np.abs(np.array(u_hist))
    last = slice(-250, None)          # final 5 seconds
    held = (~crashed) & (th[last].max(0) < np.radians(5))
    return {
        "balanced_final_5s": held.mean(),
        "crashed": crashed.mean(),
        "median_swing_up_s": np.nanmedian(up_at),
        "worst_swing_up_s": np.nanmax(up_at),
        "final_angle_deg_rms": np.degrees(np.sqrt((th[last][:, held] ** 2).mean())) if held.any() else np.nan,
        "final_cart_drift_m": np.abs(x[-1][held]).mean() if held.any() else np.nan,
        "final_force_N": 15 * u[last][:, held].mean() if held.any() else np.nan,
    }


def main(path):
    policy = Policy()
    policy.load_state_dict(torch.load(path)["policy"])
    cases = [
        ("nominal", None, False),
        ("nominal + random shoves", None, True),
        ("pole 30% longer", {"pole_length": 1.3}, False),
        ("pole 30% shorter", {"pole_length": 0.7}, False),
        ("pole twice as heavy", {"pole_mass": 2.0}, False),
        ("cart twice as heavy", {"cart_mass": 2.0}, False),
        ("friction x3", {"friction": 3.0}, False),
    ]
    print(f"{'case':28s} {'balanced':>9s} {'crashed':>8s} {'swing-up':>9s} {'worst':>6s} {'angle':>7s} {'drift':>6s} {'force':>6s}")
    for name, sc, pushes in cases:
        m = run(policy, scale=sc, pushes=pushes)
        print(f"{name:28s} {m['balanced_final_5s']:9.1%} {m['crashed']:8.1%} "
              f"{m['median_swing_up_s']:8.2f}s {m['worst_swing_up_s']:5.1f}s "
              f"{m['final_angle_deg_rms']:6.3f}° {m['final_cart_drift_m']:5.3f}m {m['final_force_N']:5.2f}N")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/main/ckpt.pt")
