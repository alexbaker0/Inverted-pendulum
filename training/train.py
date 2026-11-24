"""PPO training for the swing-up and balance task.

    python train.py --steps 3e7 --out runs/main                       # single pendulum
    python train.py --system double --steps 2e8 --out runs/double     # double pendulum

Writes checkpoints and, at the end, ../weights.json for the web page (the
policy network at several points during training, so the page can show the
agent learning).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import env as single_env
import env_double as double_env
from env import ACT_DIM, CTRL_DT, EPISODE_STEPS

SYSTEMS = {
    # name: (env module, env class, policy hidden, critic hidden)
    "single": (single_env, single_env.CartPoleVec, 64, 128),
    "double": (double_env, double_env.DoublePoleVec, 128, 256),
}


def angle_error(obs, system):
    """Worst angle from upright, in radians, for each row of observations."""
    a = np.abs(np.arctan2(obs[:, 3], obs[:, 2]))
    if system == "double":
        a = np.maximum(a, np.abs(np.arctan2(obs[:, 5], obs[:, 4])))
    return a

torch.set_num_threads(4)


class Policy(nn.Module):
    """Gaussian policy. The mean goes through tanh so it lives in [-1, 1]."""

    def __init__(self, hidden=64, obs_dim=single_env.OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, ACT_DIM), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))
        nn.init.orthogonal_(self.net[4].weight, 0.01)
        nn.init.zeros_(self.net[4].bias)

    def dist(self, obs):
        mean = self.net(obs)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


class Critic(nn.Module):
    def __init__(self, hidden=128, obs_dim=single_env.OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def export_policy(policy):
    layers = []
    for m in policy.net:
        if isinstance(m, nn.Linear):
            layers.append({
                "W": np.round(m.weight.detach().numpy(), 6).tolist(),
                "b": np.round(m.bias.detach().numpy(), 6).tolist(),
            })
    return layers


@torch.no_grad()
def evaluate(policy, episodes=64, seed=123, system="single"):
    """Deterministic policy, nominal physics, no pushes. Returns metrics."""
    env = SYSTEMS[system][1](episodes, seed=seed, randomise=False, pushes=False)
    obs = env.obs()
    total = np.zeros(episodes)
    alive = np.ones(episodes, bool)
    up_time = np.full(episodes, np.nan)
    crashed = np.zeros(episodes, bool)
    final_angle = np.full(episodes, 180.0)
    final_x = np.zeros(episodes)
    for t in range(EPISODE_STEPS):
        a = policy.net(torch.as_tensor(obs)).numpy()
        obs, r, c, tr, info = env.step(a)
        total += np.where(alive, r, 0)
        crashed |= c & alive
        ended = alive & (c | tr)
        fo = info["final_obs"]
        err = angle_error(fo, system)
        final_angle[ended] = np.degrees(err[ended])
        final_x[ended] = np.abs(fo[ended, 0] * 2.4)
        # first time it is within 10 degrees of upright
        first = alive & np.isnan(up_time) & (err < np.radians(10))
        up_time[first] = t * CTRL_DT
        alive &= ~(c | tr)
        if not alive.any():
            break
    survived = ~crashed
    return {
        "return": float(total.mean()),
        "crash_rate": float(crashed.mean()),
        "upright_at_end": float(((final_angle < 5) & survived).mean()),
        "mean_final_angle_deg": float(final_angle.mean()),
        "mean_final_offset_m": float(final_x.mean()),
        "median_time_to_upright_s": float(np.nanmedian(up_time)) if np.isfinite(up_time).any() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=list(SYSTEMS), default="single")
    ap.add_argument("--steps", type=float, default=2e7)
    ap.add_argument("--disturb", action="store_true",
                    help="train with random wind and ground shake (see disturb.py)")
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--init", default=None, help="checkpoint to continue from (fine-tuning)")
    ap.add_argument("--init-std", type=float, default=None,
                    help="when fine-tuning, reset the exploration noise to this std")
    ap.add_argument("--step-offset", type=float, default=0,
                    help="add to the step counts recorded in snapshots (for fine-tuning runs)")
    ap.add_argument("--snapshots", default="0,0.01,0.03,0.08,0.2,0.5,1.0",
                    help="fractions of training at which to save the policy for the web page")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mod, env_cls, ph, ch = SYSTEMS[args.system]
    OBS_DIM = mod.OBS_DIM
    env = env_cls(args.envs, seed=args.seed, disturb=args.disturb)
    policy, critic = Policy(ph, OBS_DIM), Critic(ch, OBS_DIM)
    if args.init:
        ck = torch.load(args.init)
        policy.load_state_dict(ck["policy"])
        critic.load_state_dict(ck["critic"])
        if args.init_std:
            with torch.no_grad():
                policy.log_std.fill_(float(np.log(args.init_std)))
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()),
                           lr=args.lr, eps=1e-5)

    N, T = args.envs, args.horizon
    batch = N * T
    updates = int(args.steps // batch)
    snap_at = sorted({min(updates, int(round(f * updates)))
                      for f in map(float, args.snapshots.split(","))})
    snapshots = []

    obs = torch.as_tensor(env.obs())
    ep_ret = np.zeros(N)
    recent = []
    t0 = time.time()
    log = open(os.path.join(args.out, "log.jsonl"), "w")

    for upd in range(updates + 1):
        if upd in snap_at:
            snapshots.append({"steps": int(args.step_offset) + upd * batch, "layers": export_policy(policy)})
        if upd == updates:
            break

        frac = upd / updates
        for g in opt.param_groups:
            g["lr"] = args.lr * (1 - 0.9 * frac)

        b_obs = torch.zeros(T, N, OBS_DIM)
        b_act = torch.zeros(T, N, ACT_DIM)
        b_logp = torch.zeros(T, N)
        b_rew = torch.zeros(T, N)
        b_val = torch.zeros(T, N)
        b_done = torch.zeros(T, N)       # episode ended (crash or truncation)
        b_boot = torch.zeros(T, N)       # value to add back for truncated episodes

        with torch.no_grad():
            for t in range(T):
                d = policy.dist(obs)
                a = d.sample()
                b_obs[t], b_act[t] = obs, a
                b_logp[t] = d.log_prob(a).sum(-1)
                b_val[t] = critic(obs)
                nobs, r, crashed, trunc, info = env.step(a.numpy())
                ep_ret += r
                done = crashed | trunc
                for i in np.nonzero(done)[0]:
                    recent.append(ep_ret[i])
                    ep_ret[i] = 0
                rr = torch.as_tensor(r)
                if trunc.any():
                    v_final = critic(torch.as_tensor(info["final_obs"]))
                    rr = rr + args.gamma * v_final * torch.as_tensor(trunc, dtype=torch.float32)
                b_rew[t] = rr
                b_done[t] = torch.as_tensor(done, dtype=torch.float32)
                obs = torch.as_tensor(nobs)
            last_val = critic(obs)

            adv = torch.zeros(T, N)
            gae = torch.zeros(N)
            for t in reversed(range(T)):
                nv = last_val if t == T - 1 else b_val[t + 1]
                nonterm = 1.0 - b_done[t]
                delta = b_rew[t] + args.gamma * nv * nonterm - b_val[t]
                gae = delta + args.gamma * args.lam * nonterm * gae
                adv[t] = gae
            ret = adv + b_val

        f_obs = b_obs.reshape(batch, OBS_DIM)
        f_act = b_act.reshape(batch, ACT_DIM)
        f_logp = b_logp.reshape(batch)
        f_adv = adv.reshape(batch)
        f_ret = ret.reshape(batch)

        for _ in range(args.epochs):
            perm = torch.randperm(batch)
            for s in range(0, batch, args.minibatch):
                idx = perm[s:s + args.minibatch]
                d = policy.dist(f_obs[idx])
                logp = d.log_prob(f_act[idx]).sum(-1)
                ratio = (logp - f_logp[idx]).exp()
                a_ = f_adv[idx]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg = -torch.min(ratio * a_, ratio.clamp(1 - args.clip, 1 + args.clip) * a_).mean()
                vloss = 0.5 * (critic(f_obs[idx]) - f_ret[idx]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + 0.5 * vloss - args.ent * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic.parameters()), 0.5)
                opt.step()
            with torch.no_grad():
                policy.log_std.clamp_(-3.0, 0.5)

        if upd % 10 == 0 or upd == updates - 1:
            steps = (upd + 1) * batch
            rec = float(np.mean(recent[-200:])) if recent else float("nan")
            row = {"update": upd, "steps": steps, "ep_return": rec,
                   "std": float(policy.log_std.exp().detach()), "sps": steps / (time.time() - t0)}
            if upd % 50 == 0 or upd == updates - 1:
                row["eval"] = evaluate(policy, system=args.system)
                torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()},
                           os.path.join(args.out, "ckpt.pt"))
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()

    final_eval = evaluate(policy, episodes=256, seed=999, system=args.system)
    print("final eval", json.dumps(final_eval))
    torch.save({"policy": policy.state_dict(), "critic": critic.state_dict()},
               os.path.join(args.out, "ckpt.pt"))
    with open(os.path.join(args.out, "weights.json"), "w") as f:
        json.dump({"activation": "tanh", "obs_dim": OBS_DIM, "system": args.system,
                   "eval": final_eval, "snapshots": snapshots}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
