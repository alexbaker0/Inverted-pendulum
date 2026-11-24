# Inverted Pendulum

A cart on a finite rail learns, with reinforcement learning (PPO), to swing a
pole up from **any** starting angle and spin and then hold it upright and still.
There are two worlds: a **single** pendulum, and a chaotic **double** pendulum
(two rods hinged end to end, still driven by one motor). Link to the double
directly with `#double`.

**Live:** https://alexbaker0.github.io/Inverted-pendulum/

On the page you can:

- watch the trained agent swing up and balance, from random, hanging, or fast-spinning starts
- grab the pole and throw it, or shove the cart (`←` `→`), and watch it recover
- switch the motor off to watch the pendulum swing freely, or drive the cart yourself
- scrub back through training to see the same network when it was clueless
- see the whole network wired up: every neuron and all 4,544 connections, with live signals, and hover to trace any neuron
- change the pole length, the masses and the friction beyond what it was trained on
- train a brand-new network from scratch in your browser and watch it learn
- turn on gusty **wind** or a **shaking rail** (or both) so the agent has to keep working

## The world

| | |
|---|---|
| Pole | uniform rod, 1.0 m, 0.2 kg, pivot friction 0.004 N·m·s |
| Cart | 1.0 kg, rail friction 0.5 N·s/m |
| Motor | ±15 N, updated at 50 Hz (zero-order hold) |
| Rail | ±2.4 m with hard end stops |
| Integrator | RK4 at 400 Hz, full nonlinear cart–pole equations |

Hitting an end stop ends a training episode with a penalty. On the web page the
cart bounces off a rubber stop instead, and the pole gets the matching angular
impulse.

## The agent

- **Observation (6):** cart position, cart velocity, cos θ, sin θ, angular velocity, last action
- **Action (1):** motor force as a fraction of 15 N
- **Policy:** MLP 6 → 64 → 64 → 1, tanh throughout (≈4.7k parameters); critic 6 → 128 → 128 → 1
- **Reward per step, in [0, 1]:** `upright² × centred × gentle × still`. It reaches 1
  only when the pole is vertical, the cart sits at the middle of the rail,
  nothing moves and the motor is idle. Hitting an end stop costs −20.
- **Robustness:** each episode jitters the physics (pole length ±15%, pole mass
  ±30%, cart mass ±20%, friction ×½ to ×2), and the cart is shoved at random.

## Results

From `training/evaluate.py` on the shipped model (30M PPO steps plus 6M steps of fine-tuning):

256 random starts per row (any angle, ±8 rad/s spin, anywhere on the rail), deterministic policy, 20 s each.
"balanced" means within 5° for the whole final 5 s. The angle, drift and force columns are averages over those 5 s.

```
case                          balanced  crashed  swing-up  worst   angle  drift  force
nominal                         100.0%     0.0%     1.12s   3.3s  0.000° 0.009m  0.00N
nominal + random shoves          55.9%     0.0%     1.12s   3.3s  0.438° 0.013m  0.16N
pole 30% longer                  99.2%     0.8%     1.27s   3.8s  0.000° 0.009m  0.00N
pole 30% shorter                100.0%     0.0%     1.15s   2.6s  0.000° 0.009m  0.00N
pole twice as heavy             100.0%     0.0%     1.11s   3.3s  0.000° 0.009m  0.00N
cart twice as heavy             100.0%     0.0%     1.22s   3.4s  0.000° 0.009m  0.00N
friction x3                     100.0%     0.0%     1.09s   3.5s  0.000° 0.009m  0.00N
```

- From hanging straight down or spinning fast, it is upright in a median of **1.1 s**, and within 3.3 s at worst.
- At rest it holds the pole dead vertical, with the cart **within 1 cm of centre** and the motor idle.
- It copes with physics well outside its training range (pole ±30% length, double masses, triple friction).
- In the "random shoves" row it is still being shoved during the last 5 s, so it is often mid-recovery. It never crashes.


## The double pendulum

| | |
|---|---|
| Links | two uniform rods, 0.6 m and 0.2 kg each, friction 0.002 N·m·s at both joints |
| Cart / motor / rail | 1.0 kg, ±25 N, ±2.4 m |
| Observation (9) | cart position and speed, cos/sin of both angles, both spins, last action |
| Policy | MLP 9 → 128 → 128 → 1, tanh (17,921 parameters); critic 9 → 256 → 256 → 1 |
| Training | PPO, 150M steps (about 35 simulated days), γ = 0.995, physics jitter ±8–15% |

The equations come from the Lagrangian of the cart and both rods; see
`training/env_double.py`. With friction off, the simulation conserves energy to
about 1 part in 10⁵ over 5 s of chaotic motion.

Swing-up only started working after about 39M steps. Before that, the agent
could only catch and balance starts that were already near upright.

```
case                          balanced  crashed  swing-up  worst   angle  drift  force
nominal                          98.8%     1.2%     2.20s   6.1s  0.000° 0.004m  0.00N
nominal + random shoves          88.3%     1.2%     2.21s   6.5s  0.501° 0.008m  0.20N
pole 30% longer                  14.5%     3.1%     0.57s  19.3s  0.677° 0.010m  0.28N
pole 30% shorter                 21.9%     0.8%     0.70s  19.7s  0.040° 0.004m  0.00N
pole twice as heavy              98.4%     1.2%     2.61s  11.7s  0.017° 0.004m  0.00N
cart twice as heavy              36.3%     4.3%     3.36s  17.0s  0.147° 0.004m  0.01N
friction x3                      99.2%     0.8%     2.22s   7.8s  0.000° 0.004m  0.00N
```

- From hanging down it is upright in a median of **2.2 s** (6.1 s at worst), and it then holds both links vertical with the cart **4 mm** from centre.
- It is much less forgiving than the single. It handles triple friction and a double-weight pole, but a pole 30% longer or shorter, or a heavy cart, mostly defeats it. The training jitter was kept small (±8% lengths) so that it could learn swing-up at all.
- On the page an end stop doesn't end the run, so the ~1% of starts that hit one usually recover.

## Wind and ground shake

Optional disturbances, added in `training/disturb.py` (Python) and `physics.js` (browser):

- **Wind**: a sideways load in N per metre of rod, spread along every rod. It follows a gusty
  random process (1.5 s correlation) around a mean that shifts every ~8 s. In a steady breeze
  the only way to stay put is to lean into the wind while the motor pushes back.
- **Ground shake**: the whole rail moves sideways, as the sum of four random sine waves
  between 0.3 and 1.5 Hz with a peak of 4 m/s² (3 m/s² for the double) at strength 1. In the
  rail's frame every body feels a pseudo-force −m·a.

Both enter the equations of motion as generalised forces. With friction off, a constant wind
or a constant rail acceleration conserves energy plus the potential of that force, to 7
decimal places. The JS and Python simulations still agree to about 10⁻⁶ with both switched
on.

The agent's inputs **never include** the wind or the shake: it has to sense them through
the pole's motion. Each brain was fine-tuned from its calm-trained version with a random
mix per episode (25% calm, 35% wind, 25% shake, 15% both, strength 0.2–1.2×). That was
25M more steps for the single and 40M for the double. The page keeps both versions on the
Brain slider ("calm" and "wind & shake") so you can compare them.

**Single, wind & shake brain** (256 random starts per row; "stays up" = within 30° for the
final 5 s; angle, drift and force are averages over that window):

```
case                          stays up within 5°  crashed  swing-up  worst   angle  drift  force
nominal                         100.0%    100.0%     0.0%     1.28s   5.2s  0.000° 0.032m  0.00N
nominal + random shoves          92.6%     66.0%     0.0%     1.30s   5.2s  2.101° 0.041m  0.64N
wind, strength 0.5              100.0%      0.4%     0.0%     1.37s   7.9s  9.089° 0.303m  1.79N
ground shake, strength 0.5      100.0%      3.5%     0.0%     1.31s   5.1s  4.533° 0.082m  2.02N
wind + shake, strength 0.5       99.2%      0.0%     0.0%     1.39s   6.7s  9.944° 0.309m  2.71N
wind, strength 1                 57.8%      0.0%     0.0%     1.31s  10.0s 13.538° 0.440m  3.55N
ground shake, strength 1        100.0%      0.4%     0.0%     1.27s   5.0s  9.112° 0.162m  4.29N
wind + shake, strength 1         26.6%      0.0%     0.0%     1.29s   9.2s 13.781° 0.398m  5.41N
wind + shake, strength 1.5        1.6%      0.0%     0.0%     1.30s  11.7s 13.107° 0.308m  5.68N
pole 30% longer                  52.3%      0.0%     0.0%     1.21s   6.7s 15.353° 0.308m 11.51N
pole 30% shorter                100.0%    100.0%     0.0%     1.18s   3.7s  0.000° 0.032m  0.00N
pole twice as heavy             100.0%    100.0%     0.0%     1.24s   4.6s  0.000° 0.032m  0.00N
cart twice as heavy              87.1%     84.8%     0.0%     1.57s   5.8s  0.443° 0.032m  0.04N
friction x3                     100.0%    100.0%     0.0%     1.28s   4.9s  0.000° 0.032m  0.00N
```

**Double, wind & shake brain:**

```
case                          stays up within 5°  crashed  swing-up  worst   angle  drift  force
nominal                          99.2%     99.2%     0.8%     2.35s   7.0s  0.000° 0.053m  0.00N
nominal + random shoves          97.7%     84.0%     0.8%     2.35s   6.7s  0.841° 0.053m  0.39N
wind, strength 0.5               99.6%      2.0%     0.4%     2.32s   8.5s  5.102° 0.248m  1.02N
ground shake, strength 0.5       99.6%      4.7%     0.4%     2.42s   7.5s  4.817° 0.118m  2.17N
wind + shake, strength 0.5       99.2%      0.0%     0.8%     2.39s   7.3s  7.015° 0.277m  2.36N
wind, strength 1                 98.4%      0.0%     0.8%     2.45s  10.5s  9.912° 0.484m  2.04N
ground shake, strength 1         99.2%      0.0%     0.4%     2.38s   7.3s  9.050° 0.218m  4.35N
wind + shake, strength 1         64.1%      0.0%     2.3%     2.54s   9.9s 12.129° 0.489m  4.77N
wind + shake, strength 1.5        4.7%      0.0%    15.6%     2.80s  17.3s 13.596° 0.556m  5.57N
pole 30% longer                   3.9%      0.0%     1.6%     0.22s   3.7s  7.527° 0.149m  4.23N
pole 30% shorter                  1.2%      1.2%     3.9%     1.10s  17.7s  0.000° 0.053m  0.00N
pole twice as heavy              74.6%     74.2%     0.4%     2.80s  14.4s  0.133° 0.053m  0.00N
cart twice as heavy               2.7%      2.3%    14.8%     3.88s  18.1s  0.796° 0.054m  0.54N
friction x3                      99.6%     99.6%     0.4%     2.34s   6.0s  0.000° 0.053m  0.00N
```

What changed compared with the calm-trained brains:

- **Wind no longer blows it off the rail.** Single at 0.5×: crashes 23% → 0%. Double at 1×:
  crashes 20% → 1%, and it stays up 73% → 98% of the time.
- **It costs a little calm precision.** The cart rests 3–5 cm off-centre instead of 1 cm or
  less, and the slowest swing-ups take longer.
- **It tolerates changed physics less well.** For example, the single now struggles with a
  pole 30% longer. The agent can't tell wind from shake from a heavier pole, so one reflex
  has to serve them all. Giving it a short memory of recent inputs is the obvious next step.
- At full strength, wind gusts approach the pole's own weight. Resisting them needs a lean of
  more than 30°, so "stays up" there is limited by physics as well as by the brain.

## Files

```
index.html            the page (simulation, drawing, controls)
physics.js            the world: same equations as training/env.py
trainer.js            PPO in plain JavaScript (Web Worker) for live training
weights.json          trained single-pendulum policy at 8 points during training
weights_double.json   trained double-pendulum policy at 8 points during training
training/env.py       vectorised numpy world (single)
training/env_double.py  vectorised numpy world (double)
training/disturb.py   wind and rail-shake generator shared by both worlds
training/train.py     PPO (PyTorch)
training/evaluate.py  stress tests from random starts, including changed physics
training/parity_test.py  checks that the JS and Python simulations match
```

## Training it yourself

```bash
cd training
pip install -r requirements.txt
python train.py --steps 3e7 --out runs/main                      # about 15 min on 4 CPU cores
python train.py --init runs/main/ckpt.pt --steps 6e6 --lr 1e-4 \
                --snapshots 1.0 --step-offset 3e7 --out runs/finetune
python evaluate.py runs/finetune/ckpt.pt
cp runs/finetune/weights.json ..    # or merge snapshots as done for the shipped file
python parity_test.py

# double pendulum: about 90 min on 4 CPU cores
python train.py --system double --steps 1.5e8 --envs 256 --minibatch 8192 --gamma 0.995 \
                --snapshots 0,0.01,0.03,0.08,0.2,0.4,0.7,1.0 --out runs/double
python evaluate.py runs/double/ckpt.pt --system double
cp runs/double/weights.json ../weights_double.json
python parity_test.py --system double

# wind & shake fine-tunes
python train.py --disturb --init runs/finetune/ckpt.pt --init-std 0.2 --steps 2.5e7 --lr 2e-4 \
                --snapshots 1.0 --step-offset 3.6e7 --out runs/disturb
python train.py --system double --disturb --init runs/double/ckpt.pt --init-std 0.15 --steps 4e7 \
                --lr 1.5e-4 --envs 256 --minibatch 8192 --gamma 0.995 --snapshots 1.0 \
                --step-offset 1.5e8 --out runs/double_disturb
```
