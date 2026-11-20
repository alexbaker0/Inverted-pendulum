# Inverted Pendulum

A cart on a finite rail learns, with reinforcement learning (PPO), to swing a
pole up from **any** starting angle and spin and then hold it upright and still.

**Live:** https://alexbaker0.github.io/Inverted-pendulum/

On the page you can:

- watch the trained agent swing up and balance, from random, hanging, or fast-spinning starts
- grab the pole and throw it, or shove the cart (`←` `→`), and watch it recover
- switch the motor off to watch the pendulum swing freely, or drive the cart yourself
- scrub back through training to see the same network when it was clueless
- see the whole network wired up: every neuron and all 4,544 connections, with live signals, and hover to trace any neuron
- change the pole length, the masses and the friction beyond what it was trained on
- train a brand-new network from scratch in your browser and watch it learn

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


## Files

```
index.html            the page (simulation, drawing, controls)
physics.js            the world: same equations as training/env.py
trainer.js            PPO in plain JavaScript (Web Worker) for live training
weights.json          trained policy at 8 points during training
training/env.py       vectorised numpy world
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
python parity_test.py ../weights.json
```
