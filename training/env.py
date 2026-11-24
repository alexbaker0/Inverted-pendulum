"""Cart-and-pole world, vectorised over N independent copies with numpy.

The browser simulation (../physics.js) implements exactly the same equations,
constants and observation/reward functions, so a policy trained here behaves
the same on the web page.

Coordinates
    x      cart position on the rail, metres, + is right
    theta  pole angle from straight UP, radians, + leans right (clockwise)
           theta = 0 is balanced, theta = pi is hanging down

Physics
    Uniform rod on a frictional pivot, cart on a frictional rail, full nonlinear
    rigid-body equations (Barto et al. 1983, with the sign correction from
    Florian 2007, plus viscous friction on both joints). Integrated with RK4 at
    400 Hz. The controller acts at 50 Hz and holds its force between decisions
    (zero-order hold), like a real motor driver.

    The rail is finite: hitting an end stop ends the episode in training.
"""

import numpy as np

from disturb import DisturbanceVec

WIND_SCALE = 1.0         # N per metre of pole at strength 1
GROUND_SCALE = 4.0       # m/s^2 peak rail acceleration at strength 1

# ---- physical constants (nominal) -------------------------------------------
G = 9.81
CART_MASS = 1.0          # kg
POLE_MASS = 0.2          # kg
POLE_LENGTH = 1.0        # m, full length of the rod (pivot to tip)
CART_FRICTION = 0.5      # N per (m/s), rail + belt drag
POLE_FRICTION = 0.004    # N m per (rad/s), pivot bearing
MAX_FORCE = 15.0         # N, motor limit
RAIL_HALF = 2.4          # m, end stops at +/- this

PHYS_DT = 0.0025         # s, RK4 step (400 Hz)
CTRL_DT = 0.02           # s, controller period (50 Hz)
SUBSTEPS = int(round(CTRL_DT / PHYS_DT))

EPISODE_STEPS = 1000     # 20 s
OBS_DIM = 6
ACT_DIM = 1


def derivs(s, force, p, w=0.0, ag=0.0):
    """Time derivative of state s=[x, xd, th, thd] (each shape (N,)).

    w: wind load on the pole, N per metre (+ right). ag: rail acceleration.
    """
    x, xd, th, thd = s
    mc, mp, l, bc, bp = p  # l is HALF length (pivot to centre of mass)
    total = mc + mp
    sin, cos = np.sin(th), np.cos(th)
    fw = w * 2 * l                                    # total wind force on the rod
    qx = fw - total * ag                              # generalised force on x
    qth = (fw - mp * ag) * l * cos                    # ... and on theta
    tmp = (force + qx + mp * l * thd * thd * sin - bc * xd) / total
    thdd = (G * sin + qth / (mp * l) - cos * tmp - bp * thd / (mp * l)) / (
        l * (4.0 / 3.0 - mp * cos * cos / total))
    xdd = tmp - mp * l * thdd * cos / total
    return np.stack([xd, xdd, thd, thdd])


def rk4(s, force, p, dt, w=0.0, ag=0.0):
    k1 = derivs(s, force, p, w, ag)
    k2 = derivs(s + 0.5 * dt * k1, force, p, w, ag)
    k3 = derivs(s + 0.5 * dt * k2, force, p, w, ag)
    k4 = derivs(s + dt * k3, force, p, w, ag)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def observe(s, last_u):
    x, xd, th, thd = s
    return np.stack([
        x / RAIL_HALF,
        xd / 3.0,
        np.cos(th),
        np.sin(th),
        thd / 10.0,
        last_u,
    ], axis=-1).astype(np.float32)


def reward(s, u):
    """In [0, 1]. 1 only when upright, centred, motionless and not pushing.

    Multiplicative, so every factor matters, but no factor alone can zero it
    except being upside-down: the agent always has a gradient towards upright.
    """
    x, xd, th, thd = s
    upright = (1.0 + np.cos(th)) / 2.0
    upright = upright ** 2                          # sharpen near the top
    centred = 0.5 + 0.3 * np.exp(-(x / 0.8) ** 2) + 0.2 * np.exp(-(x / 0.1) ** 2)
    gentle = 0.8 + 0.2 * (1.0 - u * u)
    still = 0.5 + 0.5 * np.exp(-(thd / 2.0) ** 2 - (xd / 1.0) ** 2)
    return upright * centred * gentle * still


class CartPoleVec:
    """N copies of the world. Actions are in [-1, 1] (fraction of MAX_FORCE)."""

    def __init__(self, n, seed=0, randomise=True, pushes=True, disturb=False):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.dist = DisturbanceVec(n, self.rng, WIND_SCALE, GROUND_SCALE, enabled=disturb)
        self.randomise = randomise
        self.pushes = pushes
        self.s = np.zeros((4, n))
        self.p = np.zeros((5, n))
        self.t = np.zeros(n, dtype=np.int64)
        self.last_u = np.zeros(n)
        self.push_force = np.zeros(n)
        self.push_left = np.zeros(n, dtype=np.int64)
        self.reset_all()

    # -- episode management --------------------------------------------------
    def _sample_params(self, idx):
        k = len(idx)
        r = self.rng.uniform
        if self.randomise:
            mc = CART_MASS * r(0.8, 1.2, k)
            mp = POLE_MASS * r(0.7, 1.3, k)
            l = POLE_LENGTH / 2 * r(0.85, 1.15, k)
            bc = CART_FRICTION * r(0.5, 2.0, k)
            bp = POLE_FRICTION * r(0.5, 2.0, k)
        else:
            mc = np.full(k, CART_MASS)
            mp = np.full(k, POLE_MASS)
            l = np.full(k, POLE_LENGTH / 2)
            bc = np.full(k, CART_FRICTION)
            bp = np.full(k, POLE_FRICTION)
        self.p[:, idx] = np.stack([mc, mp, l, bc, bp])

    def _sample_state(self, idx):
        k = len(idx)
        r = self.rng.uniform
        x = r(-1.5, 1.5, k)
        xd = r(-1.0, 1.0, k)
        th = r(-np.pi, np.pi, k)
        thd = r(-8.0, 8.0, k)
        # A quarter of episodes start near the top so balancing gets practice
        # from the very beginning of training.
        near = self.rng.random(k) < 0.25
        th[near] = r(-0.5, 0.5, near.sum())
        thd[near] = r(-2.0, 2.0, near.sum())
        self.s[:, idx] = np.stack([x, xd, th, thd])

    def reset_idx(self, idx):
        if len(idx) == 0:
            return
        self._sample_params(idx)
        self._sample_state(idx)
        self.dist.reset_idx(idx)
        self.t[idx] = 0
        self.last_u[idx] = 0
        self.push_left[idx] = 0
        self.push_force[idx] = 0

    def reset_all(self):
        self.reset_idx(np.arange(self.n))
        return self.obs()

    def obs(self):
        return observe(self.s, self.last_u)

    # -- stepping ------------------------------------------------------------
    def step(self, a):
        u = np.clip(np.asarray(a, dtype=np.float64).reshape(self.n), -1, 1)
        force = u * MAX_FORCE

        if self.pushes:
            # Occasional shove on the cart, like someone nudging the rig.
            start = (self.push_left == 0) & (self.rng.random(self.n) < 0.004)
            k = start.sum()
            if k:
                self.push_force[start] = self.rng.uniform(-25, 25, k)
                self.push_left[start] = self.rng.integers(3, 10, k)
            force = force + np.where(self.push_left > 0, self.push_force, 0.0)
            self.push_left = np.maximum(self.push_left - 1, 0)

        w = self.dist.step_wind(CTRL_DT)
        t0 = self.t * CTRL_DT
        for k in range(SUBSTEPS):
            ag = self.dist.ground_acc(t0 + k * PHYS_DT)
            self.s = rk4(self.s, force, self.p, PHYS_DT, w, ag)
        self.s[2] = wrap(self.s[2])
        self.t += 1

        r = reward(self.s, u)
        self.last_u = u

        crashed = np.abs(self.s[0]) > RAIL_HALF
        r = np.where(crashed, -20.0, r)
        truncated = (self.t >= EPISODE_STEPS) & ~crashed
        done = crashed | truncated

        # Value of the final state for truncated episodes (bootstrapping).
        final_obs = self.obs()
        info = {"crashed": crashed, "truncated": truncated,
                "final_obs": final_obs}
        self.reset_idx(np.nonzero(done)[0])
        return self.obs(), r.astype(np.float32), crashed, truncated, info
