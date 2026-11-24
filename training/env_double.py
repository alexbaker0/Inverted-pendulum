"""Cart with a DOUBLE pendulum (two uniform rods, hinged end to end), vectorised.

Same conventions as env.py: angles from straight up, + leans right; theta = 0
for both links is fully upright. ../physics.js implements identical equations.

Equations of motion come from the Lagrangian of a cart (mc) carrying rod 1
(m1, L1) on a pivot, with rod 2 (m2, L2) hinged at rod 1's tip:

    M(q) q'' = f(q, q', F)        q = [x, th1, th2]

    M = | mc+m1+m2        a cos th1          b cos th2        |
        | a cos th1       J1                 c cos(th1-th2)   |
        | b cos th2       c cos(th1-th2)     J2               |

    a = m1 l1 + m2 L1,  b = m2 l2,  c = m2 L1 l2   (l = half length)
    J1 = m1 l1^2 + I1 + m2 L1^2,  J2 = m2 l2^2 + I2,  I = m L^2 / 12

    f = [ F + a sin th1 th1'^2 + b sin th2 th2'^2 - bc x'
          -c sin(th1-th2) th2'^2 + a g sin th1 - b1 th1' + b2 (th2'-th1')
           c sin(th1-th2) th1'^2 + b g sin th2 - b2 (th2'-th1') ]
"""

import numpy as np

from disturb import DisturbanceVec

WIND_SCALE = 0.8         # N per metre of rod at strength 1
GROUND_SCALE = 3.0       # m/s^2 peak rail acceleration at strength 1

G = 9.81
CART_MASS = 1.0
POLE1_MASS = 0.2
POLE2_MASS = 0.2
POLE1_LENGTH = 0.6
POLE2_LENGTH = 0.6
CART_FRICTION = 0.5
JOINT_FRICTION = 0.002   # both joints, N m per rad/s
MAX_FORCE = 25.0
RAIL_HALF = 2.4

PHYS_DT = 0.0025
CTRL_DT = 0.02
SUBSTEPS = int(round(CTRL_DT / PHYS_DT))
EPISODE_STEPS = 1000
OBS_DIM = 9
ACT_DIM = 1

# param rows: mc, m1, m2, L1, L2, bc, bj


def derivs(s, force, p, w=0.0, ag=0.0):
    x, th1, th2, xd, t1d, t2d = s
    mc, m1, m2, L1, L2, bc, bj = p
    l1, l2 = L1 / 2, L2 / 2
    a = m1 * l1 + m2 * L1
    b = m2 * l2
    c = m2 * L1 * l2
    J1 = m1 * l1 * l1 + m1 * L1 * L1 / 12 + m2 * L1 * L1
    J2 = m2 * l2 * l2 + m2 * L2 * L2 / 12
    s1, c1, s2, c2 = np.sin(th1), np.cos(th1), np.sin(th2), np.cos(th2)
    s12, c12 = np.sin(th1 - th2), np.cos(th1 - th2)

    M11 = mc + m1 + m2
    M12 = a * c1
    M13 = b * c2
    M22 = J1
    M23 = c * c12
    M33 = J2
    rel = t2d - t1d
    # wind (w N/m along each rod, resultant at its centre) and the rail's
    # pseudo-force -m*ag on every body, as generalised forces
    h1 = w * L1 - m1 * ag                       # net sideways load on rod 1's centre
    h2 = w * L2 - m2 * ag                       # ... and on rod 2's centre
    q1 = h1 + h2 - mc * ag
    q2 = (h1 * l1 + h2 * L1) * c1
    q3 = h2 * l2 * c2
    f1 = force + q1 + a * s1 * t1d * t1d + b * s2 * t2d * t2d - bc * xd
    f2 = q2 - c * s12 * t2d * t2d + a * G * s1 - bj * t1d + bj * rel
    f3 = q3 + c * s12 * t1d * t1d + b * G * s2 - bj * rel

    # symmetric 3x3 solve by Cramer's rule (fast, vectorised)
    C11 = M22 * M33 - M23 * M23
    C12 = M13 * M23 - M12 * M33
    C13 = M12 * M23 - M13 * M22
    C22 = M11 * M33 - M13 * M13
    C23 = M12 * M13 - M11 * M23
    C33 = M11 * M22 - M12 * M12
    det = M11 * C11 + M12 * C12 + M13 * C13
    xdd = (C11 * f1 + C12 * f2 + C13 * f3) / det
    t1dd = (C12 * f1 + C22 * f2 + C23 * f3) / det
    t2dd = (C13 * f1 + C23 * f2 + C33 * f3) / det
    return np.stack([xd, t1d, t2d, xdd, t1dd, t2dd])


def rk4(s, force, p, dt, w=0.0, ag=0.0):
    k1 = derivs(s, force, p, w, ag)
    k2 = derivs(s + 0.5 * dt * k1, force, p, w, ag)
    k3 = derivs(s + 0.5 * dt * k2, force, p, w, ag)
    k4 = derivs(s + dt * k3, force, p, w, ag)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def observe(s, last_u):
    x, th1, th2, xd, t1d, t2d = s
    return np.stack([
        x / RAIL_HALF, xd / 3.0,
        np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2),
        t1d / 10.0, t2d / 15.0,
        last_u,
    ], axis=-1).astype(np.float32)


def reward(s, u):
    x, th1, th2, xd, t1d, t2d = s
    up1 = (1 + np.cos(th1)) / 2
    up2 = (1 + np.cos(th2)) / 2
    upright = (up1 * up2) ** 2
    centred = 0.5 + 0.3 * np.exp(-(x / 0.8) ** 2) + 0.2 * np.exp(-(x / 0.1) ** 2)
    gentle = 0.8 + 0.2 * (1.0 - u * u)
    still = 0.5 + 0.5 * np.exp(-(t1d / 2.0) ** 2 - (t2d / 2.0) ** 2 - xd ** 2)
    return upright * centred * gentle * still


def energy(s, p):
    """Total mechanical energy (for testing)."""
    x, th1, th2, xd, t1d, t2d = s
    mc, m1, m2, L1, L2, bc, bj = p
    l1, l2 = L1 / 2, L2 / 2
    v1x = xd + l1 * np.cos(th1) * t1d
    v1y = -l1 * np.sin(th1) * t1d
    v2x = xd + L1 * np.cos(th1) * t1d + l2 * np.cos(th2) * t2d
    v2y = -L1 * np.sin(th1) * t1d - l2 * np.sin(th2) * t2d
    T = (0.5 * mc * xd ** 2 + 0.5 * m1 * (v1x ** 2 + v1y ** 2) + 0.5 * m2 * (v2x ** 2 + v2y ** 2)
         + 0.5 * (m1 * L1 ** 2 / 12) * t1d ** 2 + 0.5 * (m2 * L2 ** 2 / 12) * t2d ** 2)
    V = G * (m1 * l1 * np.cos(th1) + m2 * (L1 * np.cos(th1) + l2 * np.cos(th2)))
    return T + V


class DoublePoleVec:
    def __init__(self, n, seed=0, randomise=True, pushes=True, near_frac=0.3, disturb=False):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.dist = DisturbanceVec(n, self.rng, WIND_SCALE, GROUND_SCALE, enabled=disturb)
        self.randomise = randomise
        self.pushes = pushes
        self.near_frac = near_frac
        self.s = np.zeros((6, n))
        self.p = np.zeros((7, n))
        self.t = np.zeros(n, dtype=np.int64)
        self.last_u = np.zeros(n)
        self.push_force = np.zeros(n)
        self.push_left = np.zeros(n, dtype=np.int64)
        self.reset_all()

    def _sample_params(self, idx):
        k = len(idx)
        r = (lambda lo, hi: self.rng.uniform(lo, hi, k)) if self.randomise else (lambda lo, hi: np.ones(k))
        self.p[:, idx] = np.stack([
            CART_MASS * r(0.85, 1.15), POLE1_MASS * r(0.85, 1.15), POLE2_MASS * r(0.85, 1.15),
            POLE1_LENGTH * r(0.92, 1.08), POLE2_LENGTH * r(0.92, 1.08),
            CART_FRICTION * r(0.5, 2.0), JOINT_FRICTION * r(0.5, 2.0)])

    def _sample_state(self, idx):
        k = len(idx)
        r = lambda lo, hi, m=k: self.rng.uniform(lo, hi, m)
        x, xd = r(-1.5, 1.5), r(-1, 1)
        th1, th2 = r(-np.pi, np.pi), r(-np.pi, np.pi)
        t1d, t2d = r(-6, 6), r(-8, 8)
        near = self.rng.random(k) < self.near_frac
        m = near.sum()
        th1[near], th2[near] = r(-0.3, 0.3, m), r(-0.3, 0.3, m)
        t1d[near], t2d[near] = r(-1.5, 1.5, m), r(-1.5, 1.5, m)
        self.s[:, idx] = np.stack([x, th1, th2, xd, t1d, t2d])

    def reset_idx(self, idx):
        if len(idx) == 0:
            return
        self._sample_params(idx)
        self._sample_state(idx)
        self.dist.reset_idx(idx)
        self.t[idx] = 0
        self.last_u[idx] = 0
        self.push_left[idx] = 0

    def reset_all(self):
        self.reset_idx(np.arange(self.n))
        return self.obs()

    def obs(self):
        return observe(self.s, self.last_u)

    def step(self, a):
        u = np.clip(np.asarray(a, dtype=np.float64).reshape(self.n), -1, 1)
        force = u * MAX_FORCE
        if self.pushes:
            start = (self.push_left == 0) & (self.rng.random(self.n) < 0.003)
            k = start.sum()
            if k:
                self.push_force[start] = self.rng.uniform(-20, 20, k)
                self.push_left[start] = self.rng.integers(2, 6, k)
            force = force + np.where(self.push_left > 0, self.push_force, 0.0)
            self.push_left = np.maximum(self.push_left - 1, 0)
        w = self.dist.step_wind(CTRL_DT)
        t0 = self.t * CTRL_DT
        for k in range(SUBSTEPS):
            ag = self.dist.ground_acc(t0 + k * PHYS_DT)
            self.s = rk4(self.s, force, self.p, PHYS_DT, w, ag)
        self.s[1] = wrap(self.s[1])
        self.s[2] = wrap(self.s[2])
        self.t += 1
        r = reward(self.s, u)
        self.last_u = u
        crashed = np.abs(self.s[0]) > RAIL_HALF
        r = np.where(crashed, -20.0, r)
        truncated = (self.t >= EPISODE_STEPS) & ~crashed
        done = crashed | truncated
        final_obs = self.obs()
        info = {"crashed": crashed, "truncated": truncated, "final_obs": final_obs}
        self.reset_idx(np.nonzero(done)[0])
        return self.obs(), r.astype(np.float32), crashed, truncated, info
