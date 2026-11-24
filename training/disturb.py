"""Random disturbances shared by both worlds: wind on the pole(s) and a shaking
rail. The agent never observes them directly; it has to feel them through the
pole's motion, as a real controller would.

Wind
    A horizontal load of w newtons per metre of rod, acting along every rod
    (resultant at each rod's centre). w follows a gusty Ornstein-Uhlenbeck
    process around a mean that occasionally shifts, so there are both steady
    breezes (the pole must lean into them while the motor pushes back) and
    gusts.

Ground shake
    The rail itself moves sideways, g(t) = sum_k A_k sin(2 pi f_k t + phi_k),
    a mix of four random frequencies between 0.3 and 1.5 Hz. In the rail's
    frame every body feels a pseudo-force -m g''(t). Cart position and the
    end stops are measured relative to the rail.

Generalised forces added to the equations of motion (single pendulum):
    Q_x  = F_wind - (mc + mp) a_g
    Q_th = (F_wind - mp a_g) l cos th          F_wind = w * L, l = L / 2
and analogously for each link of the double (see env_double.derivs).

At strength 1.0 the scales are WIND_SCALE N/m and GROUND_SCALE m/s^2; the
browser lets you go past what training covered.
"""

import numpy as np

TAU_WIND = 1.5          # s, gust correlation time
N_SINES = 4


class DisturbanceVec:
    def __init__(self, n, rng, wind_scale, ground_scale, enabled=True):
        self.n, self.rng = n, rng
        self.wind_scale, self.ground_scale = wind_scale, ground_scale
        self.enabled = enabled
        self.fixed = None          # (wind_on, ground_on, strength) to force a scenario
        self.wind_on = np.zeros(n, bool)
        self.ground_on = np.zeros(n, bool)
        self.strength = np.zeros(n)
        self.w = np.zeros(n)
        self.w_mean = np.zeros(n)
        self.freq = np.ones((N_SINES, n))
        self.acc_amp = np.zeros((N_SINES, n))
        self.phase = np.zeros((N_SINES, n))

    def reset_idx(self, idx):
        k = len(idx)
        if k == 0:
            return
        r = self.rng
        if self.fixed is not None:
            self.wind_on[idx], self.ground_on[idx], self.strength[idx] = self.fixed
        elif self.enabled:
            kind = r.random(k)                      # none 25%, wind 35%, ground 25%, both 15%
            self.wind_on[idx] = (kind >= 0.25) & (kind < 0.60) | (kind >= 0.85)
            self.ground_on[idx] = (kind >= 0.60)
            self.strength[idx] = r.uniform(0.2, 1.2, k)
        else:
            self.wind_on[idx] = False
            self.ground_on[idx] = False
            self.strength[idx] = 0
        W = self.wind_scale * self.strength[idx]
        self.w_mean[idx] = r.uniform(-0.6, 0.6, k) * W
        self.w[idx] = self.w_mean[idx] + 0.5 * W * r.standard_normal(k)
        self.freq[:, idx] = r.uniform(0.3, 1.5, (N_SINES, k))
        self.phase[:, idx] = r.uniform(0, 2 * np.pi, (N_SINES, k))
        a = r.uniform(0, 1, (N_SINES, k))
        a = a / a.sum(0, keepdims=True)              # amplitudes share the peak budget
        self.acc_amp[:, idx] = a * self.ground_scale * self.strength[idx]

    def step_wind(self, dt):
        """Advance the wind by one control period. Returns w (N/m) per env."""
        r = self.rng
        W = self.wind_scale * self.strength
        shift = r.random(self.n) < dt / 8.0          # mean wind changes every ~8 s
        self.w_mean = np.where(shift, r.uniform(-0.6, 0.6, self.n) * W, self.w_mean)
        self.w += (self.w_mean - self.w) * dt / TAU_WIND + \
            0.5 * W * np.sqrt(2 * dt / TAU_WIND) * r.standard_normal(self.n)
        self.w = np.clip(self.w, -2.5 * W, 2.5 * W)
        return np.where(self.wind_on, self.w, 0.0)

    def ground_acc(self, t):
        """Rail acceleration (m/s^2) at time t (s, per env array or scalar)."""
        ang = 2 * np.pi * self.freq * t + self.phase
        acc = -(self.acc_amp * np.sin(ang)).sum(0)
        return np.where(self.ground_on, acc, 0.0)
