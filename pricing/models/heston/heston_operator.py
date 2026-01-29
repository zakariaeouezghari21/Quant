import numpy as np 
from operators.base_operator import BaseFDOperator

class HestonFDOperator(BaseFDOperator):
    """
    L = L_s + L_v + L_sv
    """

    def __init__(self, grid_s, grid_v, heston_params, r, q, rho):
        self.s = grid_s
        self.v = grid_v

        self.kappa = heston_params['kappa']
        self.theta = heston_params['theta']
        self.sigma = heston_params['sigma']

        self.r = r
        self.q = q
        self.rho = rho

        self.ds = self.s[1] - self.s[0]
        self.dv = self.v[1] - self.v[0]
        
    def set_time(self, t1, t2):
        self.dt = t2 - t1

    def apply(self, u):
        return self.apply_direction(0, u) + self.apply_direction(1, u) + self.apply_mixed(u)
    
    def apply_direction(self, direction, u):
        if direction == 0:
            return self._apply_s(u)
        elif direction == 1:
            return self._apply_v(u)
        else:
            raise ValueError("Invalid direction.")
        
    def _apply_s(self, u):
        du_ds = (u[2:, :] - u[:-2, :]) / (2 * self.ds)
        d2u_ds2 = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / (self.ds ** 2)

        s_mid = self.s[1: -1, None]
        v_mid = self.v[None, :]

        term = 0.5 * v_mid * s_mid**2 * d2u_ds2 + (self.r -self.q) * s_mid * du_ds - self.r * u[1:-1, :]
        return self._pad(term)
    
    def _apply_v(self, u):
        du_dv = (u[:, 2:] - u[:, :-2]) / (2 * self.dv)
        d2u_dv2 = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (self.dv ** 2)

        v_mid = self.v[None, 1:-1]

        term = 0.5 * self.sigma**2 * v_mid * d2u_dv2 + self.kappa * (self.theta - v_mid) * du_dv
        return self._pad(term, axis=1)
    
    def apply_mixed(self, u):
        d2u = (u[2:, 2:] - u[2:, :-2] - u[:-2, 2:] + u[:-2, :-2]) / (4 * self.ds * self.dv)

        s_mid = self.s[1:-1, None]
        v_mid = self.v[None, 1:-1]

        term = self.rho * self.sigma * v_mid * s_mid * d2u
        return self._pad(term)
    
    def _pad(self, core, axis=0):
        out = np.zeros((len(self.s), len(self.v)))
        if axis == 0:
            out[1:-1, :] = core
        else: 
            out[:, 1:-1] = core
        return out
