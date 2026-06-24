# module2_localization/kalman.py — Extended Kalman Filter for GNSS positioning
import numpy as np
from .base import LocalizationBase
from .factory import LocalizationFactory


@LocalizationFactory.register("ekf")
class EKF(LocalizationBase):
    """8-state EKF: position(3), clock_bias, velocity(3), clock_drift"""
    
    def __init__(self, config=None, name="ekf"):
        super().__init__(config, name)
        self.process_noise_pos = (config or {}).get("process_noise_pos", 1.0)
        self.process_noise_vel = (config or {}).get("process_noise_vel", 0.1)
        self.meas_noise = (config or {}).get("measurement_noise", 5.0)
        self._state = None
        self._P = None
        self._initialized = False
    
    def solve(self, observations, sv_positions, sv_systems=None, additional_info=None):
        self.validate_input(observations, sv_positions)
        obs = np.asarray(observations).flatten()
        svp = np.asarray(sv_positions)
        N = len(obs)
        
        if not self._initialized:
            # Initialize with LS solution
            pos, clk, _ = self._ls_init(obs, svp)
            self._state = np.array([pos[0], pos[1], pos[2], clk, 0.0, 0.0, 0.0, 0.0])
            self._P = np.eye(8) * 100.0
            self._initialized = True
        
        dt = 1.0  # 1 second between epochs (typical GNSS)
        
        # Predict step
        F = np.eye(8)
        F[0, 4] = dt; F[1, 5] = dt; F[2, 6] = dt; F[3, 7] = dt
        Q_pos = np.eye(4) * self.process_noise_pos * dt
        Q_vel = np.eye(4) * self.process_noise_vel * dt
        Q = np.block([[Q_pos, np.zeros((4,4))], [np.zeros((4,4)), Q_vel]])
        self._state = F @ self._state
        self._P = F @ self._P @ F.T + Q
        
        # Update step
        pos = self._state[:3]
        clk = self._state[3]
        dists = np.linalg.norm(svp - pos, axis=1)
        h = dists + clk
        
        H = np.zeros((N, 8))
        for i in range(N):
            H[i, :3] = (pos - svp[i]) / dists[i]
        H[:, 3] = 1.0
        
        R = np.eye(N) * self.meas_noise
        y = obs - h
        
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._state += K @ y
        self._P = (np.eye(8) - K @ H) @ self._P
        
        details = {"converged": True, "iterations": 1, "residuals": y,
                   "covariance": self._P[:4, :4]}
        return self._state[:3], self._state[3], details
    
    def _ls_init(self, obs, svp):
        N = len(obs)
        x0 = np.mean(svp, axis=0)
        x0 = x0 / np.linalg.norm(x0) * 6371.0
        pos, clk = x0.copy(), 0.0
        for _ in range(10):
            dists = np.linalg.norm(svp - pos, axis=1)
            G = np.zeros((N, 4))
            G[:, :3] = (pos - svp) / dists[:, None]
            G[:, 3] = 1.0
            residuals = obs - (dists + clk)
            try:
                delta = np.linalg.solve(G.T @ G, G.T @ residuals)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(G.T @ G, G.T @ residuals, rcond=None)[0]
            pos += delta[:3]; clk += delta[3]
            if np.linalg.norm(delta) < 1e-6:
                break
        return pos, clk, residuals
