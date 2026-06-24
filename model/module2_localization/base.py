# module2_localization/base.py — Localization method base class (ABC)
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np


class LocalizationBase(ABC):
    """Abstract base class for all GNSS positioning methods.
    
    All positioning methods (LS, WLS, FG, etc.) must inherit this class
    and implement solve() with the unified interface.
    """
    
    def __init__(self, config: Dict = None, name: str = "unknown"):
        self.config = config or {}
        self.name = name
        self.converged = False
        self.iterations = 0
    
    @abstractmethod
    def solve(self,
              observations: np.ndarray,       # (N,) pseudorange observations (km)
              sv_positions: np.ndarray,       # (N, 3) satellite ECEF positions (km)
              sv_systems: np.ndarray = None,  # (N,) constellation IDs
              additional_info: Dict = None    # MoG output etc.
              ) -> Tuple[np.ndarray, float, Dict]:
        """Solve positioning problem.
        
        Args:
            observations: pseudorange measurements in km
            sv_positions: satellite ECEF positions in km
            sv_systems: constellation identifiers (optional)
            additional_info: dict with optional keys:
                - p_los: (N,) LOS probabilities [0,1]
                - sigma_nlos: (N,) NLOS error std (km)
                - mu_nlos: (N,) NLOS bias (km)
                - sigma_los: (N,) LOS error std (km)
        
        Returns:
            position: (3,) receiver ECEF position (km)
            clock_bias: receiver clock bias (km)
            details: dict with keys:
                - converged: bool
                - iterations: int
                - residuals: (N,) residual vector (km)
                - covariance: (4,4) covariance matrix (optional)
        """
        pass
    
    def get_name(self) -> str:
        return self.name
    
    def validate_input(self, observations, sv_positions):
        """Validate input dimensions."""
        obs = np.asarray(observations).flatten()
        svp = np.asarray(sv_positions)
        if len(obs) < 4:
            raise ValueError(f"Need >= 4 satellites, got {len(obs)}")
        if svp.ndim != 2 or svp.shape[0] != len(obs) or svp.shape[1] != 3:
            raise ValueError(f"sv_positions must be ({len(obs)}, 3), got {svp.shape}")
