"""
Abstract base class for all 3 application simulators.
Enforces interface contract — never duplicated across apps.
"""
from abc import ABC, abstractmethod
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from waveform_params import WaveformParams, SNRConfig


class BaseSimulator(ABC):
    """
    Abstract base for FiveGSimulator, UltrasoundSimulator, RadarSimulator.
    All shared logic (SNR, state export/load) lives here.
    """

    def __init__(self, params: WaveformParams):
        self.params = params
        self.snr_config = SNRConfig(snr_db=params.snr_db)
        self._running = False
        self._step_count = 0
        self._history: list[dict] = []

    @abstractmethod
    def initialize(self) -> None:
        """Set up all internal state. Called once before first run."""
        ...

    @abstractmethod
    def step(self) -> dict:
        """Advance simulation by one step; return state snapshot."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all mutable state to initial conditions."""
        ...

    @abstractmethod
    def update_params(self, params: WaveformParams) -> None:
        """Hot-reload parameters without full reset."""
        ...

    @abstractmethod
    def get_output(self) -> dict:
        """Return current full output for frontend consumption."""
        ...

    def apply_snr(self, signal: np.ndarray) -> np.ndarray:
        return self.snr_config.apply_to_signal(signal)

    def set_snr(self, snr_db: float) -> None:
        self.snr_config.set_snr(snr_db)
        self.params.snr_db = snr_db

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def export_state(self) -> dict:
        return {
            "step_count": self._step_count,
            "params": self.params.to_dict(),
            "snr_db": self.snr_config.snr_db,
            "output": self.get_output(),
        }

    def load_state(self, state: dict) -> None:
        self._step_count = state.get("step_count", 0)
        if "params" in state:
            self.params = WaveformParams.from_dict(state["params"])
        if "snr_db" in state:
            self.snr_config.set_snr(state["snr_db"])