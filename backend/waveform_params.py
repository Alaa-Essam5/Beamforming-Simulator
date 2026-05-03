"""
Shared data models for beamforming parameters.
All 7+ customizable parameters live here - never duplicated elsewhere.
"""
from dataclasses import dataclass, asdict, field
from typing import Literal
import math


WINDOW_TYPES = Literal["rectangular", "hanning", "hamming", "blackman", "chebyshev", "taylor"]


@dataclass
class WaveformParams:
    """
    Central parameter container for beamforming configuration.
    7 mandatory + 2 optional parameters per spec.
    """
    # --- 7 core parameters ---
    num_elements: int = 8               # param 1: number of array elements
    element_spacing: float = 0.5       # param 2: spacing in wavelengths (λ)
    frequency_hz: float = 2.4e9        # param 3: carrier frequency
    steering_angle_deg: float = 0.0    # param 4: beam steering angle
    amplitude: float = 1.0             # param 5: signal amplitude
    sampling_rate: float = 1e10        # param 6: sampling rate (Hz)
    pulse_width: float = 1e-6          # param 7: pulse width (seconds)

    # --- extra params ---
    apodization_window: str = "rectangular"  # param 8
    snr_db: float = 30.0                     # param 9: 0–1000

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WaveformParams":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def validate(self) -> tuple[bool, str]:
        if self.num_elements < 1:
            return False, "num_elements must be >= 1"
        if not (0.1 <= self.element_spacing <= 2.0):
            return False, "element_spacing must be between 0.1 and 2.0 λ"
        if self.frequency_hz <= 0:
            return False, "frequency_hz must be positive"
        if not (-90 <= self.steering_angle_deg <= 90):
            return False, "steering_angle_deg must be -90 to 90"
        if not (0 < self.amplitude <= 10):
            return False, "amplitude must be 0 < a <= 10"
        if not (0 < self.snr_db <= 1000):
            return False, "snr_db must be 0 < snr <= 1000"
        return True, "ok"

    def get_wavelength(self) -> float:
        """λ = c / f"""
        c = 3e8
        return c / self.frequency_hz

    def get_element_spacing_meters(self) -> float:
        return self.element_spacing * self.get_wavelength()

    def get_array_length_meters(self) -> float:
        return (self.num_elements - 1) * self.get_element_spacing_meters()

    def get_beamwidth_estimate_deg(self) -> float:
        """Approximate 3dB beamwidth in degrees"""
        L = self.get_array_length_meters()
        wl = self.get_wavelength()
        if L == 0:
            return 180.0
        return math.degrees(0.886 * wl / L)


@dataclass
class SNRConfig:
    """Encapsulates all SNR/noise logic. Range: 0–1000."""
    snr_db: float = 30.0

    def set_snr(self, value: float) -> None:
        self.snr_db = max(0.0, min(1000.0, value))

    def get_snr_linear(self) -> float:
        return 10 ** (self.snr_db / 10)

    def get_noise_variance(self, signal_power: float = 1.0) -> float:
        snr_linear = self.get_snr_linear()
        return signal_power / snr_linear if snr_linear > 0 else 1.0

    def apply_to_signal(self, signal: "np.ndarray") -> "np.ndarray":
        import numpy as np
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_var = self.get_noise_variance(signal_power)
        noise = np.sqrt(noise_var / 2) * (
            np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
        )
        return signal + noise.real if not np.iscomplexobj(signal) else signal + noise

    def generate_awgn(self, shape: tuple, signal_power: float = 1.0) -> "np.ndarray":
        import numpy as np
        noise_var = self.get_noise_variance(signal_power)
        return np.sqrt(noise_var) * np.random.randn(*shape)