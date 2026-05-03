"""
Single antenna element oscillator.
Generates time-domain signals with configurable frequency, amplitude, phase, and delay.
"""
import numpy as np
import math


class Oscillator:
    """
    Models one element of a phased-array antenna.
    Phase shifts and time delays are applied independently.
    """

    def __init__(
        self,
        frequency: float,
        amplitude: float = 1.0,
        phase: float = 0.0,
        position: tuple = (0.0, 0.0),
    ):
        self.frequency = frequency          # Hz
        self.amplitude = amplitude
        self.phase = phase                  # radians (includes steering phase shift)
        self.base_phase = phase             # original phase (for reset)
        self.position = np.array(position, dtype=float)
        self._delay_samples = 0.0          # fractional delay in samples
        self._weight = complex(amplitude * math.cos(phase), amplitude * math.sin(phase))

    # ── Signal generation ─────────────────────────────────────────────────

    def generate_signal(self, t: np.ndarray) -> np.ndarray:
        """Generate real-valued sinusoidal signal at array of time points."""
        return self.amplitude * np.cos(2 * np.pi * self.frequency * t + self.phase)

    def generate_complex_signal(self, t: np.ndarray) -> np.ndarray:
        """Generate complex (IQ) signal."""
        return self.amplitude * np.exp(1j * (2 * np.pi * self.frequency * t + self.phase))

    def generate_pulse(self, t: np.ndarray, pulse_width: float) -> np.ndarray:
        """Generate a windowed pulse signal."""
        envelope = np.zeros_like(t)
        mask = (t >= 0) & (t <= pulse_width)
        envelope[mask] = 1.0
        return envelope * self.generate_signal(t)

    # ── Phase / delay manipulation ────────────────────────────────────────

    def apply_delay(self, delay_seconds: float, sampling_rate: float) -> None:
        """Apply time delay (converted to phase shift at carrier frequency)."""
        self._delay_samples = delay_seconds * sampling_rate
        phase_from_delay = -2 * np.pi * self.frequency * delay_seconds
        self.phase = self.base_phase + phase_from_delay
        self._update_weight()

    def apply_phase_shift(self, phase_rad: float) -> None:
        """Directly add a phase shift (steering weight)."""
        self.phase = self.base_phase + phase_rad
        self._update_weight()

    def set_frequency(self, freq_hz: float) -> None:
        self.frequency = freq_hz
        self._update_weight()

    def set_amplitude(self, amplitude: float) -> None:
        self.amplitude = max(0.0, amplitude)
        self._update_weight()

    def set_apodization_weight(self, weight: float) -> None:
        """Apply window function weight to amplitude."""
        self.amplitude = self.amplitude * weight
        self._update_weight()

    def get_complex_weight(self) -> complex:
        """Return complex steering weight w = a * e^(jφ)."""
        return self._weight

    def reset_phase(self) -> None:
        self.phase = self.base_phase
        self._update_weight()

    def _update_weight(self) -> None:
        self._weight = self.amplitude * np.exp(1j * self.phase)

    def to_dict(self) -> dict:
        return {
            "frequency": self.frequency,
            "amplitude": self.amplitude,
            "phase": self.phase,
            "position": self.position.tolist(),
        }