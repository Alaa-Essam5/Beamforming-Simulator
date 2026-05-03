"""
Core Beamforming Engine - used by ALL three applications.
Never duplicated; always imported via this module.
"""
import numpy as np
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from waveform_params import WaveformParams
from oscillator import Oscillator
from math_utils import (
    array_factor, compute_steering_delays, compute_steering_phases,
    compute_interference_map, db, deg2rad, rad2deg
)


# ── Apodizer ──────────────────────────────────────────────────────────────────

class Apodizer:
    """
    Window/apodization functions to reduce side lobes.
    ALL windowing logic lives here.
    """
    AVAILABLE = ["rectangular", "hanning", "hamming", "blackman", "chebyshev", "taylor"]

    def __init__(self, window_type: str = "rectangular", num_elements: int = 8):
        self.num_elements = num_elements
        self.window_type = window_type
        self._weights = self._compute(window_type)

    def set_window(self, window_type: str) -> None:
        if window_type not in self.AVAILABLE:
            raise ValueError(f"Unknown window: {window_type}. Choose from {self.AVAILABLE}")
        self.window_type = window_type
        self._weights = self._compute(window_type)

    def get_weights(self) -> np.ndarray:
        return self._weights.copy()

    def apply(self, signal: np.ndarray) -> np.ndarray:
        w = self._weights
        if signal.ndim == 1:
            return signal * w
        return signal * w[:, np.newaxis]

    def get_sidelobe_reduction_db(self) -> float:
        REDUCTIONS = {
            "rectangular": 0,
            "hanning": 31.5,
            "hamming": 42.7,
            "blackman": 58.1,
            "chebyshev": 50.0,
            "taylor": 35.0,
        }
        return REDUCTIONS.get(self.window_type, 0)

    def _compute(self, wt: str) -> np.ndarray:
        n = self.num_elements
        if wt == "rectangular":   return self._rectangular(n)
        if wt == "hanning":       return self._hanning(n)
        if wt == "hamming":       return self._hamming(n)
        if wt == "blackman":      return self._blackman(n)
        if wt == "chebyshev":     return self._chebyshev(n)
        if wt == "taylor":        return self._taylor(n)
        return np.ones(n)

    @staticmethod
    def _rectangular(n): return np.ones(n)

    @staticmethod
    def _hanning(n): return np.hanning(n)

    @staticmethod
    def _hamming(n): return np.hamming(n)

    @staticmethod
    def _blackman(n): return np.blackman(n)

    @staticmethod
    def _chebyshev(n, sidelobe_db=50.0):
        # Use scipy explicitly for reliability over np.chebwin
        from scipy.signal.windows import chebwin
        return np.array(chebwin(n, at=sidelobe_db))

    @staticmethod
    def _taylor(n, n_bar=4, sidelobe_db=35.0):
        """Taylor window approximation."""
        A = np.arccosh(10 ** (sidelobe_db / 20)) / np.pi
        sp2 = n_bar ** 2 / (A**2 + (n_bar - 0.5)**2)
        w = np.ones(n)
        for i in range(1, n_bar):
            num = 1.0
            den = 1.0
            for m in range(1, n_bar):
                num *= (1 - (i**2) / (sp2 * (A**2 + (m - 0.5)**2)))
                if m != i:
                    den *= (1 - (i**2) / (m**2))
            Fm = ((-1)**(i+1) * num) / (2 * den) if den != 0 else 0
            idx = np.arange(n)
            w += 2 * Fm * np.cos(2 * np.pi * i * (idx - (n-1)/2) / n)
        return w / w.max()


# ── NoiseModel ────────────────────────────────────────────────────────────────

class NoiseModel:
    """All noise / path-loss computation. Injected into engine."""

    def __init__(self, snr_db: float = 30.0):
        self.snr_db = snr_db

    def set_snr(self, snr_db: float) -> None:
        self.snr_db = max(0.0, min(1000.0, snr_db))

    def add_awgn(self, signal: np.ndarray) -> np.ndarray:
        power = np.mean(signal ** 2) if signal.size > 0 else 1.0
        snr_lin = 10 ** (self.snr_db / 10)
        noise_var = power / snr_lin if snr_lin > 0 else power
        noise = np.sqrt(noise_var) * np.random.randn(*signal.shape)
        return signal + noise

    def apply_path_loss(self, signal: np.ndarray, distance_m: float, freq_hz: float) -> np.ndarray:
        c = 3e8
        wl = c / freq_hz if freq_hz > 0 else 0.125
        if distance_m <= 0:
            return signal
        pl = (4 * np.pi * distance_m / wl) ** 2
        return signal / np.sqrt(pl)

    def compute_received_snr(self, tx_snr_db: float, distance_m: float, freq_hz: float) -> float:
        """Path-loss adjusted SNR."""
        from math_utils import path_loss_db
        pl = path_loss_db(distance_m, freq_hz)
        return tx_snr_db - pl


# ── BeamformingEngine ─────────────────────────────────────────────────────────

class BeamformingEngine:
    """
    Central computation engine shared by all 3 simulators.
    Owns: oscillator array, apodizer, noise model, all beam math.
    """

    def __init__(self, params: WaveformParams):
        self.params = params
        self.apodizer = Apodizer(params.apodization_window, params.num_elements)
        self.noise_model = NoiseModel(params.snr_db)
        self._oscillators: list[Oscillator] = []
        self._build_array()

    # ── Array construction ────────────────────────────────────────────────

    def _build_array(self) -> None:
        p = self.params
        spacing_m = p.get_element_spacing_meters()
        self._oscillators = []
        for i in range(p.num_elements):
            x = (i - (p.num_elements - 1) / 2) * spacing_m
            osc = Oscillator(
                frequency=p.frequency_hz,
                amplitude=p.amplitude,
                phase=0.0,
                position=(x, 0.0),
            )
            self._oscillators.append(osc)
        self._apply_steering(p.steering_angle_deg)

    def _apply_steering(self, angle_deg: float) -> None:
        p = self.params
        phases = compute_steering_phases(p.num_elements, p.element_spacing, angle_deg)
        apo_weights = self.apodizer.get_weights()
        for i, osc in enumerate(self._oscillators):
            osc.amplitude = p.amplitude * float(apo_weights[i])
            osc.apply_phase_shift(phases[i])

    # ── Beam computation ──────────────────────────────────────────────────

    def compute_beam_profile(self, resolution: int = 361) -> dict:
        """Return angles and normalized AF magnitude (with noise) over full 360°."""
        angles = np.linspace(-180, 180, resolution)
        weights = np.array([o.get_complex_weight() for o in self._oscillators])
        af = array_factor(
            self.params.num_elements,
            self.params.element_spacing,
            angles,
            weights,
        )
        af_noisy = self.noise_model.add_awgn(af)
        af_noisy = np.clip(af_noisy, 0, None)
        af_db = db(af_noisy + 1e-12)
        return {
            "angles": angles.tolist(),
            "magnitude": af_noisy.tolist(),
            "magnitude_db": af_db.tolist(),
            "main_lobe_angle": self.get_main_lobe_angle(),
            "sidelobe_level_db": self.get_side_lobe_level(af_noisy, angles),
            "beamwidth_deg": self.get_beamwidth(af_noisy, angles),
        }

    def compute_interference_map(self, grid_size: int = 80, extent_m: float = None) -> dict:
        p = self.params
        if extent_m is None:
            extent_m = max(10.0, p.num_elements * p.get_element_spacing_meters() * 5)
        weights = np.array([o.get_complex_weight() for o in self._oscillators])
        imap = compute_interference_map(
            p.num_elements, p.get_element_spacing_meters(),
            p.steering_angle_deg, p.get_wavelength(),
            grid_size, extent_m, weights
        )
        noisy_map = self.noise_model.add_awgn(imap)
        noisy_map = np.clip(noisy_map, 0, None)
        max_val = noisy_map.max()
        if max_val > 0:
            noisy_map /= max_val
        return {
            "map": noisy_map.tolist(),
            "extent_m": extent_m,
            "grid_size": grid_size,
        }

    def get_complex_weights(self) -> list:
        return [(o.get_complex_weight().real, o.get_complex_weight().imag)
                for o in self._oscillators]

    def get_main_lobe_angle(self) -> float:
        return self.params.steering_angle_deg

    def get_side_lobe_level(self, af: np.ndarray, angles: np.ndarray) -> float:
        if len(af) == 0:
            return -999.0
        peak_idx = np.argmax(af)
        main_width = max(5, int(len(af) * 0.05))
        mask = np.ones(len(af), dtype=bool)
        mask[max(0, peak_idx - main_width):peak_idx + main_width + 1] = False
        sidelobes = af[mask]
        if len(sidelobes) == 0:
            return -999.0
        sl_db = float(db(np.array([np.max(sidelobes) + 1e-12]))[0]) if np.max(sidelobes) > 0 else -999.0
        return float(sl_db)

    def get_beamwidth(self, af: np.ndarray, angles: np.ndarray, level: float = 0.707) -> float:
        """Return 3dB beamwidth in degrees."""
        if len(af) == 0:
            return 0.0
        peak = np.max(af)
        threshold = peak * level
        above = af >= threshold
        if not np.any(above):
            return 0.0
        idxs = np.where(above)[0]
        return float(angles[idxs[-1]] - angles[idxs[0]])

    # ── Parameter updates ─────────────────────────────────────────────────

    def steer_beam(self, angle_deg: float) -> None:
        # Normalize to [-180, 180] — full 360° steering, no clamping
        angle_deg = float(angle_deg)
        while angle_deg > 180:  angle_deg -= 360
        while angle_deg < -180: angle_deg += 360
        self.params.steering_angle_deg = angle_deg
        self._apply_steering(self.params.steering_angle_deg)

    def update_params(self, params: WaveformParams) -> None:
        rebuild = (
            params.num_elements != self.params.num_elements or
            abs(params.element_spacing - self.params.element_spacing) > 1e-9 or
            abs(params.frequency_hz - self.params.frequency_hz) > 1.0
        )
        self.params = params
        self.noise_model.set_snr(params.snr_db)
        self.apodizer = Apodizer(params.apodization_window, params.num_elements)
        if rebuild:
            self._build_array()
        else:
            self._apply_steering(params.steering_angle_deg)

    def get_oscillator_positions(self) -> list:
        return [o.position.tolist() for o in self._oscillators]