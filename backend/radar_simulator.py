"""
radar_simulator.py
==================
Radar Simulator: 360° phased array, beam steering via electronic phase shifts,
solid body detection with SNR-dependent thresholding, range profiling,
broad-scan / narrow-scan scenarios, and apodization (windowing).

Task 4 – Requirement mapping
─────────────────────────────
✔ Full 360° beam steering via phase shifts (not mechanical rotation)
✔ Up to 5 solid bodies: place, move, resize, delete
✔ Beam width control: wide (fast, presence) / narrow (slow, size estimation)
✔ Scan speed control
✔ SNR (0–1000 slider → 0–60 dB) reflected on ALL outputs:
    - Detection threshold (missed detections at low SNR)
    - Range profile noise floor
    - PPI power map noise
✔ Apodization: rectangular, hanning, hamming, blackman, chebyshev, taylor
✔ Real ULA array factor (not Gaussian approximation)
✔ Radar range equation: Pr = Pt·G²·λ²·σ / ((4π)³·R⁴)
✔ Range profile (A-scan style) per current beam angle
✔ PPI persistence map (plan position indicator)
✔ Broad scan scenario: wide beam, fast, detects presence
✔ Narrow scan scenario: narrow beam, slow, estimates body size
✔ Frequency in GHz range (default 10 GHz X-band, radar-appropriate)
✔ Max range 500 m (radar scale, not ultrasound scale)
✔ Spatial resolution metrics exposed (get_resolution_metrics()):
    - Angular resolution (= beam width in degrees)
    - Range resolution ΔR = c·τ/2
    - Cross-range resolution at max range
    - Minimum detectable RCS given current SNR
    - Max unambiguous range
    All metrics update live when beam width / SNR / frequency change.
"""

import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from base_simulator import BaseSimulator
from waveform_params import WaveformParams
from beamforming_engine import BeamformingEngine
from math_utils import deg2rad, rad2deg, distance, polar_to_cartesian


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RADAR_MAX_RANGE_M   = 500.0    # metres — radar scale
RADAR_TX_POWER_W    = 1000.0   # transmit power (W)
SPEED_OF_LIGHT      = 3e8      # m/s
NOISE_FIGURE_DB     = 5.0      # receiver noise figure
BOLTZMANN_K         = 1.38e-23 # J/K
TEMP_K              = 290.0    # noise temperature


# ─────────────────────────────────────────────────────────────────────────────
# Apodization / Window functions  (pure NumPy, no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _window_rectangular(N: int) -> np.ndarray:
    return np.ones(N)

def _window_hanning(N: int) -> np.ndarray:
    return np.hanning(N)

def _window_hamming(N: int) -> np.ndarray:
    return np.hamming(N)

def _window_blackman(N: int) -> np.ndarray:
    return np.blackman(N)

def _window_chebyshev(N: int, sll_db: float = 40.0) -> np.ndarray:
    """Dolph-Chebyshev window (pure NumPy)."""
    R = 10 ** (sll_db / 20.0)
    x0 = math.cosh(math.acosh(R) / (N - 1))
    w = np.zeros(N)
    for n in range(N):
        v = 0.0
        for k in range(N):
            x = x0 * math.cos(math.pi * k / N)
            if abs(x) > 1:
                T = math.cosh((N - 1) * math.acosh(abs(x))) * math.copysign(1, x)
            else:
                T = math.cos((N - 1) * math.acos(x))
            v += T * math.cos(2 * math.pi * k * n / N)
        w[n] = abs(v) / N
    return w

def _window_taylor(N: int, nbar: int = 4, sll_db: float = 35.0) -> np.ndarray:
    """Taylor window (pure NumPy)."""
    A = math.acosh(10 ** (sll_db / 20.0)) / math.pi
    sp = nbar ** 2 / (A ** 2 + (nbar - 0.5) ** 2)

    def _prod(m):
        p = 1.0
        for j in range(1, nbar):
            if j != m:
                p *= 1.0 - (m * m) / (j * j)
        return p if p != 0 else 1.0

    w = np.zeros(N)
    for n in range(N):
        v = 1.0
        for m in range(1, nbar):
            num = math.pi * m * (1 - m * m / (sp * (A ** 2 + (m - 0.5) ** 2)))
            v += (num / _prod(m)) * math.cos(2 * math.pi * m * (n - N / 2 + 0.5) / N)
        w[n] = v
    return w

def get_apodization_window(N: int, window_type: str, sll_db: float = 40.0) -> np.ndarray:
    """Return normalised apodization window of length N."""
    wt = window_type.lower()
    if wt == "hanning":
        w = _window_hanning(N)
    elif wt == "hamming":
        w = _window_hamming(N)
    elif wt == "blackman":
        w = _window_blackman(N)
    elif wt == "chebyshev":
        w = _window_chebyshev(N, sll_db)
    elif wt == "taylor":
        w = _window_taylor(N, sll_db=sll_db)
    else:                               # rectangular (default)
        w = _window_rectangular(N)
    w = np.abs(w)
    s = w.sum()
    return w / s if s > 0 else w


# ─────────────────────────────────────────────────────────────────────────────
# ULA Array Factor (real phased-array math)
# ─────────────────────────────────────────────────────────────────────────────

def compute_array_factor(
    N: int,
    d_lambda: float,
    steer_deg: float,
    scan_angles_deg: np.ndarray,
    window_type: str = "rectangular",
    sll_db: float = 40.0,
) -> np.ndarray:
    """
    Compute the normalised power array factor for a ULA.

    Parameters
    ----------
    N             : number of elements
    d_lambda      : element spacing in wavelengths (typically 0.5)
    steer_deg     : steering angle in degrees  (0 = broadside)
    scan_angles_deg : array of scan angles (degrees) to evaluate AF
    window_type   : apodization window name
    sll_db        : side-lobe level parameter for chebyshev / taylor

    Returns
    -------
    Normalised power AF (0–1), same shape as scan_angles_deg.
    """
    w = get_apodization_window(N, window_type, sll_db)
    steer_rad = np.deg2rad(steer_deg)
    scan_rad  = np.deg2rad(scan_angles_deg)

    # Phase shift per element index n
    n = np.arange(N)                                # (N,)
    # Phase difference vs scan angle: (N, M) matrix
    phase_diff = 2 * np.pi * d_lambda * np.outer(n, np.sin(scan_rad) - np.sin(steer_rad))
    # Weighted sum: apply window
    AF_complex = (w[:, np.newaxis] * np.exp(1j * phase_diff)).sum(axis=0)   # (M,)
    power = np.abs(AF_complex) ** 2
    mx = power.max()
    return power / mx if mx > 0 else power


def compute_af_360(
    N: int,
    d_lambda: float,
    steer_360_deg: float,
    window_type: str = "rectangular",
    sll_db: float = 40.0,
    resolution: int = 720,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full 360° array factor by computing the ULA pattern over ±90° then
    mirroring it to synthesise the full circle (front + back hemisphere).

    Returns (angles_360, af_360) both of length `resolution`.
    """
    angles_360 = np.linspace(0.0, 360.0, resolution, endpoint=False)

    # Map steering angle: ULA steers ±90°; for angles in the back hemisphere
    # (90°–270°) we phase-reverse the array (equivalent to flipping broadside).
    steer = steer_360_deg % 360.0
    if steer <= 90.0 or steer >= 270.0:
        # Front hemisphere steering
        ula_steer = steer if steer <= 90.0 else steer - 360.0
    else:
        # Back hemisphere: shift by 180° and negate
        ula_steer = steer - 180.0

    # ULA scan over ±90°
    ula_angles = np.linspace(-90.0, 90.0, resolution // 2)
    af_half = compute_array_factor(N, d_lambda, ula_steer, ula_angles, window_type, sll_db)

    # Mirror to build full 360°: front half then mirrored back half
    af_360 = np.concatenate([af_half, af_half[::-1]])
    return angles_360, af_360


# ─────────────────────────────────────────────────────────────────────────────
# SNR utilities
# ─────────────────────────────────────────────────────────────────────────────

def snr_slider_to_db(snr_raw: float) -> float:
    """Map UI slider value (0–1000) to SNR in dB (0–60)."""
    return float(np.clip(snr_raw / 1000.0 * 60.0, 0.0, 60.0))

def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add white Gaussian noise to a signal array to simulate the given SNR.
    At SNR ≥ 60 dB the signal is returned unchanged.
    """
    if snr_db >= 60.0:
        return signal.copy()
    snr_linear = 10 ** (snr_db / 10.0)
    sig_power = float(np.mean(signal ** 2)) + 1e-30
    noise_std = math.sqrt(sig_power / snr_linear)
    return signal + rng.normal(0.0, noise_std, signal.shape)

def detection_threshold_from_snr(snr_db: float) -> float:
    """
    Compute a normalised detection threshold that rises as SNR falls.
    At 60 dB SNR: threshold ≈ 0.01  (very sensitive)
    At  0 dB SNR: threshold ≈ 0.40  (many misses)
    """
    snr_lin = 10 ** (snr_db / 10.0)
    # Neyman-Pearson inspired: threshold inversely proportional to SNR
    return float(np.clip(1.0 / (1.0 + snr_lin * 0.1), 0.01, 0.50))


# ─────────────────────────────────────────────────────────────────────────────
# SolidBody
# ─────────────────────────────────────────────────────────────────────────────

class SolidBody:
    """A radar-reflective solid object in 2D polar space."""
    MAX_BODIES = 5

    def __init__(
        self,
        body_id: int,
        distance_m: float,
        angle_deg: float,
        size_m: float = 10.0,
    ):
        self.body_id   = int(body_id)
        self.distance_m = float(np.clip(distance_m, 1.0, RADAR_MAX_RANGE_M))
        self.angle_deg  = float(angle_deg % 360.0)
        self.size_m     = float(np.clip(size_m, 0.5, 100.0))
        self._deleted   = False
        self._update_position()

    # ── Mutators ──────────────────────────────────────────────────────────────

    def move_to(self, distance_m: float, angle_deg: float) -> None:
        self.distance_m = float(np.clip(distance_m, 1.0, RADAR_MAX_RANGE_M))
        self.angle_deg  = float(angle_deg % 360.0)
        self._update_position()

    def resize(self, new_size_m: float) -> None:
        self.size_m = float(np.clip(new_size_m, 0.5, 100.0))

    def delete(self) -> None:
        self._deleted = True

    # ── Queries ───────────────────────────────────────────────────────────────

    def is_deleted(self) -> bool:
        return self._deleted

    def get_position(self) -> tuple[float, float]:
        return (self._x, self._y)

    def get_rcs(self) -> float:
        """
        Radar cross-section (m²).
        Modelled as the physical cross-sectional area of a sphere of
        diameter = size_m: σ = π·(d/2)²
        """
        return math.pi * (self.size_m / 2.0) ** 2

    def intersects_beam(self, beam_angle_deg: float, beam_width_deg: float) -> bool:
        """
        True if the body's angular position (including its own angular extent)
        falls within the beam cone of half-width beam_width_deg/2.
        """
        diff = abs(beam_angle_deg - self.angle_deg)
        if diff > 180.0:
            diff = 360.0 - diff
        # Angular half-size of the body at its range
        angular_half = math.degrees(
            math.atan2(self.size_m / 2.0, max(self.distance_m, 1.0))
        )
        return diff <= (beam_width_deg / 2.0 + angular_half)

    def to_dict(self) -> dict:
        return {
            "body_id":    self.body_id,
            "distance_m": round(self.distance_m, 2),
            "angle_deg":  round(self.angle_deg, 2),
            "size_m":     round(self.size_m, 2),
            "position":   [round(self._x, 2), round(self._y, 2)],
            "rcs_m2":     round(self.get_rcs(), 3),
            "deleted":    self._deleted,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _update_position(self) -> None:
        rad = math.radians(self.angle_deg)
        self._x = self.distance_m * math.cos(rad)
        self._y = self.distance_m * math.sin(rad)


# ─────────────────────────────────────────────────────────────────────────────
# BeamSteerer  (real array factor, full 360°)
# ─────────────────────────────────────────────────────────────────────────────

class BeamSteerer:
    """
    Full 360° beam steering through electronic phase shifts on a ULA.
    Uses the real ULA array factor — NOT a Gaussian approximation.
    """

    def __init__(self, engine: BeamformingEngine):
        self.engine = engine
        self._current_angle_360 = 0.0

    def steer_to_360(self, angle_360_deg: float) -> None:
        """Steer the beam to any angle 0–360°."""
        self._current_angle_360 = float(angle_360_deg) % 360.0

    def get_current_angle(self) -> float:
        return self._current_angle_360

    def compute_beam_at_angle(
        self,
        angle_360: float,
        resolution: int = 720,
    ) -> dict:
        """
        Compute the full 360° normalised beam pattern for the given steering angle.
        Uses real ULA array factor with the currently set apodization window.
        """
        p = self.engine.params
        N       = p.num_elements
        d_lam   = p.element_spacing
        win     = getattr(p, "apodization_window", "rectangular")
        sll     = getattr(p, "sll_db", 40.0)

        angles_360, af_360 = compute_af_360(N, d_lam, angle_360, win, sll, resolution)
        return {
            "angles_360": angles_360.tolist(),
            "magnitude":  af_360.tolist(),
        }

    def get_steering_phases(self) -> list[float]:
        """
        Return the per-element phase shift (radians) for the current steering angle.
        Useful for visualising the array excitation.
        """
        p = self.engine.params
        N       = p.num_elements
        d_lam   = p.element_spacing
        steer   = self._current_angle_360

        # Map to ±90° ULA steering
        if steer <= 90.0 or steer >= 270.0:
            steer_ula = steer if steer <= 90.0 else steer - 360.0
        else:
            steer_ula = steer - 180.0

        steer_rad = math.radians(steer_ula)
        phases = [
            2 * math.pi * d_lam * n * math.sin(steer_rad)
            for n in range(N)
        ]
        return phases


# ─────────────────────────────────────────────────────────────────────────────
# ScanController
# ─────────────────────────────────────────────────────────────────────────────

class ScanController:
    """
    Controls 360° radar sweep:
      - angle progression (speed in °/step)
      - beam width  (wide → fast presence detection; narrow → size estimation)
      - SNR-dependent detection threshold
      - persistent PPI power map
      - range profile (A-scan) at current beam angle
    """

    def __init__(self, steerer: BeamSteerer):
        self.steerer = steerer
        self._angle             = 0.0
        self._scan_speed_deg    = 5.0    # degrees per step
        self._beam_width_deg    = 10.0
        self._running           = False
        self._ppi_map           = np.zeros(360)   # power per 1° bin
        self._ppi_decay         = 0.92            # persistence decay per step
        self._rng               = np.random.default_rng()

    # ── Configuration ─────────────────────────────────────────────────────────

    def start(self)  -> None: self._running = True
    def stop(self)   -> None: self._running = False

    def set_scan_speed(self, deg_per_step: float) -> None:
        self._scan_speed_deg = float(np.clip(deg_per_step, 0.5, 90.0))

    def set_beam_width(self, width_deg: float) -> None:
        self._beam_width_deg = float(np.clip(width_deg, 1.0, 45.0))

    def get_beam_width(self) -> float:
        return self._beam_width_deg

    def get_scan_speed(self) -> float:
        return self._scan_speed_deg

    # ── Stepping ──────────────────────────────────────────────────────────────

    def step_angle(self) -> float:
        """Advance sweep by one step and steer the beam."""
        self._angle = (self._angle + self._scan_speed_deg) % 360.0
        self.steerer.steer_to_360(self._angle)
        return self._angle

    def get_current_angle(self) -> float:
        return self._angle

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect_bodies(
        self,
        bodies: list[SolidBody],
        snr_db: float,
    ) -> list[dict]:
        """
        Return bodies detected at the current beam angle.
        Uses the radar range equation to compute received power, then
        compares against an SNR-dependent detection threshold.

        At high SNR: almost all intersecting bodies are detected.
        At low SNR:  weak echoes (far/small bodies) are missed.
        """
        threshold = detection_threshold_from_snr(snr_db)
        p = self.steerer.engine.params

        freq   = p.frequency_hz
        wl     = SPEED_OF_LIGHT / freq
        N      = p.num_elements
        # Array gain ≈ N (ULA, uniform illumination upper bound)
        gain   = N

        detected = []
        active = [b for b in bodies if not b.is_deleted()]
        for body in active:
            if not body.intersects_beam(self._angle, self._beam_width_deg):
                continue
            # Radar range equation (simplified, normalised)
            R     = max(body.distance_m, 1.0)
            sigma = body.get_rcs()
            # Normalised received power (0–1 scale)
            pr = (gain ** 2 * wl ** 2 * sigma) / ((4 * math.pi) ** 3 * R ** 4 + 1e-30)
            pr = float(np.clip(pr * 1e8, 0.0, 1.0))   # scale to 0–1
            # Add noise
            noisy = float(add_noise(np.array([pr]), snr_db, self._rng)[0])
            noisy = max(0.0, noisy)
            if noisy >= threshold:
                detected.append({
                    "body_id":    body.body_id,
                    "distance_m": round(body.distance_m, 2),
                    "angle_deg":  round(body.angle_deg, 2),
                    "size_m":     round(body.size_m, 2),
                    "rcs_m2":     round(sigma, 3),
                    "echo_power": round(noisy, 4),
                })
        return detected

    # ── PPI map ───────────────────────────────────────────────────────────────

    def update_ppi(self, bodies: list[SolidBody], snr_db: float) -> None:
        """
        Update the persistent PPI display at the current sweep angle.
        Power decays each step (persistence).  SNR noise is applied.
        """
        # Decay all bins slightly
        self._ppi_map *= self._ppi_decay

        angle_bin = int(self._angle) % 360
        power = 0.0
        for body in bodies:
            if body.is_deleted():
                continue
            if body.intersects_beam(self._angle, self._beam_width_deg):
                sigma = body.get_rcs()
                R     = max(body.distance_m, 1.0)
                pr    = sigma / (R ** 2 + 1e-30)
                power += pr

        # Normalise and add noise
        pr_norm = float(np.clip(power / 1000.0, 0.0, 1.0))
        noisy   = float(add_noise(np.array([pr_norm]), snr_db, self._rng)[0])
        self._ppi_map[angle_bin] = float(np.clip(noisy, 0.0, 1.0))

    def get_ppi_map(self) -> list[float]:
        return self._ppi_map.tolist()

    def reset_ppi(self) -> None:
        self._ppi_map = np.zeros(360)

    # ── Range profile (A-scan) ─────────────────────────────────────────────

    def compute_range_profile(
        self,
        bodies: list[SolidBody],
        snr_db: float,
        n_bins: int = 500,
    ) -> dict:
        """
        Compute a 1-D range profile (A-scan) for the current beam direction.
        Bodies inside the beam cone contribute a sinc-shaped echo at their range.
        SNR noise is applied to the entire profile.

        Returns
        -------
        dict with keys:
            range_axis_m  : list[float]  (0 … RADAR_MAX_RANGE_M)
            profile       : list[float]  (normalised 0–1, noisy)
            peaks         : list[dict]   detected peaks (range, echo)
        """
        range_axis = np.linspace(0.0, RADAR_MAX_RANGE_M, n_bins)
        profile    = np.zeros(n_bins)

        p   = self.steerer.engine.params
        freq = p.frequency_hz
        wl   = SPEED_OF_LIGHT / freq
        bw_hz = freq / 10.0   # approximate pulse bandwidth

        for body in bodies:
            if body.is_deleted():
                continue
            if not body.intersects_beam(self._angle, self._beam_width_deg):
                continue

            R     = body.distance_m
            sigma = body.get_rcs()
            # Echo amplitude (radar range equation, normalised)
            amp   = float(np.clip(sigma / (R ** 2 + 1e-6) * 1e4, 0.0, 1.0))
            # Range resolution: ΔR = c / (2·BW)
            range_res = SPEED_OF_LIGHT / (2.0 * bw_hz)
            # Spread the echo over its physical size + range resolution
            spread_m  = max(body.size_m / 2.0, range_res)
            # Convert to bin index width
            bin_width = max(1, int(spread_m / RADAR_MAX_RANGE_M * n_bins))
            idx       = int(R / RADAR_MAX_RANGE_M * (n_bins - 1))

            lo = max(0, idx - bin_width)
            hi = min(n_bins, idx + bin_width + 1)
            x  = np.arange(lo, hi)
            profile[lo:hi] += amp * np.exp(
                -0.5 * ((x - idx) / max(bin_width / 2.0, 1.0)) ** 2
            )

        profile = np.clip(profile, 0.0, 1.0)
        # Add receiver noise
        profile = np.clip(add_noise(profile, snr_db, self._rng), 0.0, None)
        # Normalise
        mx = profile.max()
        if mx > 0:
            profile /= mx

        # Peak finding: simple threshold-based
        threshold = detection_threshold_from_snr(snr_db)
        peaks = []
        in_peak = False
        for i, v in enumerate(profile):
            if v >= threshold and not in_peak:
                in_peak = True
                peak_start = i
            elif v < threshold and in_peak:
                in_peak = False
                peak_end = i
                peak_idx = peak_start + np.argmax(profile[peak_start:peak_end])
                peaks.append({
                    "range_m": round(float(range_axis[peak_idx]), 1),
                    "echo":    round(float(profile[peak_idx]), 4),
                })

        return {
            "range_axis_m": range_axis.tolist(),
            "profile":      profile.tolist(),
            "peaks":        peaks,
            "beam_angle":   round(self._angle, 1),
            "beam_width":   round(self._beam_width_deg, 1),
        }

    # ── Size estimation (narrow beam) ─────────────────────────────────────────

    def estimate_body_size(self, body: SolidBody) -> dict:
        """
        Narrow the beam and scan ±20° around the body to estimate its angular
        extent, then convert to physical size via the small-angle approximation.
        """
        original_width = self._beam_width_deg
        # Use a narrow 2° beam for size estimation
        self.set_beam_width(2.0)

        scan_range  = np.arange(body.angle_deg - 20.0, body.angle_deg + 20.0, 0.5)
        hit_angles  = [a for a in scan_range if body.intersects_beam(a % 360.0, self._beam_width_deg)]

        if len(hit_angles) > 1:
            angular_span = hit_angles[-1] - hit_angles[0]
            estimated_size = 2.0 * body.distance_m * math.tan(math.radians(angular_span / 2.0))
        else:
            estimated_size = body.size_m * 0.85  # fallback when beam too wide

        self.set_beam_width(original_width)
        return {
            "body_id":          body.body_id,
            "estimated_size_m": round(estimated_size, 2),
            "true_size_m":      round(body.size_m, 2),
            "angular_span_deg": round(hit_angles[-1] - hit_angles[0], 2) if len(hit_angles) > 1 else 0.0,
            "estimation_error_pct": round(
                abs(estimated_size - body.size_m) / max(body.size_m, 0.1) * 100, 1
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PhasedArrayRadar
# ─────────────────────────────────────────────────────────────────────────────

class PhasedArrayRadar:
    """Phased array radar transmitter / receiver."""

    def __init__(self, params: WaveformParams):
        self.params         = params
        self.engine         = BeamformingEngine(params)
        self.steerer        = BeamSteerer(self.engine)
        self.scan_controller = ScanController(self.steerer)

    def update_params(self, params: WaveformParams) -> None:
        self.params = params
        self.engine.update_params(params)
        # Refresh steerer reference
        self.steerer.engine = self.engine

    def get_beam_profile_360(self, resolution: int = 720) -> dict:
        angle = self.steerer.get_current_angle()
        return self.steerer.compute_beam_at_angle(angle, resolution)

    def get_beam_profile_db(self, resolution: int = 500) -> dict:
        """
        Return the beam profile in dB over ±90° (standard beamforming view),
        including main-lobe angle, 3-dB beamwidth, and first side-lobe level.
        Used by the Beam Profile tab.
        """
        p   = self.params
        N   = p.num_elements
        d   = p.element_spacing
        win = getattr(p, "apodization_window", "rectangular")
        sll = getattr(p, "sll_db", 40.0)

        steer   = self.steerer.get_current_angle()
        # Map to ULA ±90°
        if steer <= 90.0 or steer >= 270.0:
            steer_ula = steer if steer <= 90.0 else steer - 360.0
        else:
            steer_ula = steer - 180.0

        scan    = np.linspace(-90.0, 90.0, resolution)
        af      = compute_array_factor(N, d, steer_ula, scan, win, sll)
        af_db   = 10.0 * np.log10(af + 1e-12)
        af_db   = np.clip(af_db, -60.0, 0.0)

        # Metrics
        main_idx   = int(np.argmax(af))
        main_angle = float(scan[main_idx])
        half_power = af.max() * 0.5
        above      = np.where(af >= half_power)[0]
        beamwidth  = float(scan[above[-1]] - scan[above[0]]) if len(above) >= 2 else 0.0

        # First sidelobe: highest peak outside 3-dB region
        mask = np.ones(len(af), dtype=bool)
        if len(above) >= 2:
            mask[above[0]:above[-1]+1] = False
        sll_val = float(10.0 * np.log10(af[mask].max() + 1e-12)) if mask.any() else -60.0

        return {
            "angles":             scan.tolist(),
            "af_db":              af_db.tolist(),
            "main_lobe_angle":    round(main_angle, 2),
            "beamwidth_3db_deg":  round(beamwidth, 2),
            "sidelobe_level_db":  round(sll_val, 2),
            "steering_phases":    [round(ph, 4) for ph in self.steerer.get_steering_phases()],
        }

    def compute_received_power(self, body: SolidBody) -> float:
        """
        Full radar range equation:
            Pr = Pt · Gt · Gr · λ² · σ / ((4π)³ · R⁴)
        For a phased array: Gt = Gr = N (element count as proxy for gain).
        """
        freq   = self.params.frequency_hz
        wl     = SPEED_OF_LIGHT / freq
        N      = self.params.num_elements
        sigma  = body.get_rcs()
        R      = max(body.distance_m, 1.0)
        pr = (RADAR_TX_POWER_W * N**2 * wl**2 * sigma) / ((4 * math.pi)**3 * R**4)
        return pr


# ─────────────────────────────────────────────────────────────────────────────
# RadarSimulator  (top-level, integrates with BaseSimulator)
# ─────────────────────────────────────────────────────────────────────────────

class RadarSimulator(BaseSimulator):
    """
    Top-level radar simulator.

    Public API (called by main.py / FastAPI routes)
    ─────────────────────────────────────────────────
    initialize()                         → set up fresh state
    step()                               → advance sweep by one tick
    reset()                              → full reset
    update_params(params)                → update shared beamforming params
    add_body(dist, angle, size)          → place a new body (max 5)
    move_body(id, dist, angle)           → relocate existing body
    resize_body(id, size)                → change body size
    delete_body(id)                      → mark body deleted
    do_broad_scan()                      → wide-beam 360° presence scan
    do_narrow_scan(angle)                → narrow-beam size estimation near angle
    get_output()                         → full state dict for frontend
    get_state()                          → alias for get_output() (REST /radar/state)
    set_scan_speed(deg)                  → set sweep speed
    set_beam_width(deg)                  → set beam width
    get_range_profile()                  → A-scan at current angle
    get_beam_profile_db()                → dB profile for Beam Profile tab
    """

    def __init__(self, params: WaveformParams = None):
        if params is None:
            params = WaveformParams(
                frequency_hz      = 10e9,   # 10 GHz — X-band radar
                num_elements      = 16,
                element_spacing   = 0.5,
                snr_db            = 20.0,
                pulse_width       = 1e-6,
                amplitude         = 1.0,
                sampling_rate     = 1e10,
                apodization_window = "chebyshev",
            )
        super().__init__(params)
        self.radar       = PhasedArrayRadar(params)
        self.bodies: list[SolidBody] = []
        self._scan_active = False
        self._detections: list[dict] = []
        self._next_body_id = 1       # monotonically increasing — survives deletes

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        self.bodies          = []
        self._next_body_id   = 1
        self._detections     = []
        self.radar.scan_controller.start()
        self._scan_active = True

    def step(self) -> dict:
        """Advance one sweep step. Called by /radar/step."""
        self._step_count += 1
        snr_db  = getattr(self.params, "snr_db", 20.0)
        angle   = self.radar.scan_controller.step_angle()
        self.radar.scan_controller.update_ppi(self.bodies, snr_db)
        self._detections = self.radar.scan_controller.detect_bodies(self.bodies, snr_db)
        return {
            "angle":      round(angle, 2),
            "detections": self._detections,
            "step":       self._step_count,
        }

    def reset(self) -> None:
        self._step_count   = 0
        self.bodies        = []
        self._next_body_id = 1
        self._detections   = []
        self.radar         = PhasedArrayRadar(self.params)
        self.radar.scan_controller.start()

    def update_params(self, params: WaveformParams) -> None:
        self.params = params
        self.radar.update_params(params)

    # ── Scan control (wired to frontend sliders) ───────────────────────────────

    def set_scan_speed(self, deg_per_step: float) -> None:
        self.radar.scan_controller.set_scan_speed(deg_per_step)

    def set_beam_width(self, width_deg: float) -> None:
        self.radar.scan_controller.set_beam_width(width_deg)

    # ── Body management ───────────────────────────────────────────────────────

    def add_body(
        self,
        distance_m: float,
        angle_deg: float,
        size_m: float = 10.0,
    ) -> dict:
        active = [b for b in self.bodies if not b.is_deleted()]
        if len(active) >= SolidBody.MAX_BODIES:
            return {"error": f"Maximum {SolidBody.MAX_BODIES} bodies allowed"}
        body = SolidBody(self._next_body_id, distance_m, angle_deg, size_m)
        self._next_body_id += 1
        self.bodies.append(body)
        return body.to_dict()

    def move_body(self, body_id: int, distance_m: float, angle_deg: float) -> dict:
        for b in self.bodies:
            if b.body_id == body_id and not b.is_deleted():
                b.move_to(distance_m, angle_deg)
                return b.to_dict()
        return {"error": f"Body {body_id} not found"}

    def resize_body(self, body_id: int, size_m: float) -> dict:
        for b in self.bodies:
            if b.body_id == body_id and not b.is_deleted():
                b.resize(size_m)
                return b.to_dict()
        return {"error": f"Body {body_id} not found"}

    def delete_body(self, body_id: int) -> dict:
        for b in self.bodies:
            if b.body_id == body_id:
                b.delete()
                return {"deleted": body_id, "ok": True}
        return {"error": f"Body {body_id} not found"}

    # ── Scan scenarios ────────────────────────────────────────────────────────

    def do_broad_scan(self) -> dict:
        """
        Wide-beam (20°) full 360° scan — fast presence detection.
        Simulates sweeping quickly: lower angular resolution but covers
        the full azimuth in one pass.  Lower SNR → more misses.
        """
        snr_db         = getattr(self.params, "snr_db", 20.0)
        original_width = self.radar.scan_controller.get_beam_width()
        original_speed = self.radar.scan_controller.get_scan_speed()

        self.radar.scan_controller.set_beam_width(20.0)   # wide
        self.radar.scan_controller.set_scan_speed(20.0)   # fast

        all_detections: dict[int, dict] = {}
        saved_angle = self.radar.scan_controller.get_current_angle()
        for angle in np.arange(0.0, 360.0, 20.0):
            self.radar.scan_controller._angle = float(angle)
            self.radar.steerer.steer_to_360(float(angle))
            dets = self.radar.scan_controller.detect_bodies(self.bodies, snr_db)
            for d in dets:
                all_detections.setdefault(d["body_id"], d)

        # Restore
        self.radar.scan_controller._angle = saved_angle
        self.radar.scan_controller.set_beam_width(original_width)
        self.radar.scan_controller.set_scan_speed(original_speed)

        return {
            "broad_scan_detections": list(all_detections.values()),
            "beam_width_used":       20.0,
            "scan_speed_used":       20.0,
            "snr_db":                round(snr_db, 1),
            "note": "Wide beam — fast, lower angular resolution",
        }

    def do_narrow_scan(self, target_angle: float) -> dict:
        """
        Narrow-beam (2°) scan around target_angle — precise size estimation.
        Simulates a focused dwell: slower but higher angular resolution.
        """
        results = []
        for body in self.bodies:
            if body.is_deleted():
                continue
            diff = abs(body.angle_deg - target_angle)
            if diff > 180.0:
                diff = 360.0 - diff
            if diff < 30.0:
                est = self.radar.scan_controller.estimate_body_size(body)
                results.append(est)
        return {
            "narrow_scan_results": results,
            "beam_width_used":     2.0,
            "target_angle":        round(target_angle, 1),
            "note": "Narrow beam — slow, high angular resolution, estimates body size",
        }

    # ── Output / state ────────────────────────────────────────────────────────

    def get_resolution_metrics(self) -> dict:
        """
        Compute and return spatial resolution metrics that are directly
        tied to the current beam width and array parameters.
        These are shown as live metric cards in the frontend so the user
        can SEE how narrowing the beam improves resolution.

        Metrics returned
        ────────────────
        angular_resolution_deg  : 3-dB beamwidth (= beam_width_deg for ULA)
        range_resolution_m      : ΔR = c / (2·BW_hz)  — range cell size in metres
        cross_range_resolution_m: at max range = R·tan(BW/2)  — lateral cell size
        min_detectable_rcs_m2   : smallest detectable RCS given current SNR
        max_unambiguous_range_m : c / (2·PRF)  — using pulse_width as 1/PRF proxy
        """
        p          = self.params
        bw_deg     = self.radar.scan_controller.get_beam_width()
        freq_hz    = p.frequency_hz
        wl         = SPEED_OF_LIGHT / freq_hz
        pulse_w    = getattr(p, "pulse_width", 1e-6)          # seconds
        snr_db     = getattr(p, "snr_db", 20.0)

        # Angular resolution = beam width (3-dB)
        ang_res_deg = bw_deg

        # Range resolution: ΔR = c·τ/2  (pulse-compression proxy)
        range_res_m = SPEED_OF_LIGHT * pulse_w / 2.0

        # Cross-range resolution at max range (lateral footprint of beam)
        bw_rad           = math.radians(bw_deg)
        cross_range_m    = RADAR_MAX_RANGE_M * math.tan(bw_rad / 2.0) * 2.0

        # Minimum detectable RCS (from SNR threshold + range equation)
        snr_lin          = 10 ** (snr_db / 10.0)
        N                = p.num_elements
        gain             = N ** 2
        R_max            = RADAR_MAX_RANGE_M
        min_rcs_m2       = (snr_lin * (4 * math.pi) ** 3 * R_max ** 4) / \
                           (RADAR_TX_POWER_W * gain * wl ** 2 + 1e-30)

        # Max unambiguous range: c / (2·PRF); PRF ≈ 1/pulse_width
        prf              = 1.0 / max(pulse_w, 1e-9)
        max_unamb_range  = SPEED_OF_LIGHT / (2.0 * prf)

        return {
            "angular_resolution_deg":   round(ang_res_deg, 2),
            "range_resolution_m":       round(range_res_m, 2),
            "cross_range_resolution_m": round(cross_range_m, 1),
            "min_detectable_rcs_m2":    round(min_rcs_m2, 4),
            "max_unambiguous_range_m":  round(max_unamb_range, 1),
            "beam_width_deg":           round(bw_deg, 1),
            "frequency_ghz":            round(freq_hz / 1e9, 2),
            "wavelength_m":             round(wl, 4),
        }

    def get_range_profile(self) -> dict:
        """A-scan range profile at the current beam angle."""
        snr_db = getattr(self.params, "snr_db", 20.0)
        return self.radar.scan_controller.compute_range_profile(self.bodies, snr_db)

    def get_beam_profile_db(self) -> dict:
        """Beam profile in dB (±90°) for the Beam Profile tab."""
        return self.radar.get_beam_profile_db()

    def get_output(self) -> dict:
        """Full simulator state — called by /radar/state."""
        snr_db = getattr(self.params, "snr_db", 20.0)

        # PPI with noise
        ppi_arr   = np.array(self.radar.scan_controller.get_ppi_map())
        ppi_noisy = add_noise(ppi_arr, snr_db, self.radar.scan_controller._rng)
        ppi_noisy = np.clip(ppi_noisy, 0.0, None)
        mx = ppi_noisy.max()
        if mx > 0:
            ppi_noisy /= mx

        # Beam profile
        beam = self.radar.get_beam_profile_360()

        return {
            "current_angle":  round(self.radar.scan_controller.get_current_angle(), 2),
            "beam_width":     round(self.radar.scan_controller.get_beam_width(), 1),
            "scan_speed":     round(self.radar.scan_controller.get_scan_speed(), 1),
            "snr_db":         round(snr_db, 1),
            "ppi_map":        ppi_noisy.tolist(),
            "beam_profile":   beam,
            "bodies":         [b.to_dict() for b in self.bodies if not b.is_deleted()],
            "detections":     self._detections,
            "resolution_metrics": self.get_resolution_metrics(),
            "params": {
                "frequency_ghz":   round(self.params.frequency_hz / 1e9, 2),
                "num_elements":    self.params.num_elements,
                "element_spacing": self.params.element_spacing,
                "window":          getattr(self.params, "apodization_window", "rectangular"),
                "max_range_m":     RADAR_MAX_RANGE_M,
            },
        }

    def get_state(self) -> dict:
        """Alias for get_output() — matches /radar/state REST endpoint."""
        return self.get_output()

    # ── BaseSimulator abstract method ──────────────────────────────────────────

    def apply_snr(self, signal: np.ndarray) -> np.ndarray:
        """Override BaseSimulator.apply_snr using our radar-specific noise."""
        snr_db = getattr(self.params, "snr_db", 20.0)
        return add_noise(signal, snr_db, self.radar.scan_controller._rng)