"""
Math and signal utilities. All shared math lives here - never duplicated in apps.
"""
import numpy as np
import math
from typing import Tuple


# ── Geometry ──────────────────────────────────────────────────────────────────

def deg2rad(deg: float) -> float:
    return math.radians(deg)

def rad2deg(rad: float) -> float:
    return math.degrees(rad)

def polar_to_cartesian(r: float, theta_deg: float) -> Tuple[float, float]:
    """Convert polar (r, θ°) → (x, y)."""
    rad = deg2rad(theta_deg)
    return r * math.cos(rad), r * math.sin(rad)

def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
    """Convert (x, y) → (r, θ°)."""
    r = math.sqrt(x**2 + y**2)
    theta = rad2deg(math.atan2(y, x))
    return r, theta

def distance(p1: Tuple, p2: Tuple) -> float:
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def angle_between(p1: Tuple, p2: Tuple) -> float:
    """Angle in degrees from p1 to p2."""
    return rad2deg(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def normalize_angle(angle_deg: float) -> float:
    """Bring angle into [-180, 180]."""
    while angle_deg > 180: angle_deg -= 360
    while angle_deg < -180: angle_deg += 360
    return angle_deg


# ── Beamforming math ──────────────────────────────────────────────────────────

def array_factor(
    num_elements: int,
    spacing_wl: float,
    angles_deg: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute normalized array factor over full 360° angle sweep.
    This function is intentionally steering-agnostic: it evaluates whatever
    complex weights are supplied. Any steering phase should be encoded in
    `weights` by the caller during weight generation.

    AF(θ) = |Σ w_n * exp(j * 2π * d * n * cos(θ))|
    """
    if weights is None:
        weights = np.ones(num_elements, dtype=complex)

    d = spacing_wl  # spacing in wavelengths
    n = np.arange(num_elements)

    angles_rad = np.array([deg2rad(a) for a in angles_deg])

    # Phase progression for each observation angle (vectorised over angles)
    # shape: (len(angles), num_elements)
    element_phases = 2 * np.pi * d * n[None, :] * np.cos(angles_rad)[:, None]
    af = np.sum(weights[None, :] * np.exp(1j * element_phases), axis=1)  # (angles,)

    af_mag = np.abs(af)
    max_val = np.max(af_mag)
    return af_mag / max_val if max_val > 0 else af_mag


def compute_steering_delays(
    num_elements: int,
    spacing_m: float,
    steering_deg: float,
    sound_speed: float = 3e8,
) -> np.ndarray:
    """Compute time delays for steering angle (seconds) — full 360° via cosine projection."""
    steering_rad = deg2rad(steering_deg)
    n = np.arange(num_elements)
    return (n * spacing_m * np.cos(steering_rad)) / sound_speed


def compute_steering_phases(
    num_elements: int,
    spacing_wl: float,
    steering_deg: float,
) -> np.ndarray:
    """Compute phase shifts (radians) for steering — full 360° via cosine projection."""
    steering_rad = deg2rad(steering_deg)
    n = np.arange(num_elements)
    return -2 * np.pi * spacing_wl * n * np.cos(steering_rad)


def compute_interference_map(
    num_elements: int,
    spacing_m: float,
    steering_deg: float,
    wavelength: float,
    grid_size: int = 100,
    extent_m: float = 10.0,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute 2D interference (pressure) map on a full 360° grid.
    Returns normalized intensity map of shape (grid_size, grid_size).
    """
    if weights is None:
        weights = np.ones(num_elements, dtype=complex)

    x = np.linspace(-extent_m, extent_m, grid_size)
    y = np.linspace(-extent_m, extent_m, grid_size)  # full space, not just y>=0
    X, Y = np.meshgrid(x, y)

    element_positions = (np.arange(num_elements) - (num_elements - 1) / 2) * spacing_m

    # Use cosine projection for steering delays (consistent with array_factor)
    steering_rad = deg2rad(steering_deg)
    steering_delays = element_positions * np.cos(steering_rad) / wavelength

    field = np.zeros_like(X, dtype=complex)
    k = 2 * np.pi / wavelength

    for i, x_elem in enumerate(element_positions):
        r = np.sqrt((X - x_elem) ** 2 + Y ** 2) + 1e-9
        phase_steering = 2 * np.pi * steering_delays[i]
        field += weights[i] * np.exp(1j * (k * r - phase_steering)) / r

    intensity = np.abs(field) ** 2
    max_val = intensity.max()
    return intensity / max_val if max_val > 0 else intensity


def db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert amplitude array to dB."""
    return 20 * np.log10(np.abs(x) + eps)


def path_loss_db(distance_m: float, freq_hz: float) -> float:
    """Free-space path loss in dB."""
    if distance_m <= 0:
        return 0.0
    c = 3e8
    wl = c / freq_hz
    return 20 * np.log10(4 * np.pi * distance_m / wl)


def doppler_shift(
    frequency_hz: float,
    velocity_m_s: float,
    sound_speed: float,
    angle_deg: float = 0.0,
) -> float:
    """
    Doppler shift: Δf = 2 * f₀ * v * cos(θ) / c
    """
    return 2 * frequency_hz * velocity_m_s * math.cos(deg2rad(angle_deg)) / sound_speed


def reflection_coefficient(z1: float, z2: float) -> float:
    """Acoustic reflection coefficient at impedance boundary."""
    denom = z1 + z2
    return (z2 - z1) / denom if denom != 0 else 0.0


def transmission_coefficient(z1: float, z2: float) -> float:
    """Acoustic transmission coefficient."""
    denom = z1 + z2
    return 2 * z2 / denom if denom != 0 else 1.0