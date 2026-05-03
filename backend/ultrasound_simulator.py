"""
Ultrasound Simulator — complete rewrite with correct physics.

DOPPLER FIX:
  - θ uses SIGNED dot product → fd can be positive (toward probe) or negative (away)
  - Range gate fixed: vessel y-coord uses image coords (+y DOWN from probe)
  - Signed Δf: flow toward probe → positive peak, away → negative peak
  - θ=90° → cos=0 → fd=0 regardless of velocity ✓

A-MODE:
  - Focal zone: beam narrows then diverges, echoes strongest at focal depth
  - Marked focal point returned in output
  - Stronger echo amplitude, clearer interface spikes

B-MODE:
  - Brighter output with better contrast
  - Tumor clearly visible

TUMOR:
  - Added a prominent tumor shape inside the brain
"""
import numpy as np
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_simulator import BaseSimulator
from waveform_params import WaveformParams
from math_utils import deg2rad, rad2deg, reflection_coefficient

SOUND_SPEED_TISSUE = 1540.0
US_FREQ_DEFAULT    = 5e6
MAX_DEPTH_CM       = 20.0

TISSUE_PROPS = {
    "background":    {"z": 0.0,    "att": 0.0,   "c": 1540, "rho": 1000, "echo": 0.00},
    "skull":         {"z": 7.8e6,  "att": 10.0,  "c": 3000, "rho": 1900, "echo": 0.95},
    "brain_white":   {"z": 1.58e6, "att": 0.30,  "c": 1560, "rho": 1040, "echo": 0.20},
    "brain_gray":    {"z": 1.60e6, "att": 0.35,  "c": 1550, "rho": 1045, "echo": 0.28},
    "csf":           {"z": 1.52e6, "att": 0.002, "c": 1515, "rho": 1000, "echo": 0.01},
    "blood":         {"z": 1.61e6, "att": 0.18,  "c": 1570, "rho": 1050, "echo": 0.04},
    "tumor":         {"z": 1.72e6, "att": 0.90,  "c": 1590, "rho": 1080, "echo": 0.75},
    "calcification": {"z": 7.5e6,  "att": 12.0,  "c": 3000, "rho": 2200, "echo": 0.98},
    "cyst":          {"z": 1.50e6, "att": 0.005, "c": 1510, "rho": 990,  "echo": 0.00},
}


def _make_shepp_logan_shapes():
    rows = [
        # id  name                cx     cy      a      b    ang   tissue
        (0,  "Skull",             0.0,   0.0,   9.20,  6.90,   0, "skull"),
        (1,  "Brain",             0.0,  -0.18,  8.74,  6.62,   0, "brain_white"),
        (2,  "CSF Left",          2.2,   0.0,   3.10,  1.10, -18, "csf"),
        (3,  "CSF Right",        -2.2,   0.0,   4.10,  1.60,  18, "csf"),
        (4,  "Ventricle",         0.0,   3.5,   2.50,  2.10,   0, "csf"),
        (5,  "Pineal Gland",      0.0,  -1.0,   0.46,  0.46,   0, "brain_gray"),
        (6,  "Tumor (Main)",      3.0,  -3.0,   1.20,  0.90,   0, "tumor"),
        (7,  "Tumor (Small)",    -2.5,  -4.5,   0.60,  0.50,  15, "tumor"),
        (8,  "Cyst",              1.5,   2.0,   0.70,  0.55,   0, "cyst"),
        (9,  "Calcification",     0.0,   1.0,   0.23,  0.46,   0, "calcification"),
        (10, "Lesion",           -3.5,   1.5,   0.40,  0.30,  20, "brain_gray"),
    ]
    out = []
    for sid, name, cx, cy, a, b, ang, tissue in rows:
        p = TISSUE_PROPS[tissue]
        out.append({"shape_id": sid, "name": name, "center": [cx, cy],
                    "semi_axes": [a, b], "angle_deg": ang, "tissue": tissue,
                    "acoustic_impedance": p["z"], "attenuation_coeff": p["att"],
                    "sound_speed": p["c"], "density": p["rho"],
                    "echogenicity": p["echo"]})
    return out


class BloodVessel:
    _next_id = 2

    def __init__(self, vessel_id, start, end, radius_cm=0.30,
                 velocity_m_s=0.40, direction_deg=0.0, name=""):
        self.vessel_id     = int(vessel_id)
        self.name          = name or f"Vessel {vessel_id}"
        self.start         = [float(start[0]), float(start[1])]
        self.end           = [float(end[0]),   float(end[1])]
        self.radius_cm     = float(radius_cm)
        self.velocity_m_s  = float(velocity_m_s)
        self.direction_deg = float(direction_deg % 360)
        self._deleted      = False

    def set_velocity(self, v):  self.velocity_m_s  = float(np.clip(v, 0.0, 5.0))
    def set_direction(self, d): self.direction_deg = float(d % 360)
    def set_radius(self, r):    self.radius_cm     = float(np.clip(r, 0.05, 2.5))
    def set_start(self, x, y): self.start = [float(x), float(y)]
    def set_end(self, x, y):   self.end   = [float(x), float(y)]
    def delete(self):           self._deleted = True
    def is_deleted(self):       return self._deleted

    def get_center(self):
        return [(self.start[0]+self.end[0])/2.0,
                (self.start[1]+self.end[1])/2.0]

    def get_flow_unit_vector(self):
        """Flow direction in phantom coords: +x=right, +y=DOWN."""
        rad = deg2rad(self.direction_deg)
        return (math.cos(rad), math.sin(rad))

    def doppler_signed(self, beam_dx: float, beam_dy: float):
        """
        Returns (theta_deg, cos_theta, fd_sign):
          - theta_deg: angle between beam and flow [0, 90]
          - cos_theta: |cos| of that angle  [0, 1]
          - fd_sign: +1 if flow has component TOWARD probe (against beam), -1 if away

        Toward probe means flow opposes the beam direction → positive Doppler shift.
        """
        fdx, fdy = self.get_flow_unit_vector()
        # SIGNED dot product: >0 means flow is in same direction as beam (away from probe)
        signed_dot = beam_dx * fdx + beam_dy * fdy
        # θ is angle between beam and flow, range [0°,90°] always
        cos_theta = abs(signed_dot)
        cos_theta = float(np.clip(cos_theta, 0.0, 1.0))
        theta_deg = math.degrees(math.acos(cos_theta))
        # fd_sign: if signed_dot < 0, flow is toward probe → positive shift
        fd_sign = -1.0 if signed_dot >= 0.0 else 1.0
        return theta_deg, cos_theta, fd_sign

    def to_dict(self):
        return {"vessel_id": self.vessel_id, "name": self.name,
                "start": self.start, "end": self.end,
                "radius_cm": round(self.radius_cm, 3),
                "velocity_m_s": round(self.velocity_m_s, 3),
                "direction_deg": round(self.direction_deg, 1),
                "deleted": self._deleted}


class SheppLoganPhantom:
    IMAGE_SIZE_CM = 20.0

    def __init__(self):
        self.shapes = _make_shepp_logan_shapes()
        self.blood_vessels = [
            BloodVessel(0, [-6.0, 2.0], [6.0, 2.0], radius_cm=0.40,
                        velocity_m_s=0.50, direction_deg=0.0,
                        name="Cerebral Artery"),
            BloodVessel(1, [1.5, -7.5], [1.5, 7.5], radius_cm=0.30,
                        velocity_m_s=0.70, direction_deg=90.0,
                        name="Jugular Vein"),
        ]

    @staticmethod
    def _point_in_ellipse(px, py, shape):
        cx, cy = shape["center"]
        a, b   = shape["semi_axes"]
        ang    = deg2rad(shape["angle_deg"])
        dx, dy = px - cx, py - cy
        rx =  dx*math.cos(ang) + dy*math.sin(ang)
        ry = -dx*math.sin(ang) + dy*math.cos(ang)
        return (rx/max(a,1e-9))**2 + (ry/max(b,1e-9))**2 <= 1.0

    def get_tissue_at(self, x_cm, y_cm):
        for s in reversed(self.shapes):
            if self._point_in_ellipse(x_cm, y_cm, s):
                return s
        return None

    def render_image(self, res=200):
        half = self.IMAGE_SIZE_CM / 2.0
        xs = np.linspace(-half, half, res)
        ys = np.linspace(-half, half, res)
        img = np.zeros((res, res), dtype=float)
        for ri, y in enumerate(ys):
            for ci, x in enumerate(xs):
                s = self.get_tissue_at(x, y)
                if s:
                    img[ri, ci] = s["echogenicity"]
        return img.tolist()

    def update_shape(self, shape_id, params_dict):
        for s in self.shapes:
            if s["shape_id"] == shape_id:
                for k, v in params_dict.items():
                    if k in s:
                        s[k] = v
                return True
        return False

    def delete_shape(self, shape_id):
        """Remove a shape from the phantom (cannot remove skull/brain)."""
        protected = {0, 1}  # skull and brain tissue
        for i, s in enumerate(self.shapes):
            if s["shape_id"] == shape_id and shape_id not in protected:
                self.shapes.pop(i)
                return True
        return False

    def add_vessel(self, start, end, radius_cm=0.30, velocity_m_s=0.40,
                   direction_deg=0.0, name=""):
        vid = BloodVessel._next_id
        BloodVessel._next_id += 1
        v = BloodVessel(vid, start, end, radius_cm, velocity_m_s,
                        direction_deg, name or f"Vessel {vid}")
        self.blood_vessels.append(v)
        return v

    def delete_vessel(self, vessel_id):
        for v in self.blood_vessels:
            if v.vessel_id == vessel_id:
                v.delete()
                return True
        return False

    def get_active_vessels(self):
        return [v for v in self.blood_vessels if not v.is_deleted()]


class UltrasoundProbe:
    """
    Probe on top edge. +y = DOWN in image coords.
    beam_angle_deg: tilt from vertical. 0=straight down, +30=tilted right.
    """
    PROBE_Y_CM = -9.5

    def __init__(self):
        self.x_cm           = 0.0
        self.beam_angle_deg = 0.0
        self.focal_depth_cm = 8.0   # user-adjustable focal zone depth

    @property
    def top_y_cm(self):
        return self.PROBE_Y_CM

    def move_to(self, x_cm, beam_angle=None, focal_depth=None):
        self.x_cm = float(np.clip(x_cm, -9.5, 9.5))
        if beam_angle is not None:
            self.beam_angle_deg = float(np.clip(beam_angle, -45.0, 45.0))
        if focal_depth is not None:
            self.focal_depth_cm = float(np.clip(focal_depth, 1.0, 18.0))

    def move(self, delta_cm=0.0, beam_angle=None):
        self.move_to(self.x_cm + delta_cm, beam_angle)

    def get_beam_direction(self):
        """Unit vector: (sin(angle), cos(angle)) where +y is DOWN."""
        rad = deg2rad(self.beam_angle_deg)
        return math.sin(rad), math.cos(rad)

    def to_dict(self):
        return {"x_cm": round(self.x_cm, 2),
                "beam_angle_deg": round(self.beam_angle_deg, 1),
                "focal_depth_cm": round(self.focal_depth_cm, 1),
                "position": [round(self.x_cm, 2), round(self.PROBE_Y_CM, 2)]}


class AmodeEngine:
    N_SAMPLES = 600

    def __init__(self, phantom: SheppLoganPhantom, probe: UltrasoundProbe,
                 params: WaveformParams):
        self.phantom = phantom
        self.probe   = probe
        self.params  = params

    def compute(self):
        N        = self.N_SAMPLES
        freq     = self.params.frequency_hz
        freq_mhz = freq / 1e6
        snr_db   = self.params.snr_db
        pulse_sec    = self.params.pulse_width
        # Axial resolution = c * pulse_duration / 2
        axial_res_cm = max(0.03, (SOUND_SPEED_TISSUE * pulse_sec) / 2.0 * 100.0)

        focal_depth  = self.probe.focal_depth_cm
        px, py       = self.probe.x_cm, self.probe.top_y_cm
        bdx, bdy     = self.probe.get_beam_direction()

        depths_cm      = np.linspace(0.0, MAX_DEPTH_CM, N)
        ds             = depths_cm[1] - depths_cm[0]
        rf             = np.zeros(N)
        cumulative_att = 0.0
        prev_tissue    = None
        prev_z         = 0.0

        for i, d_cm in enumerate(depths_cm):
            x_cm = px + bdx * d_cm
            y_cm = py + bdy * d_cm

            curr_tissue = self.phantom.get_tissue_at(x_cm, y_cm)
            curr_z      = curr_tissue["acoustic_impedance"] if curr_tissue else 0.0
            att_coeff   = curr_tissue["attenuation_coeff"]  if curr_tissue else 0.0

            cumulative_att += att_coeff * freq_mhz * ds
            att_factor      = 10.0 ** (-cumulative_att / 20.0)

            # Focal zone gain: beam is strongest at focal depth
            # Gaussian beam profile: max at focal depth, falls off with divergence
            focal_gain = math.exp(-0.5 * ((d_cm - focal_depth) / max(focal_depth * 0.5, 1.0))**2)
            focal_gain = 0.4 + 0.6 * focal_gain   # min 40% max 100%

            # TGC: compensates attenuation so deep structures remain visible
            tgc = min(10.0, 10.0 ** (0.007 * d_cm))

            # Interface reflection
            if prev_tissue is not curr_tissue and curr_z > 0.0 and prev_z > 0.0:
                z_sum = curr_z + prev_z
                rc    = abs(curr_z - prev_z) / z_sum
                echo  = float(np.clip(rc * att_factor * tgc * focal_gain, 0.0, 1.0))
                # Broaden spike by axial resolution
                half = max(1, int(axial_res_cm / ds / 2))
                for di in range(-half, half+1):
                    ii = i + di
                    if 0 <= ii < N:
                        w = math.exp(-0.5*(di/max(half,1))**2)
                        rf[ii] += echo * w

            # Backscatter
            if curr_tissue and curr_tissue["echogenicity"] > 0.0:
                rf[i] += curr_tissue["echogenicity"] * 0.18 * att_factor * focal_gain

            # Vessel wall echoes
            for vessel in self.phantom.get_active_vessels():
                vc   = vessel.get_center()
                dist = math.sqrt((x_cm-vc[0])**2 + (y_cm-vc[1])**2)
                if dist < vessel.radius_cm * 2.0:
                    fade = 1.0 - dist / max(vessel.radius_cm, 1e-3)
                    rf[i] += 0.45 * max(0.0, fade) * att_factor * focal_gain

            prev_tissue = curr_tissue
            prev_z      = curr_z

        # Normalise before adding noise
        rf_max = rf.max()
        has_signal = rf_max > 1e-6   # True when beam hits actual tissue

        if has_signal:
            rf = rf / rf_max

        # AWGN — only add meaningful noise when there IS a real signal.
        # When the beam travels through free space (no tissue), suppress noise
        # to near-zero so the A-mode line stays flat at 0 as expected.
        snr_lin  = 10.0 ** (snr_db / 20.0)
        noise    = np.random.normal(0.0, 1.0 / snr_lin, N)
        if has_signal:
            rf_noisy = np.clip(rf + noise * 0.3, -1.0, 1.0)
        else:
            # Beam in free space: show essentially flat zero (tiny residual noise)
            rf_noisy = noise * 0.005

        # Envelope detection
        envelope = np.abs(rf_noisy)
        k = max(3, int(axial_res_cm / ds) | 1)
        envelope = np.convolve(envelope, np.ones(k)/k, mode='same')
        # Normalise envelope only when there's a real signal
        env_max = envelope.max()
        if has_signal and env_max > 0:
            envelope = envelope / env_max
        elif not has_signal:
            envelope = np.zeros(N)  # pure flat line in free space

        # Focal point depth index for frontend marker
        focal_idx = int(np.clip(focal_depth / MAX_DEPTH_CM * N, 0, N-1))

        return {"rf_line": rf_noisy.tolist(), "envelope": envelope.tolist(),
                "depth_cm": depths_cm.tolist(),
                "probe_position": [round(px,2), round(py,2)],
                "beam_angle_deg": round(self.probe.beam_angle_deg,1),
                "focal_depth_cm": round(focal_depth, 1),
                "focal_idx": focal_idx,
                "max_depth_cm": MAX_DEPTH_CM,
                "axial_res_cm": round(axial_res_cm, 3),
                "beam_direction": [round(bdx,4), round(bdy,4)]}


class BmodeEngine:
    def __init__(self, phantom: SheppLoganPhantom, params: WaveformParams):
        self.phantom = phantom
        self.params  = params

    def compute(self, x_start_cm=-9.0, x_end_cm=9.0, steps=96,
                beam_angle_deg=0.0, focal_depth_cm=8.0):
        probe = UltrasoundProbe()
        probe.focal_depth_cm = focal_depth_cm
        amode = AmodeEngine(self.phantom, probe, self.params)
        N_depth = 300
        bmode = np.zeros((N_depth, steps))
        xs = np.linspace(x_start_cm, x_end_cm, steps)

        for col, x in enumerate(xs):
            probe.move_to(x, beam_angle_deg)
            result = amode.compute()
            env = np.array(result["envelope"])
            idx = np.linspace(0, len(env)-1, N_depth).astype(int)
            bmode[:, col] = env[idx]

        # Lateral blurring based on focal depth
        depth_vals = np.linspace(0, MAX_DEPTH_CM, N_depth)
        for row in range(N_depth):
            defocus  = abs(depth_vals[row] - focal_depth_cm)
            blur_pix = max(1.0, defocus / max(focal_depth_cm, 1.0) * steps * 0.05)
            if blur_pix > 1.2:
                k = int(blur_pix*2) | 1
                kv = np.exp(-0.5*(np.arange(k)-k//2)**2/(blur_pix/2)**2)
                kv /= kv.sum()
                bmode[row,:] = np.convolve(bmode[row,:], kv, mode='same')

        # Strong log-compression for bright display
        bmode = np.log1p(bmode * 50) / np.log1p(50)
        bmode = np.clip(bmode, 0.0, 1.0)

        return {"bmode_image": bmode.tolist(),
                "x_range_cm": [float(x_start_cm), float(x_end_cm)],
                "depth_range_cm": [0.0, MAX_DEPTH_CM],
                "steps": steps,
                "beam_angle_deg": beam_angle_deg,
                "focal_depth_cm": focal_depth_cm}


class DopplerEngine:
    """
    CORRECT Doppler physics:
      Δf = 2 * f0 * v * cos(θ) / c
      θ = angle between beam direction and flow direction
      Sign: flow toward probe → positive Δf (peak on right side)
            flow away from probe → negative Δf (peak on left side)
            θ=90° → cos(θ)=0 → Δf=0 (NO SHIFT, always) ✓
    """
    N_FREQS = 512

    def __init__(self, params: WaveformParams):
        self.params  = params
        self._vessel = None

    def set_vessel(self, vessel: BloodVessel):
        self._vessel = vessel

    def compute(self, probe: UltrasoundProbe):
        vessel = self._vessel
        if vessel is None or vessel.is_deleted():
            return self._empty()

        freq   = self.params.frequency_hz
        snr_db = self.params.snr_db
        c      = SOUND_SPEED_TISSUE

        # Beam unit vector (+y = DOWN in image)
        bdx, bdy = probe.get_beam_direction()

        # ── Range gate ────────────────────────────────────────────────────
        # Find closest point on beam ray to vessel center
        vx, vy   = vessel.get_center()
        px, py   = probe.x_cm, probe.top_y_cm
        t_star   = max(0.0, (vx-px)*bdx + (vy-py)*bdy)
        cp_x, cp_y = px + t_star*bdx, py + t_star*bdy
        dist     = math.sqrt((cp_x-vx)**2 + (cp_y-vy)**2)
        gate_r   = vessel.radius_cm * 5.0
        in_gate  = dist <= gate_r

        # ── Doppler angle with sign ───────────────────────────────────────
        theta_deg, cos_theta, fd_sign = vessel.doppler_signed(bdx, bdy)

        v_blood  = vessel.velocity_m_s
        # SIGNED Doppler shift: positive = toward probe, negative = away
        fd = fd_sign * 2.0 * freq * v_blood * cos_theta / c

        # ── Spectrum ──────────────────────────────────────────────────────
        f_max  = 2.0 * freq * 4.0 / c
        freqs  = np.linspace(-f_max, f_max, self.N_FREQS)

        # Suppress spectrum when perpendicular (cos_theta ≈ 0 → fd ≈ 0 → no detectable shift)
        has_flow_signal = in_gate and v_blood > 1e-4 and cos_theta > 0.02

        if has_flow_signal:
            sigma = max(abs(fd)*0.08 + float(freq)*0.00002, 50.0)
            power = np.exp(-0.5 * ((freqs - fd) / sigma)**2)
            # Small wall-filter artefact near zero
            power += 0.05 * np.exp(-0.5 * (freqs / max(sigma*3,1))**2)
            proximity = max(0.05, 1.0 - dist/gate_r)
            power *= proximity
        else:
            # No vessel / perpendicular / zero velocity → truly flat zero
            power = np.zeros(self.N_FREQS)

        if has_flow_signal:
            # Only add noise when real flow is present
            snr_lin = 10.0**(snr_db/20.0)
            noise   = np.abs(np.random.normal(0.0, 1.0/snr_lin, self.N_FREQS))
            power   = np.clip(power + noise, 0.0, None)
            power  /= max(power.max(), 1e-9)
        # else: power is already zeros → flat line at 0, no noise

        # Estimated velocity from peak
        peak_f = float(freqs[int(np.argmax(power))])
        if in_gate and cos_theta > 0.01:
            est_vel = peak_f * c / (2.0 * freq * cos_theta)  # signed!
        else:
            est_vel = 0.0

        fdx, fdy = vessel.get_flow_unit_vector()
        return {
            "frequencies":            freqs.tolist(),
            "power":                  power.tolist(),
            "peak_shift_hz":          round(float(fd), 2),
            "estimated_velocity_m_s": round(float(est_vel), 4),
            "theta_deg":              round(float(theta_deg), 1),
            "cos_theta":              round(float(cos_theta), 4),
            "in_gate":                bool(in_gate),
            "dist_to_vessel_cm":      round(float(dist), 2),
            "gate_radius_cm":         round(float(gate_r), 2),
            "vessel_id":              vessel.vessel_id,
            "vessel_name":            vessel.name,
            "beam_dir":               [round(float(bdx),4), round(float(bdy),4)],
            "flow_dir":               [round(float(fdx),4), round(float(fdy),4)],
            "fd_sign":                float(fd_sign),
        }

    def _empty(self):
        freqs = np.linspace(-20000, 20000, self.N_FREQS)
        return {
            "frequencies": freqs.tolist(),
            "power": np.zeros(self.N_FREQS).tolist(),
            "peak_shift_hz": 0.0, "estimated_velocity_m_s": 0.0,
            "theta_deg": 0.0, "cos_theta": 0.0,
            "in_gate": False, "dist_to_vessel_cm": 999.0, "gate_radius_cm": 0.0,
            "vessel_id": -1, "vessel_name": "none",
            "beam_dir": [0.0, 1.0], "flow_dir": [1.0, 0.0], "fd_sign": 0.0,
        }


class UltrasoundSimulator(BaseSimulator):

    def __init__(self, params: WaveformParams = None):
        if params is None:
            params = WaveformParams(
                frequency_hz=US_FREQ_DEFAULT, num_elements=64,
                element_spacing=0.5, snr_db=40.0, amplitude=1.0, pulse_width=5e-7)
        super().__init__(params)
        self.phantom = SheppLoganPhantom()
        self.probe   = UltrasoundProbe()
        self.amode   = AmodeEngine(self.phantom, self.probe, params)
        self.bmode   = BmodeEngine(self.phantom, params)
        self.doppler = DopplerEngine(params)
        self.doppler.set_vessel(self.phantom.blood_vessels[0])

    def initialize(self):
        self.probe = UltrasoundProbe()
        self.amode = AmodeEngine(self.phantom, self.probe, self.params)
        self.bmode = BmodeEngine(self.phantom, self.params)
        self.doppler = DopplerEngine(self.params)
        active = self.phantom.get_active_vessels()
        if active:
            self.doppler.set_vessel(active[0])

    def step(self):
        self._step_count += 1
        return self.get_output()

    def reset(self):
        self._step_count = 0
        self.phantom = SheppLoganPhantom()
        BloodVessel._next_id = 2
        self.initialize()

    def update_params(self, params):
        self.params = params
        self.amode.params = params
        self.bmode.params = params
        self.doppler.params = params

    def get_phantom_image(self):
        img = self.phantom.render_image(res=150)
        return {"image": img, "shapes": self.phantom.shapes,
                "vessels": [v.to_dict() for v in self.phantom.blood_vessels],
                "probe": self.probe.to_dict(), "size_cm": self.phantom.IMAGE_SIZE_CM}

    def get_amode(self):          return self.amode.compute()
    def get_bmode(self, x_start=-9.0, x_end=9.0, steps=96, focal_depth_cm=None):
        fd = focal_depth_cm if focal_depth_cm is not None else self.probe.focal_depth_cm
        return self.bmode.compute(float(x_start), float(x_end), int(steps),
                                  self.probe.beam_angle_deg, float(fd))
    def get_bmode_live(self):
        return self.bmode.compute(-9.0, 9.0, 64, self.probe.beam_angle_deg,
                                  self.probe.focal_depth_cm)

    def get_doppler(self, vessel_id=None):
        if vessel_id is not None:
            v = next((v for v in self.phantom.get_active_vessels()
                      if v.vessel_id == vessel_id), None)
            if v:
                self.doppler.set_vessel(v)
        return self.doppler.compute(self.probe)

    def set_probe_absolute(self, x_cm, beam_angle=None, focal_depth=None):
        self.probe.move_to(float(x_cm), beam_angle, focal_depth)
        return self.probe.to_dict()

    def move_probe(self, delta_cm=0.0, beam_angle=None):
        self.probe.move(float(delta_cm), beam_angle)
        return self.probe.to_dict()

    def add_vessel(self, start_x, start_y, end_x, end_y,
                   radius_cm=0.3, velocity_m_s=0.4, direction_deg=0.0, name=""):
        v = self.phantom.add_vessel([start_x,start_y],[end_x,end_y],
                                    radius_cm,velocity_m_s,direction_deg,name)
        return v.to_dict()

    def delete_vessel(self, vessel_id):
        ok = self.phantom.delete_vessel(vessel_id)
        if not ok: return {"error": f"Vessel {vessel_id} not found"}
        active = self.phantom.get_active_vessels()
        if active: self.doppler.set_vessel(active[0])
        return {"deleted": vessel_id}

    def update_vessel(self, vessel_id, **kwargs):
        v = next((v for v in self.phantom.get_active_vessels()
                  if v.vessel_id == vessel_id), None)
        if v is None: return {"error": f"Vessel {vessel_id} not found"}
        if "velocity"  in kwargs: v.set_velocity(kwargs["velocity"])
        if "direction" in kwargs: v.set_direction(kwargs["direction"])
        if "radius_cm" in kwargs: v.set_radius(kwargs["radius_cm"])
        if "start_x" in kwargs and "start_y" in kwargs:
            v.set_start(kwargs["start_x"], kwargs["start_y"])
        if "end_x" in kwargs and "end_y" in kwargs:
            v.set_end(kwargs["end_x"], kwargs["end_y"])
        if "name" in kwargs: v.name = str(kwargs["name"])
        self.doppler.set_vessel(v)
        return v.to_dict()

    def delete_shape(self, shape_id):
        ok = self.phantom.delete_shape(shape_id)
        return {"deleted": shape_id} if ok else {"error": f"Cannot delete shape {shape_id}"}

    def get_output(self):
        return {"probe": self.probe.to_dict(),
                "vessels": [v.to_dict() for v in self.phantom.blood_vessels],
                "params": self.params.to_dict()}