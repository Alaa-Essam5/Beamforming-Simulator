"""
5G Simulator module.
Implements network topology, beam connectivity, auto-parameter updates, 
and coverage mapping for multi-user, multi-tower scenarios.
"""
import numpy as np
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from base_simulator import BaseSimulator
from waveform_params import WaveformParams
from beamforming_engine import BeamformingEngine
from math_utils import distance, angle_between, path_loss_db, array_factor


class UserEquipment:
    """Represents a 5G network user (UE) that can move within the simulation."""
    
    def __init__(self, user_id: int, name: str, position: tuple):
        self.user_id = user_id
        self.name = name
        self.position = list(position)
        self.connected_tower_id = None
        self.connected_sector_id = None
        self.received_snr_db = 0.0
        self.signal_strength_dbm = -120.0

    def get_state(self) -> dict:
        """Serialize user state for frontend consumption."""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "position": self.position,
            "connected_tower_id": self.connected_tower_id,
            "connected_sector_id": self.connected_sector_id,
            "received_snr_db": self.received_snr_db,
            "signal_strength_dbm": self.signal_strength_dbm
        }


class Tower:
    """Represents a 5G Base Station (gNB) with its own beamforming engine."""
    NUM_SECTORS = 3
    SECTOR_WIDTH_DEG = 120.0
    SECTOR_CENTER_ANGLES_DEG = (-120.0, 0.0, 120.0)
    MIN_LINK_DISTANCE_M = 1.0
    
    def __init__(self, tower_id: int, name: str, position: tuple, base_params: WaveformParams):
        self.tower_id = tower_id
        self.name = name
        self.position = list(position)
        
        # Give each tower its own independent copy of the waveform parameters & engine
        self.params = WaveformParams.from_dict(base_params.to_dict())
        self.engine = BeamformingEngine(self.params)
        self.connected_user_ids = []

    @staticmethod
    def _normalize_angle(angle_deg: float) -> float:
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg < -180:
            angle_deg += 360
        return float(angle_deg)

    @classmethod
    def get_sector_for_angle(cls, angle_deg: float) -> tuple[int, float]:
        """
        Return non-overlapping 120° sector ownership and sector center.
        Sectors are hard partitions of the full 360°:
        S0: [-180, -60), S1: [-60, 60), S2: [60, 180].
        """
        a = cls._normalize_angle(angle_deg)
        idx = int(math.floor((a + 180.0) / cls.SECTOR_WIDTH_DEG))
        idx = max(0, min(cls.NUM_SECTORS - 1, idx))
        return idx, float(cls.SECTOR_CENTER_ANGLES_DEG[idx])

    @classmethod
    def clamp_link_distance(cls, distance_m: float) -> float:
        """Avoid singular near-field path loss values."""
        return max(cls.MIN_LINK_DISTANCE_M, float(distance_m))

    def steer_to(self, angle_deg: float) -> None:
        """Automatically update steering parameters to target a user."""
        self.params.steering_angle_deg = angle_deg
        self.engine.steer_beam(angle_deg)

    def get_state(self) -> dict:
        """Fetch current live beamforming parameters and connection status."""
        profile = self.engine.compute_beam_profile(resolution=90)
        return {
            "tower_id": self.tower_id,
            "name": self.name,
            "position": self.position,
            "frequency_hz": self.params.frequency_hz,
            "steering_angle_deg": self.params.steering_angle_deg,
            "num_elements": self.params.num_elements,
            "element_spacing": self.params.element_spacing,
            "amplitude": self.params.amplitude,
            "apodization_window": self.params.apodization_window,
            "beamwidth_deg": profile.get("beamwidth_deg", 0.0),
            "sidelobe_db": profile.get("sidelobe_level_db", -999.0),
            "num_arrays": self.NUM_SECTORS,
            "sector_width_deg": self.SECTOR_WIDTH_DEG,
            "sector_centers_deg": list(self.SECTOR_CENTER_ANGLES_DEG),
            "connected_user_ids": self.connected_user_ids
        }


class CoverageMapper:
    """Calculates the 2D signal strength heatmap over the simulation area."""

    def __init__(self, towers: list[Tower]):
        self.towers = towers

    @staticmethod
    def _tower_sectorized_gain_db(t: Tower, angles_deg: np.ndarray) -> np.ndarray:
        """
        Compute per-angle directivity (dB) using strict non-overlapping
        120° sector ownership (no sector overlap).
        """
        p = t.params
        n = np.arange(p.num_elements)
        apo = np.array(t.engine.apodizer.get_weights(), dtype=np.complex128)
        sector_profiles_db = {}
        for sidx, center_deg in enumerate(t.SECTOR_CENTER_ANGLES_DEG):
            center_rad = np.deg2rad(center_deg)
            steering_vec = np.exp(-1j * 2 * np.pi * p.element_spacing * n * np.cos(center_rad))
            weights = apo * steering_vec
            mag = array_factor(
                num_elements=p.num_elements,
                spacing_wl=p.element_spacing,
                angles_deg=angles_deg,
                weights=weights,
            )
            sector_profiles_db[sidx] = 20 * np.log10(np.clip(mag, 1e-12, None))

        # Assign each angle to exactly one 120° sector.
        flat_angles = np.asarray(angles_deg).reshape(-1)
        owned_gain = np.zeros_like(flat_angles, dtype=float)
        for i, a in enumerate(flat_angles):
            sidx, _ = t.get_sector_for_angle(float(a))
            owned_gain[i] = float(sector_profiles_db[sidx].reshape(-1)[i])
        return owned_gain.reshape(np.asarray(angles_deg).shape)

    def compute_coverage_map(
        self,
        extent_m: float = 600.0,
        grid_size: int = 50,
        max_influence_radius: float = 500.0,
    ) -> dict:
        """Coverage map: each point is served by the geographically nearest in-range tower."""
        x = np.linspace(-extent_m, extent_m, grid_size)
        y = np.linspace(-extent_m, extent_m, grid_size)
        X, Y = np.meshgrid(x, y)

        if not self.towers:
            return {
                "coverage_map": np.zeros((grid_size, grid_size)).tolist(),
                "tower_map": np.full((grid_size, grid_size), -1, dtype=int).tolist(),
            }

        dist_stack = np.stack(
            [
                np.sqrt((X - t.position[0]) ** 2 + (Y - t.position[1]) ** 2)
                for t in self.towers
            ],
            axis=0,
        )
        in_range = dist_stack <= max_influence_radius
        dist_for_argmin = np.where(in_range, dist_stack, np.inf)
        tower_map = np.argmin(dist_for_argmin, axis=0).astype(int)
        no_service = ~np.any(in_range, axis=0)
        tower_map[no_service] = -1

        coverage = np.full(X.shape, -120.0)
        c = 3e8

        for ti, t in enumerate(self.towers):
            mask = tower_map == t.tower_id
            if not np.any(mask):
                continue

            dist = np.maximum(dist_stack[ti], Tower.MIN_LINK_DISTANCE_M)
            angles_rad = np.arctan2(Y - t.position[1], X - t.position[0])
            angles_deg = np.degrees(angles_rad)
            mag_db = self._tower_sectorized_gain_db(t, angles_deg)
            wl = c / t.params.frequency_hz
            pl = 20 * np.log10(4 * np.pi * dist / wl)
            sig = 40.0 + mag_db - pl
            coverage[mask] = sig[mask]

        # Normalize coverage specifically for the frontend heatmap scaling
        min_sig, max_sig = -100.0, -40.0
        norm_coverage = np.clip((coverage - min_sig) / (max_sig - min_sig), 0, 1)

        return {
            "coverage_map": norm_coverage.tolist(),
            "tower_map": tower_map.tolist()
        }


class FiveGSimulator(BaseSimulator):
    """
    Core 5G routing and processing environment.
    Enforces the BaseSimulator contract.
    """

    def __init__(self, params: WaveformParams = None):
        super().__init__(params or WaveformParams())
        self.towers: list[Tower] = []
        self.users: list[UserEquipment] = []
        self.coverage_mapper = CoverageMapper(self.towers)
        self.beam_profiles = {}
        self.tower_beam_profiles = {}
        self.tower_sector_beam_profiles = {}
        self.max_influence_radius = 500.0  # 500 meters

    @staticmethod
    def _normalize_angle(angle_deg: float) -> float:
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg < -180:
            angle_deg += 360
        return float(angle_deg)

    def _compute_tower_mimo_profile(
        self,
        tower: Tower,
        user_angles_deg: list[float],
        resolution: int = 181,
        sector_idx: int | None = None,
    ) -> dict:
        """
        Compute a single composite MU-MIMO beam profile for a tower by
        superposing steering vectors for all scheduled users on the same array.
        """
        p = tower.params
        n = np.arange(p.num_elements)
        apo = np.array(tower.engine.apodizer.get_weights(), dtype=np.complex128)

        combined = np.zeros(p.num_elements, dtype=np.complex128)
        for ang_deg in user_angles_deg:
            ang_rad = np.deg2rad(ang_deg)
            steering_vec = np.exp(-1j * 2 * np.pi * p.element_spacing * n * np.cos(ang_rad))
            combined += steering_vec

        if np.max(np.abs(combined)) > 0:
            combined = combined / np.max(np.abs(combined))
        mimo_weights = apo * combined

        angles = np.linspace(-180, 180, resolution)
        mag = array_factor(
            num_elements=p.num_elements,
            spacing_wl=p.element_spacing,
            angles_deg=angles,
            weights=mimo_weights,
        )

        # Enforce strict 120° non-overlapping sector ownership when requested.
        if sector_idx is not None:
            for i, a in enumerate(angles):
                owner_idx, _ = tower.get_sector_for_angle(float(a))
                if owner_idx != sector_idx:
                    mag[i] = 1e-12

        mag_db = 20 * np.log10(np.clip(mag, 1e-12, None))
        return {
            "angles": angles.tolist(),
            "magnitude": mag.tolist(),
            "magnitude_db": mag_db.tolist(),
            "beam_mode": "mimo",
            "num_users": len(user_angles_deg),
            "sector_idx": sector_idx,
        }

    def _profile_gain_db(self, profile: dict, angle_deg: float) -> float:
        """Interpolate directivity gain (dB) from a beam profile at a given angle."""
        angles = np.array(profile.get("angles", []), dtype=float)
        mag_db = np.array(profile.get("magnitude_db", []), dtype=float)
        if len(angles) == 0 or len(mag_db) == 0:
            return 0.0
        a = self._normalize_angle(angle_deg)
        return float(np.interp(a, angles, mag_db))

    def _build_sector_profiles(self, tower: Tower, resolution: int = 181) -> dict[int, dict]:
        """Build one profile per fixed 120° sector center for a tower."""
        profiles = {}
        for idx, center in enumerate(tower.SECTOR_CENTER_ANGLES_DEG):
            profiles[idx] = self._compute_tower_mimo_profile(
                tower,
                [center],
                resolution=resolution,
                sector_idx=idx,
            )
        return profiles

    def initialize(self) -> None:
        """Set up 3 base stations and 2 network users."""
        self.towers = [
            Tower(0, "gNB-Alpha", (-200.0, 150.0), self.params),
            Tower(1, "gNB-Beta",  (200.0, 150.0), self.params),
            Tower(2, "gNB-Gamma", (0.0, -200.0), self.params)
        ]
        self.coverage_mapper.towers = self.towers

        self.users = [
            UserEquipment(0, "UE-0 (Mobile)", (0.0, 0.0)),
            UserEquipment(1, "UE-1 (IoT)",    (50.0, -50.0))
        ]
        self.reset()

    def reset(self) -> None:
        """Reset mutable state to initial conditions."""
        self._step_count = 0
        for u in self.users:
            u.position = [-30.0, 500.0] if u.user_id == 0 else [50.0, -50.0]

        for t in self.towers:
            t.params = WaveformParams.from_dict(self.params.to_dict())
            t.engine.update_params(t.params)

        self._update_network()

    def step(self) -> dict:
        """Advance time step manually (mostly driven by user movement in 5G)."""
        self._step_count += 1
        self._update_network()
        return self.get_output()

    def update_params(self, params: WaveformParams) -> None:
        """Hot-reload global parameters onto all towers."""
        self.params = params
        for t in self.towers:
            t.params = WaveformParams.from_dict(params.to_dict())
            t.engine.update_params(t.params)
        self._update_network()

    def move_user(self, user_id: int, dx: float, dy: float, to_x: float = None, to_y: float = None) -> dict:
        """Handle keyboard-driven user movement and bound checking."""
        if 0 <= user_id < len(self.users):
            u = self.users[user_id]
            if to_x is not None and to_y is not None:
                u.position = [float(to_x), float(to_y)]
            else:
                u.position[0] += float(dx)
                u.position[1] += float(dy)

            # Bound check: Keep within a 1.2km x 1.2km area
            u.position[0] = max(-900.0, min(900.0, u.position[0]))
            u.position[1] = max(-900.0, min(900.0, u.position[1]))

        self._update_network()
        return self.get_output()

    def _update_network(self) -> None:
        """
        Core physics routing:
        1. Resets old links.
        2. Evaluates distances & SNR to assign the best tower.
        3. Generates multiple beam profiles if a tower has multiple users.
        """
        # Clear legacy connection state
        for u in self.users:
            u.connected_tower_id = None
            u.connected_sector_id = None
            u.received_snr_db = 0.0
            u.signal_strength_dbm = -120.0

        for t in self.towers:
            t.connected_user_ids.clear()

        self.beam_profiles.clear()
        self.tower_beam_profiles.clear()
        self.tower_sector_beam_profiles.clear()

        # Build baseline 3-sector patterns for each tower (used in attachment logic).
        sector_seed_profiles: dict[int, dict[int, dict]] = {}
        for t in self.towers:
            sector_seed_profiles[t.tower_id] = self._build_sector_profiles(t, resolution=181)

        # Track served users per tower-sector.
        tower_sector_users: dict[int, dict[int, list[int]]] = {
            t.tower_id: {s: [] for s in range(t.NUM_SECTORS)} for t in self.towers
        }

        # Step 1: Link Attachment — nearest in-range tower (distance), then sector + RSS
        for u in self.users:
            candidates: list[tuple[float, int, Tower]] = []
            for t in self.towers:
                d_raw = distance(u.position, t.position)
                if d_raw > self.max_influence_radius:
                    continue
                candidates.append((d_raw, t.tower_id, t))

            if not candidates:
                continue

            candidates.sort(key=lambda x: (x[0], x[1]))
            best_tower = candidates[0][2]
            d = Tower.clamp_link_distance(candidates[0][0])

            ang = angle_between(best_tower.position, u.position)
            sector_idx, _ = best_tower.get_sector_for_angle(ang)
            seed_profile = sector_seed_profiles[best_tower.tower_id][sector_idx]
            dir_gain_db = self._profile_gain_db(seed_profile, ang)
            pl = path_loss_db(d, best_tower.params.frequency_hz)
            best_sig = 40.0 + dir_gain_db - pl
            # SNR slider (params.snr_db) represents the transmit SNR budget.
            # Received SNR = TX SNR budget - path loss + directivity gain.
            best_snr = best_tower.params.snr_db - pl + dir_gain_db

            u.connected_tower_id = best_tower.tower_id
            u.connected_sector_id = sector_idx
            u.signal_strength_dbm = best_sig
            u.received_snr_db = best_snr
            best_tower.connected_user_ids.append(u.user_id)
            tower_sector_users[best_tower.tower_id][sector_idx].append(u.user_id)

        # Step 2: MU-MIMO beamforming per tower-sector (3 arrays per tower)
        for t in self.towers:
            if not t.connected_user_ids:
                # Keep an idle tower profile for compatibility.
                self.tower_beam_profiles[f"T{t.tower_id}"] = sector_seed_profiles[t.tower_id][1]
                continue

            sector_profiles = {}
            all_user_angles = []

            for sector_idx in range(t.NUM_SECTORS):
                uids = tower_sector_users[t.tower_id][sector_idx]
                if uids:
                    user_angles = [angle_between(t.position, self.users[uid].position) for uid in uids]
                    profile = self._compute_tower_mimo_profile(
                        t,
                        user_angles,
                        resolution=181,
                        sector_idx=sector_idx,
                    )
                    all_user_angles.extend(user_angles)
                else:
                    profile = sector_seed_profiles[t.tower_id][sector_idx]

                sector_profiles[sector_idx] = profile
                self.tower_sector_beam_profiles[f"T{t.tower_id}-S{sector_idx}"] = profile

            # Preserve existing key for tower-level consumers using the busiest sector.
            busiest_sector = max(
                range(t.NUM_SECTORS),
                key=lambda s: len(tower_sector_users[t.tower_id][s])
            )
            self.tower_beam_profiles[f"T{t.tower_id}"] = sector_profiles[busiest_sector]

            # Preserve existing pair keys: each user gets its own serving-sector profile.
            for uid in t.connected_user_ids:
                u = self.users[uid]
                profile = sector_profiles[u.connected_sector_id]
                self.beam_profiles[f"T{t.tower_id} - UE{uid}"] = profile

            # UI steering angle: mean direction of all active served users.
            mean_sin = np.mean([np.sin(np.deg2rad(a)) for a in all_user_angles])
            mean_cos = np.mean([np.cos(np.deg2rad(a)) for a in all_user_angles])
            t.steer_to(np.degrees(np.arctan2(mean_sin, mean_cos)))

            # Update per-user link metrics with their serving-sector MU-MIMO gain.
            for uid in t.connected_user_ids:
                u = self.users[uid]
                d = distance(u.position, t.position)
                d = Tower.clamp_link_distance(d)
                pl = path_loss_db(d, t.params.frequency_hz)
                ang = angle_between(t.position, u.position)
                profile = sector_profiles[u.connected_sector_id]
                uids_sec = tower_sector_users[t.tower_id][u.connected_sector_id]
                pug = profile.get("per_user_gains_db") or []

                if (
                    profile.get("beam_mode") == "mu_mimo_zf"
                    and len(uids_sec) > 1
                    and len(pug) == len(uids_sec)
                ):
                    dir_gain_db = float(pug[uids_sec.index(uid)])
                else:
                    dir_gain_db = self._profile_gain_db(profile, ang)

                sig = 40.0 + dir_gain_db - pl
                # Received SNR = TX SNR budget (slider) - path loss + directivity gain.
                snr = t.params.snr_db - pl + dir_gain_db
                u.signal_strength_dbm = sig
                u.received_snr_db = snr

    def get_output(self) -> dict:
        """Return the fully serialized state matching frontend API expectations."""
        return {
            "towers": [t.get_state() for t in self.towers],
            "users": [u.get_state() for u in self.users],
            "connectivity": {u.user_id: u.connected_tower_id for u in self.users},
            "beam_profiles": self.beam_profiles,
            "tower_beam_profiles": self.tower_beam_profiles,
            "tower_sector_beam_profiles": self.tower_sector_beam_profiles,
        }