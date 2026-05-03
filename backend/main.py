"""
FastAPI Backend — Beamforming Simulator
All simulation logic is in the app classes; this file is API routing only.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import numpy as np

from waveform_params import WaveformParams
from beamforming_engine import BeamformingEngine, Apodizer
from fiveg_simulator import FiveGSimulator
from ultrasound_simulator import UltrasoundSimulator
from radar_simulator import RadarSimulator

app = FastAPI(title="Beamforming Simulator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global simulator instances ────────────────────────────────────────────────

_params = WaveformParams()
_engine = BeamformingEngine(_params)
_fiveg  = FiveGSimulator()
_us     = UltrasoundSimulator()
_radar  = RadarSimulator()

_fiveg.initialize()
_us.initialize()
_radar.initialize()


# ── Pydantic request models ───────────────────────────────────────────────────

class ParamsRequest(BaseModel):
    num_elements: Optional[int] = None
    element_spacing: Optional[float] = None
    frequency_hz: Optional[float] = None
    steering_angle_deg: Optional[float] = None
    amplitude: Optional[float] = None
    sampling_rate: Optional[float] = None
    pulse_width: Optional[float] = None
    apodization_window: Optional[str] = None
    snr_db: Optional[float] = None

class MoveUserRequest(BaseModel):
    user_id: int
    dx: Optional[float] = 0
    dy: Optional[float] = 0
    to_x: Optional[float] = None
    to_y: Optional[float] = None

class AddBodyRequest(BaseModel):
    distance_m: float
    angle_deg: float
    size_m: Optional[float] = 5.0

class MoveBodyRequest(BaseModel):
    body_id: int
    distance_m: float
    angle_deg: float

class ResizeBodyRequest(BaseModel):
    body_id: int
    size_m: float

class ProbeRequest(BaseModel):
    delta_cm: Optional[float] = 0
    beam_angle: Optional[float] = None

class ProbeAbsoluteRequest(BaseModel):
    x_cm: float
    beam_angle: Optional[float] = None
    focal_depth: Optional[float] = None

class DeleteShapeRequest(BaseModel):
    shape_id: int

class VesselRequest(BaseModel):
    vessel_id: Optional[int] = 0
    velocity: Optional[float] = None
    direction: Optional[float] = None
    radius_cm: Optional[float] = None
    name: Optional[str] = None
    start_x: Optional[float] = None
    start_y: Optional[float] = None
    end_x: Optional[float] = None
    end_y: Optional[float] = None

class AddVesselRequest(BaseModel):
    start_x: float = -5.0
    start_y: float = 0.0
    end_x: float = 5.0
    end_y: float = 0.0
    radius_cm: Optional[float] = 0.3
    velocity_m_s: Optional[float] = 0.4
    direction_deg: Optional[float] = 0.0
    name: Optional[str] = ""

class DeleteVesselRequest(BaseModel):
    vessel_id: int

class ShapeParamsRequest(BaseModel):
    shape_id: int
    params: dict

class ScanRequest(BaseModel):
    speed: Optional[float] = None
    beam_width: Optional[float] = None


# ── Helper ────────────────────────────────────────────────────────────────────

def _apply_params(req: ParamsRequest) -> WaveformParams:
    global _params, _engine
    d = _params.to_dict()
    updates = req.dict(exclude_none=True)
    d.update(updates)
    _params = WaveformParams.from_dict(d)
    valid, msg = _params.validate()
    if not valid:
        raise HTTPException(status_code=400, detail=msg)
    _engine.update_params(_params)
    return _params


# ── Core beamforming endpoints ────────────────────────────────────────────────

@app.get("/api/params")
def get_params():
    return _params.to_dict()

@app.post("/api/params")
def set_params(req: ParamsRequest):
    p = _apply_params(req)
    return p.to_dict()

@app.get("/api/beam/profile")
def get_beam_profile():
    return _engine.compute_beam_profile()

@app.get("/api/beam/interference")
def get_interference_map(grid_size: int = 80):
    return _engine.compute_interference_map(grid_size=grid_size)

@app.post("/api/beam/steer")
def steer_beam(angle_deg: float):
    _engine.steer_beam(angle_deg)
    _params.steering_angle_deg = angle_deg
    return {"steering_angle_deg": angle_deg, "profile": _engine.compute_beam_profile()}

@app.get("/api/beam/weights")
def get_weights():
    return {"weights": _engine.get_complex_weights(),
            "positions": _engine.get_oscillator_positions()}

@app.get("/api/windows")
def list_windows():
    return {"windows": Apodizer.AVAILABLE}


# ── 5G endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/5g/state")
def fiveg_state():
    return _fiveg.get_output()

@app.post("/api/5g/params")
def fiveg_params(req: ParamsRequest):
    d = _fiveg.params.to_dict()
    d.update(req.dict(exclude_none=True))
    new_p = WaveformParams.from_dict(d)
    _fiveg.update_params(new_p)
    return _fiveg.get_output()

@app.post("/api/5g/user/move")
def fiveg_move_user(req: MoveUserRequest):
    return _fiveg.move_user(req.user_id, req.dx, req.dy, req.to_x, req.to_y)

@app.post("/api/5g/reset")
def fiveg_reset():
    _fiveg.reset()
    return _fiveg.get_output()

@app.get("/api/5g/coverage")
def fiveg_coverage():
    return _fiveg.coverage_mapper.compute_coverage_map(
        max_influence_radius=_fiveg.max_influence_radius,
    )

@app.get("/api/5g/beams")
def fiveg_beams():
    output = _fiveg.get_output()
    return {
        "beam_profiles": output.get("beam_profiles", {}),
        "tower_sector_beam_profiles": output.get("tower_sector_beam_profiles", {}),
    }


# ── Ultrasound endpoints ──────────────────────────────────────────────────────

@app.get("/api/us/phantom")
def us_phantom():
    return _us.get_phantom_image()

@app.get("/api/us/amode")
def us_amode():
    return _us.get_amode()

@app.get("/api/us/bmode")
def us_bmode(x_start: float = -9.0, x_end: float = 9.0, steps: int = 96):
    return _us.get_bmode(x_start, x_end, steps)

@app.get("/api/us/bmode/live")
def us_bmode_live():
    """Fast live B-mode centered on current probe position."""
    return _us.get_bmode_live()

@app.get("/api/us/doppler")
def us_doppler(vessel_id: Optional[int] = None):
    return _us.get_doppler(vessel_id)

@app.post("/api/us/probe/move")
def us_probe_move(req: ProbeRequest):
    return _us.move_probe(req.delta_cm, req.beam_angle)

@app.post("/api/us/probe/set")
def us_probe_set(req: ProbeAbsoluteRequest):
    """Set probe to absolute position — preferred over delta-based moves."""
    return _us.set_probe_absolute(req.x_cm, req.beam_angle, req.focal_depth)

@app.post("/api/us/shape/delete")
def us_delete_shape(req: DeleteShapeRequest):
    return _us.delete_shape(req.shape_id)

@app.get("/api/us/vessels")
def us_get_vessels():
    return {"vessels": [v.to_dict() for v in _us.phantom.blood_vessels]}

@app.post("/api/us/vessel/add")
def us_add_vessel(req: AddVesselRequest):
    v = _us.add_vessel(
        req.start_x, req.start_y, req.end_x, req.end_y,
        req.radius_cm, req.velocity_m_s, req.direction_deg, req.name
    )
    return v

@app.post("/api/us/vessel/delete")
def us_delete_vessel(req: DeleteVesselRequest):
    return _us.delete_vessel(req.vessel_id)

@app.post("/api/us/vessel")
def us_vessel(req: VesselRequest):
    kwargs = req.dict(exclude_none=True)
    vid = kwargs.pop("vessel_id", 0)
    # Legacy: support old direct vessel_id update via query
    active = _us.phantom.get_active_vessels()
    vessel = next((v for v in active if v.vessel_id == vid), None)
    if vessel is None:
        raise HTTPException(404, "Vessel not found")
    return _us.update_vessel(vid, **kwargs)

@app.post("/api/us/shape")
def us_shape_params(req: ShapeParamsRequest):
    shape = next((s for s in _us.phantom.shapes if s["shape_id"] == req.shape_id), None)
    if shape is None:
        raise HTTPException(404, "Shape not found")
    _us.phantom.update_shape(req.shape_id, req.params)
    return next(s for s in _us.phantom.shapes if s["shape_id"] == req.shape_id)

@app.post("/api/us/params")
def us_params(req: ParamsRequest):
    d = _us.params.to_dict()
    d.update(req.dict(exclude_none=True))
    _us.update_params(WaveformParams.from_dict(d))
    return _us.get_output()

@app.post("/api/us/reset")
def us_reset():
    _us.reset()
    return {"status": "reset"}


# ── Radar endpoints ───────────────────────────────────────────────────────────

@app.get("/api/radar/state")
def radar_state():
    return _radar.get_output()

@app.post("/api/radar/step")
def radar_step():
    return _radar.step()

@app.post("/api/radar/body/add")
def radar_add_body(req: AddBodyRequest):
    return _radar.add_body(req.distance_m, req.angle_deg, req.size_m)

@app.post("/api/radar/body/move")
def radar_move_body(req: MoveBodyRequest):
    return _radar.move_body(req.body_id, req.distance_m, req.angle_deg)

@app.post("/api/radar/body/resize")
def radar_resize_body(req: ResizeBodyRequest):
    return _radar.resize_body(req.body_id, req.size_m)

@app.post("/api/radar/body/delete")
def radar_delete_body(body_id: int):
    return _radar.delete_body(body_id)

@app.post("/api/radar/scan")
def radar_scan(req: ScanRequest):
    if req.speed is not None:
        _radar.radar.scan_controller.set_scan_speed(req.speed)
    if req.beam_width is not None:
        _radar.radar.scan_controller.set_beam_width(req.beam_width)
    return _radar.step()

@app.get("/api/radar/broad-scan")
def radar_broad_scan():
    return _radar.do_broad_scan()

@app.get("/api/radar/narrow-scan")
def radar_narrow_scan(target_angle: float):
    return _radar.do_narrow_scan(target_angle)

@app.post("/api/radar/params")
def radar_params(req: ParamsRequest):
    d = _radar.params.to_dict()
    d.update(req.dict(exclude_none=True))
    _radar.update_params(WaveformParams.from_dict(d))
    return _radar.get_output()

@app.post("/api/radar/reset")
def radar_reset():
    _radar.reset()
    _radar.initialize()
    return {"status": "reset"}


# ── Static frontend ───────────────────────────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)