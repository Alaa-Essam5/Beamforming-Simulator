<img width="3168" height="500" alt="Image" src="https://github.com/user-attachments/assets/bb677e12-1d5a-423a-a462-2a05e5f2a43b" />



<div align="center">



<h1>BeamForge</h1>
<p><b>Interactive 2D Phased Array Beamforming Simulator</b></p>

<!-- Badges -->
<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
<img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black"/>
<img src="https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white"/>
<img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white"/>

</div>

---




> A full-stack interactive 2D beamforming simulator covering Core Beamforming, 5G Network Simulation, Ultrasound Imaging, and Radar Detection — built with a Python/FastAPI backend and a pure HTML/JS frontend.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation & Running](#installation--running)
- [Core Beamforming](#1-core-beamforming)
- [5G Simulator](#2-5g-simulator)
- [Ultrasound Simulator](#3-ultrasound-simulator)
- [Radar Simulator](#4-radar-simulator)
- [Global Parameters Reference](#global-parameters-reference)
- [Physics & Math Summary](#physics--math-summary)
- [Known Issues & Notes](#known-issues--notes)

---

## Overview

BeamForge is a real-time 2D phased array simulator that demonstrates constructive and destructive interference, beam steering via electronic phase shifts, apodization and windowing for side-lobe control, and SNR effects — applied across three real-world domains: wireless communications, medical imaging, and radar.

The entire system is built around a single shared `BeamformingEngine` and `WaveformParams` dataclass. No physics logic is duplicated across the three applications — they all import and extend the same core.

| Module | Domain | Spatial Scale | Typical Frequency |
|---|---|---|---|
| Core BF | General phased array theory | Abstract / normalized | 0.1 – 10 GHz |
| 5G | Wireless network beamforming | Meters (up to 1.2 km) | 2.4 GHz |
| Ultrasound | Medical pulse-echo imaging | Centimeters / millimeters | 1 – 20 MHz |
| Radar | Target detection and ranging | Meters to kilometers | 1 – 35 GHz (X-band default) |

---

## Project Structure

```
beamforge/
│
├── backend/
│   ├── main.py                  # FastAPI app — all REST endpoints, routing only
│   ├── base_simulator.py        # Abstract base: initialize, step, reset, SNR, export/load state
│   ├── beamforming_engine.py    # Apodizer, NoiseModel, BeamformingEngine (shared by all apps)
│   ├── waveform_params.py       # WaveformParams dataclass (9 params) + SNRConfig
│   ├── oscillator.py            # Single array element: phase shift, time delay, complex weight
│   ├── math_utils.py            # array_factor, interference_map, Doppler, path_loss, geometry
│   ├── fiveg_simulator.py       # Tower, UserEquipment, CoverageMapper, FiveGSimulator
│   ├── ultrasound_simulator.py  # SheppLoganPhantom, BloodVessel, Probe, Amode, Bmode, Doppler
│   └── radar_simulator.py       # SolidBody, BeamSteerer, ScanController, RadarSimulator
│
└── frontend/
    └── index.html               # Single-file UI — all 4 simulators, Chart.js, pure JS
```

### Design Principles

- **`base_simulator.py`** enforces the interface contract. Every simulator must implement `initialize()`, `step()`, `reset()`, `update_params()`, and `get_output()`. Shared functionality like SNR application and state export lives here once.
- **`beamforming_engine.py`** is the physics core. The `BeamformingEngine` class owns the oscillator array, apodizer, and noise model. It is instantiated inside each application simulator — never re-implemented.
- **`math_utils.py`** holds every mathematical formula used anywhere in the system: array factor, steering phases, interference map, Doppler shift, reflection coefficient, path loss. This ensures a single source of truth for all physics.
- **`waveform_params.py`** is the central parameter container. Changing any parameter anywhere flows through this dataclass, validated before use.

---

## Installation & Running

### Requirements

```bash
pip install fastapi uvicorn numpy scipy
```

### Start the backend

```bash
cd backend
python main.py
# FastAPI server starts at http://localhost:8000
# Interactive API docs at http://localhost:8000/docs
```

### Open the frontend

Open `frontend/index.html` directly in any modern browser (Chrome, Firefox, Edge).
The status indicator in the top-right corner turns **green (ONLINE)** when the API responds.

> **Note:** The Radar tab is fully self-contained in JavaScript and works even when the backend is offline. All other tabs require the backend.

---

---

# 1. Core Beamforming

> The mathematical and visual foundation of the entire simulator. This tab isolates pure beamforming theory so you can directly observe how array parameters produce a steerable, controllable beam.

---

## What Is Beamforming?

A phased array antenna consists of multiple individual radiating elements placed at known positions (usually equally spaced along a line — a Uniform Linear Array, ULA). Each element transmits the same signal, but with a carefully chosen time delay or phase shift applied to it.

Because electromagnetic waves superpose linearly, the signals from all elements add together in space. In the direction you want the beam, the signals from every element arrive **in phase** — they add constructively and produce a strong combined signal. In all other directions, the signals arrive at different phases — they partially or fully cancel, producing destructive interference.

This is the core idea: **you control where the wave energy goes by controlling the phase of each element.** No mechanical movement is needed. The beam can be redirected electronically in microseconds.

---

## The Three Visualization Panels

### Panel 1 — Interference / Constructive-Destructive Field Map

This is a 2D color map of the electromagnetic or acoustic pressure field produced by the entire array. Every pixel represents a point in space. Its color shows the total field intensity — the result of all element waves superposing.

- **Bright regions (cyan → white):** constructive interference — waves arrive in phase and amplify each other.
- **Dark regions (black → deep blue):** destructive interference — waves arrive out of phase and cancel.
- **The main beam:** a bright lobe extending in the steering direction — this is where nearly all energy is concentrated.
- **Side lobes:** smaller bright regions at other angles — unavoidable energy leakage outside the main beam.
- **Nulls:** dark bands between lobes — directions where cancellation is nearly perfect.

The map updates live as you change any parameter:
- Increasing N narrows the main beam dramatically.
- Spacing beyond 0.5λ creates additional bright grating lobes at unwanted angles.
- Apodization reduces side lobe brightness while slightly widening the main beam.
- Reducing SNR adds a noise floor — dark destructive regions fill with speckle.

> With N=8 and rectangular window, significant bright side lobes are visible. Switch to Blackman — the side lobes nearly disappear, but the main beam widens slightly. This is the fundamental apodization trade-off.

---

### 📸 Image  — Interference Map (Rectangular Window, N=16, steered 30°)

<img width="1919" height="920" alt="Image" src="https://github.com/user-attachments/assets/de59facc-8c8c-4a15-b35b-15f723db175d" />

---

### 📸 Image  — Interference Map (Blackman Window, same settings)

<img width="1919" height="850" alt="Image" src="https://github.com/user-attachments/assets/6e753562-83ff-4ba3-ad28-75e4fbd37b7f" />
---

### Panel 2 — Beam Profile (dB)

The beam profile plots the normalized array factor AF(θ) against angle in decibels. This is the standard engineering representation of antenna directivity.

- **X-axis:** observation angle from −180° to +180°.
- **Y-axis:** normalized magnitude in dB. 0 dB = peak. Side lobes typically appear at −13 dB for a rectangular window.
- **Main lobe:** the central peak at the steering angle. Its width (3 dB beamwidth) narrows as N increases or spacing increases.
- **Side lobes:** secondary peaks whose level (SLL) is what apodization controls.
- **Grating lobes:** if spacing > 0.5λ, full-strength peaks appear at wrong angles.
- **Two overlaid curves:** the dB profile (cyan) and a scaled normalized magnitude (purple) for reference.

The right-panel beam metrics are computed from this profile:
- **Main Lobe Angle:** the angle at which AF is maximum — matches the set steering angle.
- **3 dB Beamwidth:** angular width between −3 dB points. Beamwidth ≈ 0.886 λ/L where L is the array aperture.
- **Side Lobe Level (dB):** the highest side lobe relative to the main lobe.

---

### 📸 Image  — Beam Profile Chart

<img width="1202" height="799" alt="Image" src="https://github.com/user-attachments/assets/5ca4c4f0-db15-4824-8ece-8d00b35356ab" />

---

### Panel 3 — Array Weights

Visualizes the complex excitation weight applied to each element:
- **Amplitude bars:** height shows per-element amplitude from the apodization window. Center elements are strongest with Hamming/Hanning/Blackman windows — forming a bell-curve shape.
- **Real part (Re{w}):** cosine component of the phase shift.
- **Imaginary part (Im{w}):** sine component.
- **Phase ramp:** when steering angle changes, the real and imaginary bars oscillate like a cosine wave sweeping across elements — this spatial phase ramp is what points the beam.
- **Element strip (top of right panel):** physical positions of all elements along the array axis.

### 📸 Image  — Array Weights

<img width="1919" height="912" alt="Image" src="https://github.com/user-attachments/assets/08e119bd-365e-423e-9da0-30d4f66b306a" />

---

## Parameter Effects — Detailed

### Number of Elements (N): 2 to 64

The most impactful parameter.

- **Increasing N** narrows the main beam (beamwidth ≈ λ/(N·d)) and increases directivity (gain ≈ N²). N=2: very wide beam, almost no directional selectivity. N=64: extremely sharp, narrow beam.
- **Decreasing N** makes the beam wide and the array unable to distinguish directions.
- **Physical meaning:** more elements = larger aperture = finer angular resolution — analogous to a larger telescope lens giving a sharper image.

### Element Spacing (d): 0.1λ to 2.0λ

- **d = 0.5λ:** the standard and optimal spacing. Provides full 180° steering range without grating lobes. Used in almost all practical phased arrays.
- **d < 0.5λ:** the array is physically smaller than optimal, reducing directivity.
- **d > 0.5λ:** grating lobes appear — full-strength secondary main lobes at other angles. In communications: interference. In radar: false targets. In ultrasound: ghost images.
- **Rule:** always keep d ≤ 0.5λ for a clean single-beam pattern.

### Steering Angle (θ): −90° to +90°

- Moves the main lobe direction across the visible hemisphere.
- As |θ| approaches 90° (end-fire), the beam widens significantly because the cosine projection becomes very sensitive near 90°.
- Steering works by applying a linear phase ramp across the elements: element n receives phase φₙ = −2π·d·n·cos(θ).

### Apodization Window

The window function modifies element amplitude weights. Without windowing (rectangular = all weights 1.0), the array factor has a sinc-like pattern with first side lobes at −13.3 dB.

| Window | First Side Lobe | Main Lobe Width | Best Use |
|---|---|---|---|
| Rectangular | −13.3 dB | Narrowest | Maximum resolution, highest side lobes |
| Hanning | −31.5 dB | +46% wider | General purpose |
| Hamming | −42.7 dB | +48% wider | Good balance |
| Blackman | −58.1 dB | +69% wider | Side lobes must be minimized |
| Chebyshev | −50.0 dB | Adjustable | Equiripple — all side lobes equal height |
| Taylor | −35.0 dB | +30% wider | Smooth transition, used in radar |

Windowing works because side lobes are caused by the sharp edges of a rectangular aperture — abruptly switching elements on/off creates spectral leakage. Tapering amplitude smoothly to zero at the edges eliminates this leakage.

### SNR: 0 to 1000

Applied as additive white Gaussian noise (AWGN) to the array factor output.
- **High SNR (100+ dB):** clean beam pattern, deep nulls, sharp side lobes.
- **Low SNR (5 dB):** noise floor rises, nulls fill in, side lobes are obscured, beam profile loses definition.

### Amplitude (A): 0.1 to 5

Scales all element signal levels uniformly. Does not affect the normalized beam shape but affects absolute received power and SNR margin. Higher amplitude = more transmit power = extended range.

### Pulse Width (τ): 0.1 to 10 µs

Sets the transmitted pulse duration. Determines axial resolution: `axial_res = c·τ/2`. Shorter pulse = finer range discrimination but less energy. Most visible in Ultrasound A-mode and Radar range resolution.

### Sampling Rate (fs): 1 to 100 GHz

The ADC rate. Must satisfy Nyquist: fs ≥ 2·f. When fs < 2·f, received signals alias — frequencies fold and appear at incorrect positions. Visible in the Ultrasound element waveform panel as distorted waveforms.

---

## Core Beamforming Math

```
Array Factor (normalized):
  AF(θ) = |Σ wₙ · exp(j·2π·d·n·(cos θ − cos θₛ))| / |AF|_max

  n  = element index (0 to N−1)
  d  = element spacing in wavelengths
  θ  = observation angle
  θₛ = steering angle
  wₙ = apodization_weight × exp(j·φₙ_steering)

Steering phase for element n:
  φₙ = −2π · d · n · cos(θₛ)

3 dB Beamwidth estimate:
  BW₃dB ≈ 0.886 · λ / L = 0.886 / (N · d)   [radians]
         ≈ 50.7° / (N · d)                    [degrees]

Interference Map field:
  E(x,y) = Σ wₙ · exp(j·(k·rₙ − φₙ)) / rₙ
  k = 2π/λ   rₙ = √((x−xₙ)² + y²)
  Intensity = |E(x,y)|²
```

---

---

# 2. 5G Simulator

> Simulates a multi-tower, multi-user 5G network where each base station uses the shared beamforming engine to automatically track its connected users. Demonstrates MU-MIMO, path-loss-based handover, and signal coverage mapping.

---

## What Is 5G Beamforming?

In 5G NR (New Radio), the base station (called a gNB — next-generation NodeB) uses a phased array antenna to form narrow beams directed precisely at each user device (UE — User Equipment). This is called **massive MIMO beamforming**.

Key advantages over omnidirectional antennas:
- **Higher gain:** concentrating energy toward the user increases received signal strength without increasing transmit power.
- **Spatial multiplexing:** multiple users served simultaneously using beams in different directions without interfering — this is **MU-MIMO (Multi-User MIMO)**.
- **Reduced interference:** narrow beams mean less energy radiating toward non-intended receivers.
- **Automatic tracking:** as the user moves, the beam angle updates electronically in real time.

---

## Network Architecture

```
Simulation area: 1200 m × 1200 m

gNB-Alpha  position: (−200 m, +150 m)   color: cyan
gNB-Beta   position: (+200 m, +150 m)   color: purple
gNB-Gamma  position: (  0 m,  −200 m)   color: green

UE-0 "Mobile"  → move with W / A / S / D keys   (30 m per press)
UE-1 "IoT"     → move with Arrow keys            (30 m per press)

Influence radius: 500 m per tower
Boundary: ±600 m on both axes
```

---

## The Two View Panels

### Panel 1 — Network Map

The main canvas shows the live state of the entire network:

- **Tower nodes (T0, T1, T2):** circles with a glow ring. A steering arrow points in the current beam direction. Sub-labels show tower name, steering angle, number of elements, frequency, and apodization window — all updating live.
- **Influence radius rings:** dashed 500 m circles. A user inside is eligible for connection.
- **Beam footprint polygons:** the actual beam shape computed from the real array factor, shown as a semi-transparent filled polygon pointing toward the connected user — not a simple geometric triangle.
- **Connection lines:** gradient dashed lines from tower to user, colored from tower color to user color. Annotated with link distance (m) and received SNR in dB (green > 15 dB, amber > 5 dB, red < 5 dB).
- **User nodes (diamonds):** yellow for UE-0, red for UE-1. Labeled with signal strength in dBm.

### Panel 2 — Beam Profiles

Shows normalized beam magnitude vs. angle for every active tower-user link. If a tower serves two users, a single MU-MIMO composite profile is shown — you can see two lobes pointing in two directions simultaneously.


---

## How Automatic Beam Steering Works

Every time a user moves, `_update_network()` runs through four steps:

1. **Link Attachment:** For each user, compute distance to every tower. Apply free-space path loss. Connect user to the tower with the highest received signal.
2. **MU-MIMO Profile:** For each tower with connected users, compute the angle to each user. Sum the individual steering vectors into a composite MU-MIMO weight vector.
3. **Steering Direction:** Set the tower's steering angle to the circular mean of all connected user angles.
4. **Per-user metrics:** Interpolate directivity gain from the beam profile at each user's angle. Recompute received signal and SNR including the beam gain.

```
Tower beam gain toward user (dB):
  dir_gain_dB = interpolate(beam_profile.magnitude_db, user_angle)

Received signal (dBm):
  sig_dBm = 40 dBm (TX power) + dir_gain_dB − path_loss_dB

Received SNR (dB):
  snr_dB = sig_dBm + 90 dB  (assumes −90 dBm noise floor)
```

---

## MU-MIMO Composite Beam

When two users at angles θ₁ and θ₂ are both served by the same tower, a composite beam is formed:

```
combined_weight[n] = exp(−j·2π·d·n·cos(θ₁)) + exp(−j·2π·d·n·cos(θ₂))
→ normalized, then multiplied by apodization weights
→ results in a beam with two lobes, one toward θ₁, one toward θ₂
```

This is a simplified but physically correct demonstration of simultaneous multi-direction beamforming.

---

## Path Loss Model

```
Free-space path loss (Friis):
  PL(dB) = 20·log₁₀(4π·d·f/c)

  d = distance in meters   f = frequency in Hz   c = 3×10⁸ m/s

At 2.4 GHz:
  100 m  →  PL ≈ 60 dB   →  sig ≈ −20 dBm   (excellent)
  300 m  →  PL ≈ 70 dB   →  sig ≈ −30 dBm   (good)
  500 m  →  PL ≈ 76 dB   →  sig ≈ −36 dBm   (usable edge of coverage)
```

---

### Network Map with Active Connections


<img width="1203" height="796" alt="Image" src="https://github.com/user-attachments/assets/da795a72-a4bf-4224-a147-e78bd9f83fe6" />

---


### 📹 Video  — User Movement and Automatic Handover

https://github.com/user-attachments/assets/7b08a4e1-61b6-40b9-91a7-1754b2797f3f

---

### 📹 Video  — MU-MIMO Dual-Lobe Beam

https://github.com/user-attachments/assets/76b70d01-bf42-450a-be9b-8b6422e81f37

---

# 3. Ultrasound Simulator

> Simulates A-mode, B-mode, and Doppler ultrasound scanning on a Shepp-Logan brain phantom with realistic acoustic tissue properties. All three modes display simultaneously and update live as the probe moves.

---

## What Is Ultrasound Imaging?

Medical ultrasound works on the **pulse-echo principle**: a piezoelectric transducer (the probe) transmits a short burst of high-frequency sound into tissue. At each boundary between two materials with different acoustic impedance, part of the sound reflects back as an echo and part continues deeper. The probe receives the returning echoes and plots their amplitude vs. arrival time — since sound speed in tissue is approximately constant (1540 m/s), arrival time maps directly to depth.

The phased array probe contains many small piezoelectric elements. By applying different time delays to each element, the beam can be steered and focused to any point in tissue — electronically, without moving the probe.

### 📹 Ultrasound and moveing of the prob and steering 


https://github.com/user-attachments/assets/5d5e34dd-028d-4b48-860d-34b9451b58a3


**Acoustic impedance Z = ρ·c** (density × sound speed) is the key tissue property. At any boundary:

```
Reflection coefficient: RC = (Z₂ − Z₁) / (Z₁ + Z₂)
Transmission coefficient: TC = 2·Z₂ / (Z₁ + Z₂)
```

A large RC (e.g. at tissue/bone) = strong echo. A small RC (e.g. between similar soft tissues) = weak echo.

---

## The Shepp-Logan Phantom

The Shepp-Logan phantom is a standard test image in medical imaging research representing a simplified human head cross-section. Every shape has realistic acoustic tissue properties:

| ID | Name | Tissue | Z (MRayl) | Att (dB/cm/MHz) | Speed (m/s) | Echo | Clinical Appearance |
|---|---|---|---|---|---|---|---|
| 0 | Skull | Bone | 7.8 | 10.0 | 3000 | 0.95 | Very bright, strong shadow behind |
| 1 | Brain | White matter | 1.58 | 0.30 | 1560 | 0.20 | Low-level gray echoes |
| 2, 3 | CSF | Fluid | 1.52 | 0.002 | 1515 | 0.01 | Near-anechoic (dark) |
| 4 | Ventricle | CSF | 1.52 | 0.002 | 1515 | 0.01 | Anechoic cavity |
| 5 | Pineal Gland | Gray matter | 1.60 | 0.35 | 1550 | 0.28 | Slightly echogenic |
| 6 | Tumor (Main) | Tumor | 1.72 | 0.90 | 1590 | 0.75 | Bright, well-defined |
| 7 | Tumor (Small) | Tumor | 1.72 | 0.90 | 1590 | 0.75 | Bright spot |
| 8 | Cyst | Fluid | 1.50 | 0.005 | 1510 | 0.00 | Completely anechoic (black) |
| 9 | Calcification | Calcified | 7.5 | 12.0 | 3000 | 0.98 | Hyperechoic with acoustic shadow |
| 10 | Lesion | Gray matter | 1.60 | 0.35 | 1550 | 0.28 | Subtle echogenic lesion |

**Hover** over any shape to see its tissue properties in a tooltip. **Click** any shape to open the editor and change impedance, attenuation, sound speed, or echogenicity in real time — A-mode and B-mode update immediately.


<img width="1915" height="926" alt="Image" src="https://github.com/user-attachments/assets/059b7955-9021-4d40-9908-0ae0056e31ad" />

---

## The Five Display Panels

### Panel 1 — Phantom Canvas

Shows the phantom in teal-white echogenicity scale. Overlaid:

- **Yellow probe body:** sits on the top edge (y = −9.5 cm). Drag horizontally with mouse, or use ← → keys.
- **Dashed beam ray:** extends from the probe through the phantom in the current beam direction.
- **Beam cone:** shows near-field (converging, pre-focal) and far-field (diverging, post-focal) zones.
- **Cyan focal point marker (★):** the depth of narrowest beam and best lateral resolution. Labeled "F = x cm." Adjust with ↑ ↓ keys.
- **Blood vessels:** colored tubes with flow direction arrows.
- **Doppler angle θ label:** the angle between beam and vessel flow, color-coded: green (< 45°), amber (45–75°), red (> 75°).

### 📸 Phantom with Probe, Beam, and Focal Point

- <img width="257" height="384" alt="Image" src="https://github.com/user-attachments/assets/4a48dff9-b32c-4998-9f46-29f72280142c" />

### Panel 2 — A-Mode (Amplitude Mode)

A-mode is the simplest ultrasound output: echo amplitude vs. depth from a single scan line.

- **Green curve (RF line):** raw received echo after demodulation. Sharp spikes at tissue boundaries (impedance mismatches). Spike height depends on reflection coefficient and accumulated attenuation.
- **Amber curve (Envelope):** smoothed envelope of the RF signal — this is what forms brightness values in B-mode.
- **Dashed cyan line:** marks the focal depth. Echoes near this depth are strongest — the beam is most focused there.
- **Attenuation decay:** deep echoes are weaker due to frequency-dependent absorption: `att = α·f·z dB`.
- **TGC compensation:** Time Gain Compensation applies increasing gain with depth to partially restore lost amplitude, keeping deep structures visible alongside shallow ones.

Changes visibly as you:
- Move the probe to different lateral positions (different tissue cross-sections).
- Tilt the beam angle (different path through the phantom).
- Change frequency (higher f → sharper spikes, faster attenuation).
- Change pulse width (longer τ → merged spikes, lower axial resolution).
- Change SNR (lower → visible noise floor baseline).

### 📸  A-Mode Echo Trace

<img width="1280" height="390" alt="Image" src="https://github.com/user-attachments/assets/b318b933-e060-4eda-804b-6c1e53891f70" />

### Panel 3 — Array Element Waveforms

Shows individual transmit waveforms of up to 8 selected elements. Educational features:

- **Phase ramp:** adjacent elements have progressively shifted phases. Waveforms shift left/right relative to each other — visualizing the steering wavefront tilt.
- **Focal phase offsets:** edge elements fire slightly earlier than center elements, creating a converging wavefront for focusing.
- **Grating lobe warning:** red background when spacing > 0.5λ. Ghost oscillations superimposed representing grating lobe contribution.
- **Nyquist warning:** if fs < 2f, the displayed waveform frequency is incorrect (aliased) and a yellow warning appears.
- **Yellow wavefront line:** connects the peak of each element's waveform — its slope visually confirms the steering direction.
- **Status bar:** all 10 active parameters in a single line at the bottom.

<img width="1500" height="171" alt="Image" src="https://github.com/user-attachments/assets/c7a0ad48-d458-47b4-bf36-ec5d86b5991a" />


### Panel 4 — B-Mode (Brightness Mode)

B-mode assembles many A-mode lines at different lateral positions into a 2D image.

- **Full Sweep button:** runs 128 A-mode lines across x = −9 to +9 cm, assembling a full image (1–2 seconds).
- **Live mode:** continuously runs 64 A-mode lines for real-time preview.
- **Log compression:** `log₁₀(1 + 50·v) / log₁₀(51)` — compresses dynamic range so both weak and strong echoes are simultaneously visible.
- **Lateral blur:** at depths far from the focal zone, a Gaussian blur is applied to B-mode rows proportional to defocusing distance — simulating the physical widening of the beam.
- **Focal zone line:** dashed cyan horizontal line at focal depth.
- **Probe position line:** dashed amber vertical line at current probe x-position.

### 📸 B-Mode Sonogram (Full Sweep)


<img width="772" height="303" alt="Image" src="https://github.com/user-attachments/assets/587363fe-2361-4a1c-9f15-30b4b8670560" />


### Panel 5 — Doppler Spectrum

Shows the power spectrum of the received signal from the selected blood vessel:

```
Doppler equation:
  Δf = 2 · f₀ · v · cos(θ) / c

  f₀ = transmit frequency (Hz)
  v  = blood flow velocity (m/s)
  θ  = angle between beam direction and flow direction
  c  = sound speed in tissue (1540 m/s)
```

**The cos(θ) factor is critical:**

| θ | cos(θ) | Δf fraction | What you see |
|---|---|---|---|
| 0° (parallel to flow) | 1.00 | 100% | Maximum Doppler shift, full velocity |
| 30° | 0.866 | 86.6% | Slight underestimate |
| 45° | 0.707 | 70.7% | Velocity underestimated by ~30% |
| 60° | 0.500 | 50.0% | Velocity underestimated by 50% |
| 75° | 0.259 | 25.9% | Almost no detectable shift |
| 90° (perpendicular) | 0.000 | 0% | **Δf = 0 always — flat spectrum, no peak** |

**Sign convention:**
- Flow **toward** the probe → **positive Δf** → peak on the right side.
- Flow **away** from the probe → **negative Δf** → peak on the left side.
- Perpendicular flow (θ = 90°) → **flat zero spectrum**, no peak, regardless of velocity magnitude.

The Doppler spectrum is also affected by SNR — low SNR raises the noise floor and can obscure a weak Doppler peak from a slow or distant vessel.

### 📸 Doppler Spectrum with Active Signal

<img width="771" height="290" alt="Image" src="https://github.com/user-attachments/assets/80f426f2-350c-4c36-9be4-46075ddbe46f" />

---

## The 10 Ultrasound Parameters — Detailed Effects

### 1. Frequency (1–20 MHz, default 5 MHz)

The fundamental resolution-vs-penetration trade-off:
- **Higher f (e.g. 15 MHz):** shorter wavelength → finer axial resolution (sharper spikes), better lateral resolution. But attenuation is proportional to frequency (dB/cm/MHz) — signal dies quickly. Used for shallow structures (thyroid, breast, superficial muscles, ~1–5 cm depth).
- **Lower f (e.g. 2 MHz):** longer wavelength → poorer resolution but deeper penetration. Used for abdominal imaging (liver, kidneys, 10–20 cm depth).
- **Effect in simulator:** A-mode spike sharpness changes, Doppler Δf shifts (Δf ∝ f), element waveform frequency changes, grating lobe threshold moves.

### 2. Pulse Width (0.05–2 µs, default 0.5 µs)

Determines axial resolution directly:
```
axial_res = c · τ / 2 = 1540 × 0.5×10⁻⁶ / 2 ≈ 0.39 mm (at default)
```
- **Shorter pulse (0.05 µs):** axial_res ≈ 0.04 mm — skull inner and outer walls appear as two clearly separate spikes.
- **Longer pulse (2 µs):** axial_res ≈ 1.54 mm — closely spaced boundaries merge into a single wide hump, losing spatial detail.
- Trade-off: shorter pulse = less energy transmitted = worse SNR at depth.

### 3. Number of Elements (4–128, default 64)

- Determines aperture = N × d. Larger aperture → narrower beam → better lateral resolution at the focal depth.
- N=4: very wide beam, structures blurry and overlapping horizontally in B-mode.
- N=128: very narrow beam, fine lateral detail. The element waveform panel shows more waveforms with finer phase gradations.

### 4. Amplitude (0.1–5 V, default 1.0 V)

- Transmit power P ∝ A². Higher amplitude → stronger echoes → better SNR at depth.
- Visible in A-mode as uniform scaling of all echo amplitudes. Deep echoes that were at the noise floor become detectable.

### 5. Element Spacing (0.2–1.5λ, default 0.5λ)

At 5 MHz: λ = 1540 / 5×10⁶ = 0.308 mm. Half-wavelength spacing = 0.154 mm.
- **d > 0.5λ:** grating lobes create ghost images in B-mode — structures appear at incorrect lateral positions. The element waveform panel shows red warning and ghost oscillations.
- **d = 0.5λ:** clean beam, no artifacts.
- **d < 0.5λ:** array physically compact, reduced directivity.

### 6. Sampling Rate (10–200 MHz, default 40 MHz)

- Must be ≥ 2f (Nyquist) to correctly digitize received echoes.
- At 5 MHz: minimum fs = 10 MHz. Default 40 MHz = 8× oversampling.
- Below Nyquist: Doppler spectrum aliases, apparent peak shifts to incorrect frequency. Element waveforms show aliased distortion. Yellow warning appears in waveform panel.

### 7. SNR (0–100 dB, default 30 dB)

Applied to all three outputs:
- **A-mode:** AWGN added to RF signal. Deep weak echoes become indistinguishable from noise at low SNR.
- **Doppler:** noise added to spectrum. Weak peaks (low velocity or poor angle) disappear into noise baseline.
- **B-mode:** noise propagated through envelope computation. Deep image regions become grainy.

### 8. TGC — Time Gain Compensation (0–10 dB/cm, default 2 dB/cm)

```
gain(z) = 10^(TGC_slope × z / 20)   [exponential growth with depth]
```
- **TGC = 0:** no compensation. Shallow structures (skull) are bright; deep structures (brain interior) are barely visible. The A-mode trace drops steadily to near-zero by 10 cm.
- **TGC = 2 dB/cm:** adds +2 dB per cm of depth. Structures at 10 cm are amplified by +20 dB — restoring their visibility.
- **TGC = 8 dB/cm:** heavy compensation. Deep regions appear nearly as bright as shallow ones but noise at depth is also amplified, producing a grainy deep image.

### 9. Focal Depth (1–15 cm, default 5 cm)

```
lateral_res ≈ λ · F / D   (F = focal depth, D = aperture = N·d)
```
- Before the focal zone: beam converges → lateral resolution improves with depth.
- At focal depth: minimum beam width → best lateral resolution.
- Beyond focal zone: beam diverges → lateral resolution degrades.
- In B-mode: structures at the focal depth appear sharpest. Structures far from focal depth show lateral blurring.
- Keyboard: ↑ decreases focal depth, ↓ increases it.

### 10. Sound Speed (1400–1700 m/s, default 1540 m/s)

```
depth = c · t / 2   (t = echo round-trip time)
```
- Changing c shifts the apparent depth of all structures proportionally.
- **c = 1700 m/s:** same echo arrival time → structures appear deeper than they are (17.5 cm instead of 15 cm).
- **c = 1400 m/s:** structures appear shallower. Measurements become inaccurate.
- Real tissue speeds: fat ~1450 m/s, soft tissue ~1540 m/s, bone ~3000 m/s. The simulator uses 1540 m/s as the soft tissue average.

---


### 📹 Live Probe Movement and A-Mode , B-Mode and doppler Updates

https://github.com/user-attachments/assets/c9b74c9d-ac81-41ab-b5da-cc8c2de00907


---


---

# 4. Radar Simulator

> A self-contained 360° electronic-steering radar with broad/narrow scan automation, SNR-dependent detection, radar range equation, PRF ambiguity, rain attenuation, and clutter modeling. Runs entirely in the browser — no backend required.

---

## What Is Phased Array Radar?

Traditional radar rotates a directional antenna mechanically — one full rotation per second for a typical air traffic control radar. This is slow, mechanically complex, and cannot dwell on targets for more than a fraction of a second per rotation.

A **phased array radar** replaces the rotating antenna with a stationary array whose beam is steered electronically via phase shifts:

1. **Transmit:** the array fires a short pulse in the current beam direction with precise phase shifts concentrating energy in one direction.
2. **Wait:** the pulse travels outward at the speed of light (3×10⁸ m/s) and returns as an echo if it hits a target.
3. **Receive:** the array collects the returning echo. The same phase shifts are reversed to coherently combine received signals — providing receive beamforming gain.
4. **Advance angle:** the steering angle increments and the process repeats.

**Key advantages over mechanical rotation:**
- Can jump to any angle in microseconds (vs. seconds for mechanical).
- Can dwell on a target of interest without stopping the scan.
- No mechanical wear, much higher reliability.
- Electronic steering is what makes this simulator's beam "rotate" — not any physical movement.

### Radar Simulator
<img width="1916" height="914" alt="Image" src="https://github.com/user-attachments/assets/f6954e00-7d54-4e57-88b1-beb6981f6164" />

---

## Two-Phase Operation

The simulator implements a classic two-stage radar strategy, switching automatically based on what it detects:

### Phase 1 — SCANNING (Broad Beam, Fast Sweep)

- The beam sweeps continuously around 360° at the user-set scan speed.
- Beam width is wide (default 20°) — fast coverage but poor angular resolution.
- Any target detected within the beam is logged, and its approximate angle is added to a **focus queue**.
- Purpose: rapid detection. Does not attempt to measure size accurately.
- **Analogy:** a security guard sweeping a flashlight around a dark room.

### Phase 2 — FOCUSING (Narrow Beam, Dwell Mode)

- Triggered when the broad scan passes a cluster flagged in the focus queue.
- Beam narrows to **3° width** and locks onto the target angle.
- The system dwells for **1.5 seconds**, accumulating echo power.
- The target's angular extent is measured (angles where echo power falls to 50% of peak) to estimate physical size.
- After measurement, the target enters the **Confirmed Targets** list with angle, distance, and measured size.
- **Analogy:** the guard stops and shines the flashlight directly at the suspicious object to examine it in detail.

**Wide vs. narrow beam trade-off:**

| | Wide Beam (20°) | Narrow Beam (3°) |
|---|---|---|
| Coverage speed | Fast — full 360° quickly | Slow — dwells 1.5s per target |
| Angular resolution | Poor — 20° uncertainty | Good — 3° uncertainty |
| Size estimation | Cannot resolve size | Can estimate physical size |
| Energy per direction | Spread over 20° | Concentrated in 3° |
| Use case | Presence detection | Target characterization |

---

## PPI Display (Plan Position Indicator)

The circular scope is the classic radar display format:

- **Concentric rings:** range markers at 25%, 50%, 75%, 100% of max range, labeled in meters.
- **Spokes:** faint radial lines at every 30° for azimuth reference.
- **Beam wedge:** current beam position as a filled wedge. **Cyan = SCANNING, orange = FOCUSING.**
- **Leading edge glow:** brighter glow on the forward edge of the beam — indicating wave propagation direction.
- **Target circles:**
  - Bright cyan + detection ring = currently detected in beam.
  - Dim blue = exists but not in beam or below detection threshold.
  - Amber dashed ring = range-ambiguous (beyond PRF unambiguous range).
  - "OOR" label = out of range (beyond max range setting).
- **PRF limit ring:** dashed amber circle at the PRF unambiguous range.
- **Range resolution arc:** small arc at beam tip showing the range resolution cell size.
- **"LOCKED" label:** appears at beam tip in orange during FOCUSING.

### Interaction

- **Drag:** click and drag any target to move it.
- **Scroll wheel:** hover over a target and scroll to resize it (size 1–20 units).
- **Right-click:** delete a target instantly.
- **Click empty space:** add a new target at that position (max 5 total).
- **Tooltip:** hover over a target to see angle, distance, size, signal strength, and ambiguity warnings.

---

## Signal History Graph

8-second rolling power trace:
- **Y-axis:** signal power in dB (−100 to +20 dB).
- **White dashed line:** noise floor (−80 dB).
- **Red dashed line:** detection threshold — targets must exceed this line to be detected. Rises as SNR decreases (system becomes less sensitive).
- **Cyan trace:** power during SCANNING. Orange trace: power during FOCUSING (typically higher).
- **Target labels (T1, T2, ...):** appear above signal peaks when the target is detected.

---

## 7 Radar Parameters — Detailed Effects

### Beam Width (2°–90°, default 20°)

Controls the fundamental scan vs. resolution trade-off:
- **Wide beam (30°+):** sweeps 360° very quickly. Detects presence of targets but cannot tell if two targets 10° apart are one or two (angular resolution = beam width).
- **Narrow beam (3°–5°):** fine angular discrimination. Can resolve two targets separated by just a few degrees. But takes much longer to sweep 360° (or must dwell on specific angles).
- **Effect on metrics:** beamwidth directly = angular resolution = HPBW in the status panel. Cross-range resolution at 2000 m: 20° beam → 706 m; 5° beam → 175 m; 3° beam → 105 m.

### Scan Speed (5°–720°/s, default 60°/s)

- **Higher speed:** faster 360° coverage. But the beam spends less time at each angle — fewer pulses hit each target, reducing the signal accumulation and detection probability.
- **Lower speed:** more pulses per angle = better SNR per position = higher detection probability. But takes longer to scan the full space.
- **Automatic coupling:** in SCANNING phase, the effective speed scales with beam width (wide beam scans faster by design).

### Receiver SNR (0–40 dB, default 20 dB)

Controls receiver sensitivity:
- **High SNR (40 dB):** the detection threshold drops to near zero — even tiny distant targets are detected. Signal history graph shows detections far above threshold.
- **Low SNR (5 dB):** the threshold rises significantly — only large nearby targets with strong echoes are detected. Many targets are missed. This is the most important parameter for testing detection limits.

### TX Power Boost (−20 to +30 dB, default 0 dB)

Adds directly to signal strength for all targets:
- **+10 dB:** doubles the detection range for any given target (since Pr ∝ R⁻⁴, +6 dB is needed to double range, +10 dB extends range by ×1.78).
- **−10 dB:** reduces effective range. Far targets disappear from detections.
- Combined with SNR, determines the total effective SNR for the range equation.

### Frequency (1–35 GHz, default 9.5 GHz X-band)

```
λ = c / f = 0.3 / f_GHz   (wavelength in metres)

At 9.5 GHz:  λ = 31.6 mm
At 3 GHz:    λ = 100 mm
At 24 GHz:   λ = 12.5 mm
```
- **Higher frequency:** shorter wavelength → better angular resolution for the same aperture. Also more rain attenuation. X-band (8–12 GHz) is standard for air traffic radar. Ka-band (26–40 GHz) is used for precision automotive radar.
- **Lower frequency:** less atmospheric attenuation, longer range. L-band (1–2 GHz) is used for long-range surveillance.
- Affects: RCS calculation via λ in the range equation, range resolution independent of frequency (ΔR = c·τ/2), beam metrics (EIRP, directivity).

### Pulse Width τ (0.1–10 µs, default 1.0 µs)

```
Range resolution: ΔR = c · τ / 2

τ = 0.1 µs → ΔR = 15 m    (fine resolution, −10 dB vs 1 µs)
τ = 1.0 µs → ΔR = 150 m   (default)
τ = 10 µs  → ΔR = 1500 m  (coarse, +10 dB SNR boost)
```
- **Short pulse:** fine range discrimination — two targets 50 m apart can be resolved. But less energy per pulse = lower SNR = shorter detection range.
- **Long pulse:** poor range discrimination — targets 1 km apart might appear as one. But more energy = better SNR = longer detection range.
- Visible in the status panel as "Range Res" metric updating live.

### PRF (100–5000 Hz, default 1000 Hz)

```
Max unambiguous range: R_max = c / (2 · PRF)

PRF = 100 Hz   → R_max = 1,500 km  (long-range surveillance)
PRF = 1000 Hz  → R_max = 150 km
PRF = 5000 Hz  → R_max = 30 km     (short-range precision)
```
- **High PRF:** fast updates, frequent pulses, good Doppler resolution. But targets beyond R_max appear at a folded (wrong) range — shown as "AMB" with dashed amber ring in the display.
- **Low PRF:** long unambiguous range, no range folding for distant targets.
- The PRF limit ring on the PPI shows the boundary beyond which targets become range-ambiguous.

### Clutter Level (−60 to 0 dBc, default −40 dBc)

Simulates ground/sea return that raises the noise floor:
- **−60 dBc (calm):** very little background return. Even small distant targets are detectable.
- **−40 dBc (default):** moderate clutter. Some sensitivity loss.
- **0 dBc (heavy):** clutter completely dominates. Only the largest, closest targets survive. Real radars use Moving Target Indicator (MTI) filtering to suppress stationary clutter — this simulator shows what happens without it.

### Rain Attenuation (0–10 dB/km, default 0)

```
Two-way loss: Loss(dB) = 2 × atten_rate × range(km)

Light drizzle: ~0.01 dB/km  (negligible at 100 km)
Moderate rain: ~0.5 dB/km   (−100 dB at 100 km — significant)
Heavy storm:   ~5 dB/km     (−1000 dB at 100 km — obliterated)
```
- Increasing rain attenuation makes distant targets fade and disappear — visible as their signal labels dropping in the signal history graph and their detection rings disappearing on the PPI.
- Rain attenuation displayed on detected target tooltips and echo labels.

---

## Radar Physics — Core Equations

```
Radar Range Equation:
  Pr = (Pt · G² · λ² · σ) / ((4π)³ · R⁴)

  Pt = transmit power
  G  = antenna gain ≈ N (array elements)
  λ  = c / f (wavelength)
  σ  = π · (size/2)² (RCS — radar cross section)
  R  = target range

Note: R⁴ dependence means doubling range reduces received power by 12 dB.

Effective SNR:
  SNR_eff = SNR_base + TX_boost + 10·log₁₀(τ_µs)
          − clutter_penalty − rain_loss(R)

Detection threshold:
  threshold = 1 / (1 + SNR_linear × 0.1)
  → SNR 60 dB: threshold ≈ 0.01  (very sensitive)
  → SNR 20 dB: threshold ≈ 0.09
  → SNR  0 dB: threshold ≈ 0.40  (many missed detections)

Cross-range resolution at range R:
  CR_res = 2 · R · tan(BW_deg/2)
```

---

### 📹 Video  — Full Scan → Detect → Focus → Confirm Cycle also  Drag, Resize, Delete, and Physics Parameter Effects

https://github.com/user-attachments/assets/6db96e85-d046-422c-ba0b-cd38aa01690b


---

# Global Parameters Reference

| Parameter | Key | Range | Default | Simulators |
|---|---|---|---|---|
| Number of Elements | `num_elements` | 2–64 | 8 | All |
| Element Spacing | `element_spacing` | 0.1–2.0 λ | 0.5 | All |
| Frequency | `frequency_hz` | > 0 Hz | 2.4 GHz | All |
| Steering Angle | `steering_angle_deg` | −90° to +90° | 0° | All |
| Amplitude | `amplitude` | 0–10 | 1.0 | All |
| Sampling Rate | `sampling_rate` | — | 10 GHz | All |
| Pulse Width | `pulse_width` | — | 1 µs | All |
| Apodization Window | `apodization_window` | 6 types | rectangular | All |
| SNR | `snr_db` | 0–1000 | 30 dB | All |

---

# Physics & Math Summary

### Array Factor

```
AF(θ) = |Σ wₙ · exp(j·2π·d·n·(cos θ − cos θₛ))| / max
wₙ = apodization_weight × exp(j·φₙ_steering)
φₙ = −2π · d · n · cos(θₛ)
```

### Interference Map

```
E(x,y) = Σ wₙ · exp(j·(k·rₙ − φₙ)) / rₙ
k = 2π/λ   rₙ = √((x−xₙ)² + y²)   Intensity = |E(x,y)|²
```

### 5G Path Loss

```
PL(dB) = 20·log₁₀(4π·d·f/c)
```

### Acoustic Reflection

```
RC = (Z₂ − Z₁)/(Z₁ + Z₂)    TC = 2·Z₂/(Z₁ + Z₂)    Z = ρ·c
```

### Acoustic Attenuation

```
A(z) = A₀ · 10^(−α·f·z / 20)    [α in dB/cm/MHz, f in MHz, z in cm]
```

### Doppler Shift

```
Δf = 2·f₀·v·cos(θ) / c    [θ=90° always → Δf=0]
```

### Radar Range Equation

```
Pr = (Pt·G²·λ²·σ) / ((4π)³·R⁴)    σ = π·(size/2)²
```

### Range Resolution

```
ΔR = c·τ/2    (both ultrasound and radar)
```

### PRF Unambiguous Range

```
R_max = c / (2·PRF)
```

---

# Known Issues & Notes

1. **SNR inconsistency across modules:** The global SNR slider maps directly as dB in Ultrasound and 5G. The Radar module maps its own SNR slider (0–40 dB) separately. The 5G received SNR is geometry-based only and does not respond to the global SNR slider.

2. **5G body limit not enforced in API:** `MAX_BODIES = 5` is defined on the `SolidBody` class but `RadarSimulator.add_body()` does not check the active body count. More than 5 bodies can be added via direct API calls.

3. **Radar is frontend-only:** The Radar tab runs entirely in JavaScript and does not call the Python backend. The Python `radar_simulator.py` with a real ULA array factor computation exists but is not wired to the frontend display — the frontend draws a geometric wedge, not a true phased array beam pattern.

4. **Keyboard routing:** WASD and arrow keys are intercepted only when the correct tab is active. In Ultrasound: A/D = beam angle (±2° per press), ← → = probe position (±0.3 cm per press), ↑ ↓ = focal depth (±0.5 cm per press). In 5G: WASD = UE-0, arrows = UE-1 (±30 m per press).

5. **B-mode Full Sweep speed:** 128 A-mode computations server-side takes 1–2 seconds due to ray-marching through the phantom for each scan line. The live B-mode uses 64 columns for real-time response (~200 ms per update).

6. **Steering angle validation:** `WaveformParams.validate()` limits steering to ±90° for the core BF and 5G simulators. The radar simulator uses a separate full 360° steering mechanism via `BeamSteerer.steer_to_360()` that is not constrained by `WaveformParams`.

---



## 👥 Team

| Name | GitHub |
|---|---|
| Abdullah Gamil | [@AbdullahGamil05](https://github.com/AbdullahGamil05) |
| Abdulrahman Hassan | [@AbdulrahmanHassan](https://github.com/abdulrahman-hassan-74) |
| Saga Sadek | [@SagaSadek](https://github.com/saga913) |
| Alaa Essam | [@AlaaEssam](https://github.com/Alaa-Essam5) |


---

*BeamForge — where phase shifts meet interference, and interference meets insight.*
