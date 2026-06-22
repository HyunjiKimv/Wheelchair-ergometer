[![DOI](https://zenodo.org/badge/1274117215.svg)](https://doi.org/10.5281/zenodo.20792196)

# Wheelchair Ergometer Control Software

Open-source Python implementation of the ergometer measurement and control platform described in:

> Kim H. *et al.* "A Detachable Measurement and Control Platform for Wheelchair Ergometers." *Journal of NeuroEngineering and Rehabilitation* (2026).

Replaces the original MATLAB implementation. No additional toolboxes are required.

---

## Hardware

| Component | Model |
|---|---|
| DAQ | LabJack U3-HV |
| DAC (brake control) | LJTick-DAC (FIO4/FIO5, address 0x12) |
| Torque sensor | RSCR RI-50W (500 kgf·cm / 10 V) |
| RPM encoder | 1000 RPM / 10 V analog output |
| Brake | PORA PRB-5Y4 powder brake |

Wiring: AIN0 = torque L, AIN1 = RPM L, AIN2 = torque R, AIN3 = RPM R.

---

## Installation

```bash
pip install -r requirements.txt
```

> **No hardware?** The software runs in simulation mode automatically when `labjackpython` is unavailable or when `--mode demo` is used.

---

## Quick start

```bash
# Full Wingate protocol with hardware
python run_ergometer.py --mode wingate

# Graded Exercise Test
python run_ergometer.py --mode gxt

# Offline demo (no LabJack required)
python run_ergometer.py --mode demo
```

---

## Configuration

Edit `config.py` before each session:

```python
PARTICIPANT = {
    "id":               "P01",
    "total_force":      700,     # N  (person + wheelchair on scale)
    "wheelchair_force": 100,     # N  (wheelchair alone)
    "target_velocity":  1.0,     # m/s  (target rim speed for Wingate)
    "sport":            "basketball",
}
```

Calibration constants (`CALIBRATION`) and hardware wiring (`DAQ`) should only be changed if a sensor is replaced.

---

## Protocol overview

| Stage | Description |
|---|---|
| Warmup | 3 min sub-maximal propulsion |
| Isometric | 3 × 5 s maximal push against locked wheel |
| Sprint | 2 × 10 s maximal sprint for Fiso-based load prescription |
| Wingate | 30 s all-out test at prescribed resistance |
| GXT | Graded 15-step incremental test |

Wingate load is prescribed from peak isometric force (Fiso):

```
P30_predicted = (0.625 × Fiso/BW + 1.5) × BW   [W]
```

Zero-torque calibration runs automatically before each stage.

---

## Output

Results are saved to `results/` as CSV:

```
results/
  W4_wingate_20260101_120000.csv   # raw sample data
  W4_summary_20260101_120500.csv   # per-stage summary metrics
```

---

## Project structure

```
software/python/
├── config.py              Hardware calibration & participant parameters
├── run_ergometer.py       Main entry point (replaces HJCODE_250718.m)
├── requirements.txt
└── ergometer/
    ├── daq.py             LabJack U3-HV interface + brake control
    ├── protocol.py        Wingate / sprint / GXT protocols
    ├── visualization.py   Real-time scrolling display
    └── data_manager.py    CSV logging
```

---

## License

MIT License — see [LICENSE](LICENSE).

## Citation

If you use this software, please cite the associated journal article (see above).
