"""
Ergometer system configuration.
Edit this file before each session to match your participant and hardware.
"""

# ---------------------------------------------------------------------------
# Hardware calibration  (DO NOT change unless sensor/brake is replaced)
# ---------------------------------------------------------------------------
CALIBRATION = {
    # Torque sensor: RSCR RI-50W, 500 kgf·cm full scale → 10 V
    # 500 kgf·cm = 500 × 9.80665 × 0.01 N·m = 49.03 N·m
    "torque_scale":  49.03 / 10,   # N·m per volt (kgf·cm per volt × 9.8 × 0.01)
    "torque_offset": 0.0,           # V — updated by zero-calibration at runtime

    # RPM encoder: 1000 RPM full scale → 10 V
    "rpm_scale": 1000 / 10,         # RPM per volt

    # Electromagnetic powder brake: PORA PRB-5Y4, 25 N·m at 10 V
    # Nonlinear fit: τ = 1.673 − 0.026V + 0.123V² + 0.012V³  (R² = 0.9946)
    # Use polynomial for accurate resistance prescription; linear below for
    # simple voltage-only override.
    "brake_scale":  25 / 10,        # N·m per volt (nominal linear)
    "brake_poly":   [0.012, 0.123, -0.026, 1.673],  # [a3,a2,a1,a0] of τ(V)
    "max_voltage":  10.0,           # V

    # Acquisition
    "sample_rate":  50,             # Hz
}

# ---------------------------------------------------------------------------
# Hardware wiring  (LabJack U3-HV pin assignments)
# ---------------------------------------------------------------------------
DAQ = {
    # LabJack U3-HV serial numbers
    "input_serial":  320101580,
    "output_serial": 320107286,

    # Analog inputs (AIN channel → signal)
    "ain_torque_L": 0,   # FIO0 ← left torque sensor indicator output
    "ain_rpm_L":    1,   # FIO1 ← left RPM encoder output
    "ain_torque_R": 2,   # FIO2 ← right torque sensor indicator output
    "ain_rpm_R":    3,   # FIO3 ← right RPM encoder output

    # LJTick-DAC I²C (for brake voltage control)
    "tdac_scl_pin": 4,   # FIO4
    "tdac_sda_pin": 5,   # FIO5
    "tdac_address": 0x12,
}

# ---------------------------------------------------------------------------
# Physical dimensions
# ---------------------------------------------------------------------------
ERGOMETER = {
    "roller_radius": 0.035,   # m  (Wheely-X roller diameter 70 mm)
    "wheel_radius":  0.305,   # m  (standard wheelchair wheel 610 mm)
    "rim_radius":    0.265,   # m  (handrim 530 mm)
}

# ---------------------------------------------------------------------------
# Participant  — update before each session
# ---------------------------------------------------------------------------
PARTICIPANT = {
    "id":               "W4",
    "total_force":      659,    # N  (participant + wheelchair weighed on scale)
    "wheelchair_force": 98,     # N  (wheelchair alone, ≈ 10 kg × 9.8)
    "target_velocity":  1.0,    # m/s  (target rim velocity for Wingate)
    "sport":            "fencing",
}
# Derived
PARTICIPANT["body_weight"] = (
    (PARTICIPANT["total_force"] - PARTICIPANT["wheelchair_force"]) / 9.81
)  # kg
PARTICIPANT["total_mass"] = PARTICIPANT["total_force"] / 9.81  # kg

# ---------------------------------------------------------------------------
# Protocol defaults
# ---------------------------------------------------------------------------
PROTOCOL = {
    "warmup_duration":   180,   # s
    "warmup_voltage":    1.5,   # V

    "isometric_trials":  3,
    "isometric_push_s":  5,     # s per push
    "isometric_rest_s":  30,    # s rest between trials

    "sprint_trials":     2,
    "sprint_duration":   10,    # s
    "sprint_rest_s":     120,   # s
    "sprint_mu":         0.012, # rolling resistance coefficient

    "wingate_duration":  30,    # s

    "gxt_steps":         15,
    "gxt_step_duration": 60,    # s
}
