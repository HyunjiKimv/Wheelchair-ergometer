"""
LabJack U3-HV DAQ interface.

Reads AIN0-3 (torque L, RPM L, torque R, RPM R) and controls the powder
brake via LJTick-DAC on FIO4/FIO5.

Requires: labjackpython  (pip install labjackpython)
"""

from __future__ import annotations
import math
import time
import logging
import numpy as np

try:
    import u3  # labjackpython
    LABJACK_AVAILABLE = True
except ImportError:
    LABJACK_AVAILABLE = False
    logging.warning("labjackpython not found — running in SIMULATION mode.")

log = logging.getLogger(__name__)


class ErgometerDAQ:
    """Low-level interface for LabJack U3-HV + LJTick-DAC."""

    def __init__(self, cfg: dict, daq_cfg: dict):
        self._cal = cfg
        self._pins = daq_cfg
        #self._device = None
        self._input_device = None
        self._output_device = None
        self._zero_offset_L = 0.0  # V — updated by zero_calibrate()
        self._zero_offset_R = 0.0

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self) -> None:
        if not LABJACK_AVAILABLE:
            log.info("Simulation mode: no hardware connected.")
            return
        
        # Input DAQ: reads AIN0-3
        self._input_device = u3.U3(
        firstFound=False,
        serial=self._pins["input_serial"]
        )

        # Output DAQ: controls LJTick-DAC via I2C
        self._output_device = u3.U3(
            firstFound=False,
            serial=self._pins["output_serial"]
        )

        # Input DAQ: FIO0-3 as analog inputs
        self._input_device.configIO(FIOAnalog=0x0F)

        # Output DAQ: FIO4/FIO5 remain digital for I2C
        # Do NOT use writeRegister(7000 + pin) here.

        log.info(
        "Input LabJack connected. Serial: %s, FW: %s",
        self._pins["input_serial"],
        self._input_device.firmwareVersion,
        )

        log.info(
            "Output LabJack connected. Serial: %s, FW: %s",
            self._pins["output_serial"],
            self._output_device.firmwareVersion,
        )

    def disconnect(self) -> None:
        self.set_brake_voltage(0.0)
        if self._input_device:
            self._input_device.close()
            self._input_device = None

        if self._output_device:
            self._output_device.close()
            self._output_device = None

        log.info("LabJacks disconnected.")

    # ------------------------------------------------------------------
    # Zero calibration (call before each test with brake released)
    # ------------------------------------------------------------------
    def zero_calibrate(self, n_samples: int = 50) -> tuple[float, float]:
        """Record no-load torque baseline and store as zero offset."""
        log.info("Zero calibration — keep rollers unloaded …")
        vl, vr = [], []
        for _ in range(n_samples):
            raw = self._read_raw()
            vl.append(raw["V0"])
            vr.append(raw["V2"])
            time.sleep(1 / self._cal["sample_rate"])
        self._zero_offset_L = float(np.mean(vl))
        self._zero_offset_R = float(np.mean(vr))
        log.info(
            "Zero offsets — L: %.4f V, R: %.4f V",
            self._zero_offset_L,
            self._zero_offset_R,
        )
        return self._zero_offset_L, self._zero_offset_R

    # ------------------------------------------------------------------
    # Single sample read
    # ------------------------------------------------------------------
    def read_sample(self) -> dict:
        """Return one calibrated sample dict."""
        raw = self._read_raw()

        torque_L = (raw["V0"] - self._zero_offset_L) * self._cal["torque_scale"]
        rpm_L    = raw["V1"] * self._cal["rpm_scale"]
        torque_R = (raw["V2"] - self._zero_offset_R) * self._cal["torque_scale"]
        rpm_R    = raw["V3"] * self._cal["rpm_scale"]

        # Power = T [N·m] × ω [rad/s]   (T in kgf·cm → N·m: ×9.8×0.01)
        omega_L  = rpm_L * 2 * math.pi / 60
        omega_R  = rpm_R * 2 * math.pi / 60
        power_L  = torque_L * 9.8 * 0.01 * omega_L   # W  (torque still kgf·cm here)
        power_R  = torque_R * 9.8 * 0.01 * omega_R

        return {
            "torque_L": torque_L,  # kgf·cm
            "rpm_L":    rpm_L,
            "power_L":  power_L,
            "torque_R": torque_R,
            "rpm_R":    rpm_R,
            "power_R":  power_R,
        }

    # ------------------------------------------------------------------
    # Brake control
    # ------------------------------------------------------------------
    def set_brake_voltage(self, voltage: float) -> None:
        """Output voltage [0–10 V] to both DACA and DACB of LJTick-DAC."""
        voltage = max(0.0, min(self._cal["max_voltage"], voltage))
        
        if self._output_device is None:
            return  # simulation
        
        self._tdac_set(channel=0, voltage=voltage)  # DACA → left brake
        self._tdac_set(channel=1, voltage=voltage)  # DACB → right brake

    def voltage_to_torque(self, voltage: float) -> float:
        """Convert brake voltage to torque using polynomial calibration."""
        p = self._cal["brake_poly"]  # [a3, a2, a1, a0]
        return p[0]*voltage**3 + p[1]*voltage**2 + p[2]*voltage + p[3]

    def torque_to_voltage(self, torque_nm: float) -> float:
        """Invert polynomial to get voltage for a target torque [N·m]."""
        # Solve cubic numerically via np.roots
        p = self._cal["brake_poly"]
        coeffs = [p[0], p[1], p[2], p[3] - torque_nm]
        roots = np.roots(coeffs)
        # Pick real root in [0, max_voltage]
        real_roots = [
            r.real for r in roots
            if abs(r.imag) < 1e-6 and 0 <= r.real <= self._cal["max_voltage"]
        ]
        if not real_roots:
            # Fallback to linear approximation
            return torque_nm / self._cal["brake_scale"]
        return float(min(real_roots))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _read_raw(self) -> dict:
        if self._input_device is None:
            return self._simulate_raw()
        # Batch-read AIN0-3 for temporal consistency
        voltages = [
            self._input_device.getAIN(self._pins["ain_torque_L"]),
            self._input_device.getAIN(self._pins["ain_rpm_L"]),
            self._input_device.getAIN(self._pins["ain_torque_R"]),
            self._input_device.getAIN(self._pins["ain_rpm_R"]),
        ]
        return {
            "V0": voltages[0],
            "V1": voltages[1],
            "V2": voltages[2],
            "V3": voltages[3],
        }

    def _tdac_set(self, channel: int, voltage: float) -> None:
        """Write voltage to LJTick-DAC channel (0=A, 1=B).

        LJTick-DAC uses TLV5618 12-bit DAC via I²C.
        With an external 10 V reference the output spans 0–10 V.
        """
        dac_val = int(voltage / self._cal["max_voltage"] * 4095)
        dac_val = max(0, min(4095, dac_val))
        msb = (dac_val >> 4) & 0xFF
        lsb = (dac_val & 0x0F) << 4
        reg_byte = 0x00 if channel == 0 else 0x10
        self._output_device.i2c(
            self._pins["tdac_address"],
            [reg_byte, msb, lsb],
            SDAPinNum=self._pins["tdac_sda_pin"],
            SCLPinNum=self._pins["tdac_scl_pin"],
        )

    @staticmethod
    def _simulate_raw() -> dict:
        """Generate synthetic sensor data for offline testing."""
        t = time.time()
        return {
            "V0": 2.0 + 0.3 * math.sin(t * 2),  # torque_L ~ 10 N·m
            "V1": 3.0 + 0.5 * math.sin(t * 1.5),  # rpm_L ~ 300 RPM
            "V2": 1.9 + 0.3 * math.sin(t * 2 + 0.1),
            "V3": 3.0 + 0.5 * math.sin(t * 1.5 + 0.1),
        }
