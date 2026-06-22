"""
WingateProtocol — mirrors WingateProtocol.m

Implements:
  warmup · isometric · sprint · wingate · GXT
"""

from __future__ import annotations
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .daq import ErgometerDAQ
from .visualization import ErgometerMonitor
from .data_manager import DataLogger

log = logging.getLogger(__name__)


@dataclass
class SprintResults:
    POmax: float = 0.0
    POmean: float = 0.0
    RPMmax: float = 0.0
    RPMmean: float = 0.0
    vmax: float = 0.0
    vmean: float = 0.0
    valid: bool = True
    max_POmax: float = 0.0
    mean_POmean: float = 0.0
    max_RPM: float = 0.0
    mean_RPM: float = 0.0


@dataclass
class WingateResults:
    P30: float = 0.0
    P5: float = 0.0
    POmax: float = 0.0
    vmax: float = 0.0
    valid: bool = True
    target_torque: float = 0.0
    target_voltage: float = 0.0


@dataclass
class IsometricResults:
    max_fiso: float = 0.0
    mean_fiso: float = 0.0
    fiso_values: list = field(default_factory=list)


class WingateProtocol:
    """Full exercise testing protocol for the wheelchair ergometer."""

    def __init__(
        self,
        daq: ErgometerDAQ,
        calibration: dict,
        participant: dict,
        ergometer: dict,
        protocol_cfg: dict,
    ):
        self.daq = daq
        self.cal = calibration
        self.p   = participant
        self.erg = ergometer
        self.cfg = protocol_cfg

        print(f"\nWingateProtocol initialized")
        print(f"  Participant: {self.p['id']}")
        print(f"  Body weight: {self.p['body_weight']:.1f} kg")
        print(f"  Roller radius: {self.erg['roller_radius']*1000:.0f} mm")
        print(f"  Wheel radius:  {self.erg['wheel_radius']*1000:.0f} mm")
        print(f"  Brake scale:   {self.cal['brake_scale']:.1f} N·m/V")

    # ──────────────────────────────────────────────────────────────────
    # Internal acquisition loop — shared by all protocol stages
    # ──────────────────────────────────────────────────────────────────
    def _run_test(
        self,
        test_name: str,
        voltage: float,
        duration_s: float,
        monitor: Optional[ErgometerMonitor] = None,
        logger: Optional[DataLogger] = None,
        display_mode: str = "warmup",
        stage_num: int = 0,
        stage_total: int = 0,
        instruction: str = "",
        target_speed: Optional[float] = None,
        gxt_target_lo: Optional[float] = None,
        gxt_target_hi: Optional[float] = None,
    ) -> dict:
        """Core acquisition loop used by every protocol stage."""

        print(f"\n[{test_name}] Starting — {duration_s:.0f} s @ {voltage:.3f} V")

        if monitor:
            monitor.set_stage(
                stage_name=test_name,
                stage_num=stage_num,
                stage_total=stage_total,
                stage_dur_s=duration_s,
                display_mode=display_mode,
                instruction=instruction,
                target_speed=target_speed,
                gxt_target_lo=gxt_target_lo,
                gxt_target_hi=gxt_target_hi,
            )

        self.daq.zero_calibrate()
        self.daq.set_brake_voltage(voltage)

        records = []
        t0 = time.perf_counter()
        sample_period = 1.0 / self.cal["sample_rate"]

        while True:
            # Respect pause — returns False if stop was requested
            if monitor and not monitor.wait_if_paused():
                print(f"[{test_name}] Stopped by user.")
                break

            t_now = time.perf_counter() - t0
            if t_now >= duration_s:
                break

            sample = self.daq.read_sample()
            sample["time"]    = t_now
            sample["voltage"] = voltage
            records.append(sample)

            if monitor:
                monitor.push(sample)
            if logger:
                logger.write(sample)

            elapsed = time.perf_counter() - t0 - t_now
            sleep_t = sample_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        self.daq.set_brake_voltage(0.0)
        print(f"[{test_name}] Done — {len(records)} samples collected.")
        return self._pack_results(records)

    @staticmethod
    def _pack_results(records: list[dict]) -> dict:
        keys = ["time", "torque_L", "rpm_L", "power_L",
                "torque_R", "rpm_R", "power_R", "voltage"]
        return {k: np.array([r[k] for r in records]) for k in keys}

    def _rest(
        self,
        rest_s: float,
        monitor: Optional[ErgometerMonitor] = None,
        next_stage: str = "",
    ):
        """Rest between trials. Shows countdown on participant display."""
        print(f"  Rest {rest_s:.0f} s …")
        if monitor:
            monitor.start_rest(rest_s, next_stage)
        time.sleep(rest_s)
        if monitor:
            monitor.end_rest()

    # ──────────────────────────────────────────────────────────────────
    # 0. Warmup
    # ──────────────────────────────────────────────────────────────────
    def warmup(self, monitor=None, logger=None) -> SprintResults:
        if monitor:
            monitor.announce_stage("Warmup", "Stage 1 / 5 — sub-maximal propulsion")
            monitor.wait_for_start()
            monitor.countdown(3)
        raw = self._run_test(
            test_name="Warmup",
            voltage=self.cfg["warmup_voltage"],
            duration_s=self.cfg["warmup_duration"],
            monitor=monitor, logger=logger,
            display_mode="warmup",
            stage_num=1, stage_total=5,
        )
        return self._analyze_sprint(raw)

    # ──────────────────────────────────────────────────────────────────
    # 1. Isometric test
    # ──────────────────────────────────────────────────────────────────
    def run_isometric(self, monitor=None, logger=None) -> IsometricResults:
        n_trials = self.cfg["isometric_trials"]
        push_s   = self.cfg["isometric_push_s"]
        rest_s   = self.cfg["isometric_rest_s"]

        if monitor:
            monitor.reset_iso_results()

        fiso_values = []
        for trial in range(1, n_trials + 1):
            print(f"\n--- Isometric trial {trial}/{n_trials} ---")

            if monitor:
                monitor.announce_stage(
                    "Isometric",
                    f"Trial {trial} / {n_trials} — maximum push",
                )
                monitor.wait_for_start()
                monitor.countdown(3)

            raw = self._run_test(
                test_name="Isometric",
                voltage=10.0,
                duration_s=push_s,
                monitor=monitor, logger=logger,
                display_mode="force",
                stage_num=trial, stage_total=n_trials,
            )
            total_torque = raw["torque_L"] + raw["torque_R"]
            fiso = float(np.max(total_torque) * 9.8 * 0.01)  # kgf·cm → N·m
            fiso_values.append(fiso)
            print(f"  Trial {trial} Fiso: {fiso:.2f} N·m")

            if monitor:
                monitor.record_iso_trial(trial, fiso)

            if trial < n_trials:
                next_lbl = f"Isometric Trial {trial + 1}"
                self._rest(rest_s, monitor, next_stage=next_lbl)

        result = IsometricResults(
            max_fiso=max(fiso_values),
            mean_fiso=float(np.mean(fiso_values)),
            fiso_values=fiso_values,
        )
        print(f"\n=== Isometric complete ===")
        print(f"  Max Fiso:  {result.max_fiso:.2f} N·m")
        print(f"  Mean Fiso: {result.mean_fiso:.2f} N·m")
        return result

    # ──────────────────────────────────────────────────────────────────
    # 2. Sprint test
    # ──────────────────────────────────────────────────────────────────
    def calculate_sprint_torque(self) -> tuple[float, float]:
        resistance_force = self.cfg["sprint_mu"] * self.p["total_force"]
        sprint_torque    = resistance_force * self.erg["roller_radius"]
        sprint_voltage   = self.daq.torque_to_voltage(sprint_torque)
        sprint_voltage   = min(sprint_voltage, self.cal["max_voltage"])

        print(f"\n=== Sprint settings ===")
        print(f"  μ = {self.cfg['sprint_mu']}")
        print(f"  Resistance force: {resistance_force:.1f} N")
        print(f"  Brake torque:     {sprint_torque:.3f} N·m")
        print(f"  Voltage:          {sprint_voltage:.3f} V")
        return sprint_torque, sprint_voltage

    def run_sprint(self, monitor=None, logger=None) -> SprintResults:
        _, voltage = self.calculate_sprint_torque()
        n_trials   = self.cfg["sprint_trials"]
        rest_s     = self.cfg["sprint_rest_s"]
        duration   = self.cfg["sprint_duration"]

        trial_results = []
        for trial in range(1, n_trials + 1):
            print(f"\n--- Sprint trial {trial}/{n_trials} ---")

            if monitor:
                monitor.announce_stage(
                    "Sprint",
                    f"Trial {trial} / {n_trials} — all-out sprint",
                )
                monitor.wait_for_start()
                monitor.countdown(3)

            raw = self._run_test(
                test_name="Sprint",
                voltage=voltage,
                duration_s=duration,
                monitor=monitor, logger=logger,
                display_mode="sprint",
                stage_num=trial, stage_total=n_trials,
            )
            trial_results.append(self._analyze_sprint(raw))

            if trial < n_trials:
                next_lbl = f"Sprint Trial {trial + 1}"
                self._rest(rest_s, monitor, next_stage=next_lbl)

        result = SprintResults(
            max_POmax   = max(r.POmax for r in trial_results),
            mean_POmean = float(np.mean([r.POmean for r in trial_results])),
            max_RPM     = max(r.RPMmax for r in trial_results),
            mean_RPM    = float(np.mean([r.RPMmean for r in trial_results])),
        )
        result.POmax  = result.max_POmax
        result.POmean = result.mean_POmean
        print(f"\n=== Sprint complete ===")
        print(f"  Peak power: {result.max_POmax:.0f} W")
        print(f"  Peak RPM:   {result.max_RPM:.0f}")
        return result

    # ──────────────────────────────────────────────────────────────────
    # 3. Wingate test
    # ──────────────────────────────────────────────────────────────────
    def calculate_wingate_from_fiso(
        self, fiso_max: float
    ) -> tuple[float, float, float]:
        fiso_rel    = fiso_max / self.p["body_weight"]
        P30pred_rel = 0.625 * fiso_rel + 1.5
        P30pred     = P30pred_rel * self.p["body_weight"]

        target_rim_omega = self.p["target_velocity"] / self.erg["rim_radius"]
        mtotal = self.p["total_mass"]
        mu     = P30pred / (mtotal * 9.81 * self.p["target_velocity"])

        roller_omega = target_rim_omega * (self.erg["wheel_radius"] / self.erg["roller_radius"])
        torque       = P30pred / roller_omega
        voltage      = self.daq.torque_to_voltage(torque)
        voltage      = min(voltage, self.cal["max_voltage"])

        print(f"\n=== Wingate prescription (Fiso-based) ===")
        print(f"  Fiso:         {fiso_max:.2f} N·m  ({fiso_rel:.3f} N·m/kg)")
        print(f"  P30pred:      {P30pred:.0f} W")
        print(f"  μ:            {mu:.4f}")
        print(f"  Brake torque: {torque:.2f} N·m")
        print(f"  Voltage:      {voltage:.3f} V")
        return torque, voltage, P30pred

    def run_wingate(self, voltage: float, monitor=None, logger=None) -> WingateResults:
        if monitor:
            monitor.announce_stage("Wingate", "30 s all-out — maintain power!")
            monitor.wait_for_start()
            monitor.countdown(3)

        raw = self._run_test(
            test_name="Wingate",
            voltage=voltage,
            duration_s=self.cfg["wingate_duration"],
            monitor=monitor, logger=logger,
            display_mode="wingate",
            stage_num=4, stage_total=5,
        )
        return self._analyze_wingate(raw)

    # ──────────────────────────────────────────────────────────────────
    # 4. GXT
    # ──────────────────────────────────────────────────────────────────
    def calculate_gxt_voltages(self, wingate_results: WingateResults) -> np.ndarray:
        steps     = self.cfg["gxt_steps"]
        ref_omega = 333 * 2 * math.pi / 60
        start_v   = self.daq.torque_to_voltage(wingate_results.POmax * 0.15 / ref_omega)
        end_v     = min(self.daq.torque_to_voltage(wingate_results.POmax * 1.20 / ref_omega),
                        self.cal["max_voltage"])
        seq = np.linspace(start_v, end_v, steps)
        print(f"\n=== GXT voltage sequence ===")
        print(f"  {steps} steps: {start_v:.2f} V → {end_v:.2f} V")
        return seq

    def run_gxt(
        self,
        voltage_sequence: np.ndarray,
        monitor=None,
        logger=None,
        target_speed: Optional[float] = None,
    ) -> list[dict]:
        step_duration = self.cfg["gxt_step_duration"]
        n_steps       = len(voltage_sequence)
        results = []
        for i, v in enumerate(voltage_sequence):
            print(f"\n--- GXT step {i+1}/{n_steps} — {v:.3f} V ---")

            if monitor:
                monitor.announce_stage(
                    "GXT",
                    f"Step {i+1} / {n_steps}",
                )
                monitor.wait_for_start()
                monitor.countdown(3)

            target_rim_spd = target_speed

            raw = self._run_test(
                test_name="GXT",
                voltage=v,
                duration_s=step_duration,
                monitor=monitor, logger=logger,
                display_mode="gxt",
                stage_num=i + 1, stage_total=n_steps,
                target_speed=target_rim_spd,
            )
            results.append(raw)

            if i < n_steps - 1:
                self._rest(
                    self.cfg.get("gxt_rest_s", 0),
                    monitor,
                    next_stage=f"GXT Step {i+2}",
                )
        return results

    # ──────────────────────────────────────────────────────────────────
    # Analysis helpers
    # ──────────────────────────────────────────────────────────────────
    def _analyze_sprint(self, raw: dict) -> SprintResults:
        total_power = raw["power_L"] + raw["power_R"]
        total_rpm   = (raw["rpm_L"] + raw["rpm_R"]) / 2
        rim_vel = (
            total_rpm * 2 * math.pi / 60
            * (self.erg["roller_radius"] / self.erg["wheel_radius"])
            * self.erg["rim_radius"]
        )
        r = SprintResults(
            POmax   = float(np.max(total_power)),
            POmean  = float(np.mean(total_power)),
            RPMmax  = float(np.max(total_rpm)),
            RPMmean = float(np.mean(total_rpm)),
            vmax    = float(np.max(rim_vel)),
            vmean   = float(np.mean(rim_vel)),
        )
        r.valid       = r.vmax < 4.0
        r.max_POmax   = r.POmax
        r.mean_POmean = r.POmean
        r.max_RPM     = r.RPMmax
        r.mean_RPM    = r.RPMmean
        return r

    def _analyze_wingate(self, raw: dict) -> WingateResults:
        total_power = raw["power_L"] + raw["power_R"]
        total_rpm   = (raw["rpm_L"] + raw["rpm_R"]) / 2
        sr          = self.cal["sample_rate"]
        window      = 5 * sr

        if len(total_power) >= window:
            p5 = max(
                float(np.mean(total_power[i : i + window]))
                for i in range(len(total_power) - window + 1)
            )
        else:
            p5 = float(np.max(total_power))

        rim_vel = (
            total_rpm * 2 * math.pi / 60
            * (self.erg["roller_radius"] / self.erg["wheel_radius"])
            * self.erg["rim_radius"]
        )
        r = WingateResults(
            P30   = float(np.mean(total_power)),
            P5    = p5,
            POmax = float(np.max(total_power)),
            vmax  = float(np.max(rim_vel)),
        )
        r.valid = r.vmax < 3.0
        print(f"\n=== Wingate results ===")
        print(f"  P30:   {r.P30:.0f} W")
        print(f"  P5:    {r.P5:.0f} W")
        print(f"  POmax: {r.POmax:.0f} W")
        print(f"  Valid: {r.valid}")
        return r
