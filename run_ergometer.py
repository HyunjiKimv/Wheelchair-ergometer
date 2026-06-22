"""
run_ergometer.py — main entry point (replaces HJCODE_250718.m)

Usage:
    python run_ergometer.py [--mode {wingate|sprint|gxt|isometric|demo}]
    python run_ergometer.py --mode demo     # offline simulation, no hardware
"""

import argparse
import sys
import logging
import threading
import traceback

from config import CALIBRATION, DAQ, ERGOMETER, PARTICIPANT, PROTOCOL
from ergometer import (
    ErgometerDAQ,
    WingateProtocol,
    DataLogger,
    SessionSummary,
)
from ergometer.visualization import ErgometerMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)

"""
main thread:
    mon.start()  # Matplotlib 창 2개 실행

worker thread:
    warmup / isometric / sprint / wingate / gxt 실행
    DAQ read/write
    monitor.push(sample)
"""

# ---------------------------------------------------------------------------
def _build_objects(simulate: bool = False):
    daq = ErgometerDAQ(CALIBRATION, DAQ)

    if simulate:
        # Force simulation (no hardware)
        daq._input_device = None
        daq._output_device = None
    else:
        daq.connect()

    proto = WingateProtocol(daq, CALIBRATION, PARTICIPANT, ERGOMETER, PROTOCOL)
    summary = SessionSummary(PARTICIPANT["id"])
    return daq, proto, summary


# ---------------------------------------------------------------------------
def _run_with_gui(worker_func, simulate: bool = False):
    """
    Run Matplotlib GUI on the main thread and the experiment protocol
    on a background worker thread.

    This avoids:
        UserWarning: Starting a Matplotlib GUI outside of the main thread
    """

    daq, proto, summary = _build_objects(simulate)
    
    # Total expected active test time (warmup + iso + sprint + wingate)
    total_active_s = (
        PROTOCOL["warmup_duration"]
        + PROTOCOL["isometric_trials"] * PROTOCOL["isometric_push_s"]
        + PROTOCOL["sprint_trials"]    * PROTOCOL["sprint_duration"]
        + PROTOCOL["wingate_duration"]
    )
    mon = ErgometerMonitor(
        window_s=10,
        sample_rate=CALIBRATION["sample_rate"],
        ergometer_dims=ERGOMETER,
    )
    mon.configure_session(mode="wingate", total_active_s=total_active_s)

    # Optional: if protocol.py uses self.monitor internally.
    proto.monitor = mon # 프로토콜 코드에서 monitor.push(sample)을 쓰는 구조라면

    worker_error = []

    def worker():
        try:
            worker_func(daq, proto, summary, mon)

        except Exception as exc:
            worker_error.append(exc)
            traceback.print_exc()

        finally:
            try:
                daq.disconnect()
            except Exception:
                traceback.print_exc()

            try:
                summary.save()
            except Exception:
                traceback.print_exc()

            try:
                mon.stop()
            except Exception:
                pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # IMPORTANT:
    # Matplotlib GUI must run on the main thread.
    mon.start()

    # When GUI closes, wait briefly for worker cleanup.
    t.join(timeout=5)

    if worker_error:
        raise worker_error[0]


# ---------------------------------------------------------------------------
def _wingate_worker(daq, proto, summary, mon):
    """
    Full Wingate pipeline:
        warmup -> isometric -> sprint -> wingate
    """

    # Step 1: warmup
    print("\n========== WARMUP ==========")
    with DataLogger("results", PARTICIPANT["id"], "warmup") as data_log:
        warmup_r = proto.warmup(monitor=mon, logger=data_log)
    summary.add("warmup", warmup_r)

    # Step 2: isometric
    print("\n========== ISOMETRIC ==========")
    with DataLogger("results", PARTICIPANT["id"], "isometric") as data_log:
        iso_r = proto.run_isometric(monitor=mon, logger=data_log)
    summary.add("isometric", iso_r)

    # Step 3: sprint
    print("\n========== SPRINT ==========")
    with DataLogger("results", PARTICIPANT["id"], "sprint") as data_log:
        sprint_r = proto.run_sprint(monitor=mon, logger=data_log)
    summary.add("sprint", sprint_r)

    # Step 4: wingate
    print("\n========== WINGATE ==========")
    torque, voltage, P30pred = proto.calculate_wingate_from_fiso(iso_r.max_fiso)

    print(
        f"[Wingate Prescription] "
        f"Torque={torque:.3f} N·m, "
        f"Voltage={voltage:.3f} V, "
        f"P30pred={P30pred:.1f} W"
    )

    with DataLogger("results", PARTICIPANT["id"], "wingate") as data_log:
        wingate_r = proto.run_wingate(voltage, monitor=mon, logger=data_log)

    wingate_r.target_torque = torque
    wingate_r.target_voltage = voltage
    summary.add("wingate", wingate_r)

    print("\nSession complete.")


def run_wingate(simulate: bool = False):
    _run_with_gui(_wingate_worker, simulate=simulate)


# ---------------------------------------------------------------------------
def _gxt_worker(daq, proto, summary, mon):
    """
    GXT pipeline:
        sprint pre-test -> calculate GXT voltages -> GXT steps
    """
    total_active_s = (
        PROTOCOL["sprint_trials"] * PROTOCOL["sprint_duration"]
        + PROTOCOL["gxt_steps"]   * PROTOCOL["gxt_step_duration"]
    )
    mon.configure_session(mode="gxt", total_active_s=total_active_s)

    print("\n========== SPRINT (for GXT prescription) ==========")
    with DataLogger("results", PARTICIPANT["id"], "sprint_pre_gxt") as data_log:
        sprint_r = proto.run_sprint(monitor=mon, logger=data_log)
    summary.add("sprint", sprint_r)

    from ergometer.protocol import WingateResults

    dummy = WingateResults(POmax=sprint_r.POmax)
    voltages = proto.calculate_gxt_voltages(dummy)

    print("\n========== GXT ==========")
    for i, raw_step in enumerate(proto.run_gxt(voltages, monitor=mon)):
        with DataLogger("results", PARTICIPANT["id"], f"gxt_step{i + 1}") as data_log:
            for j in range(len(raw_step["time"])):
                row = {k: float(raw_step[k][j]) for k in raw_step}
                data_log.write(row)

    print("\nGXT complete.")


def run_gxt(simulate: bool = False):
    _run_with_gui(_gxt_worker, simulate=simulate)
    
# ---------------------------------------------------------------------------
def run_demo():
    """Offline simulation: runs full Wingate pipeline with synthetic data."""
    print("=== DEMO MODE (no hardware) ===")
    run_wingate(simulate=True)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Wheelchair ergometer test controller"
    )
    parser.add_argument(
        "--mode",
        choices=["wingate", "gxt", "demo"],
        default="wingate",
        help="Test protocol to run (default: wingate)",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    elif args.mode == "wingate":
        run_wingate()
    elif args.mode == "gxt":
        run_gxt()
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
