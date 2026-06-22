"""
Dual-window ergometer monitor.

Experimenter Monitor (13×8 in)
  Row 0: Session Status box | Power Feedback bar
  Row 1: Torque (L/R)        | RPM (L/R)
  Row 2: Mechanical Power    | Brake Voltage
  Buttons: [START ▶] [PAUSE] [STOP]

Participant Display (9×7 in, dark theme)
  Row 0: Speed chart
  Row 1: Stage-specific feedback panel
          "announce" -> stage name + subtitle, wait for experimenter START
          "rest"     -> REST title + live countdown + next-stage label
          "warmup"   -> speed chart + instruction
          "force"    -> vertical trial-comparison bar chart (isometric)
          "sprint"   -> speed + instruction
          "wingate"  -> power-retention bar
          "gxt"      -> target speed band + step info
          "vt_band"  -> VT1/VT2 zone with marker
  Row 2: Stage progress bar
  Row 3: Speed value | Stage time

Thread model:
  main thread  -> mon.start()  (plt.show blocks here)
  worker thread -> protocol loop ->
      mon.announce_stage(name, sub)   # show splash, set _waiting_for_start
      mon.wait_for_start()            # block until experimenter presses START
      mon.countdown(n)                # 3-2-1 overlay
      mon.push(sample)                # 50 Hz samples
      mon.start_rest(rest_s, next)    # switch to rest countdown display
      mon.end_rest()                  # clear rest display
      mon.record_iso_trial(n, fiso)
      mon.wait_if_paused()
"""

from __future__ import annotations
import math
import queue
import threading
import time
from collections import deque
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

_N_ISO_BARS = 3


class ErgometerMonitor:

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        window_s: float = 10,
        sample_rate: int = 50,
        ergometer_dims: Optional[dict] = None,
    ):
        sr  = sample_rate
        win = int(window_s * sr)
        self._sr = sr
        erg = ergometer_dims or {}
        self._roller_r = erg.get("roller_radius", 0.035)
        self._wheel_r  = erg.get("wheel_radius",  0.305)
        self._rim_r    = erg.get("rim_radius",    0.265)

        self._q: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=2000)
        self._lock       = threading.Lock()
        self._running    = False
        self._paused     = False
        self._stop_event = threading.Event()

        # Scrolling buffers
        self._t        = deque(maxlen=win)
        self._torque_L = deque(maxlen=win)
        self._torque_R = deque(maxlen=win)
        self._rpm_L    = deque(maxlen=win)
        self._rpm_R    = deque(maxlen=win)
        self._power_L  = deque(maxlen=win)
        self._power_R  = deque(maxlen=win)
        self._voltage  = deque(maxlen=win)
        self._speed_L  = deque(maxlen=win)
        self._speed_R  = deque(maxlen=win)

        # Session state
        self._mode           = "demo"
        self._total_active_s = 0.0
        self._session_start_t: Optional[float] = None
        self._rest_s         = 0.0
        self._rest_count     = 0
        self._pause_start_t: Optional[float] = None

        # Stage state
        self._stage_name    = "—"
        self._stage_num     = 0
        self._stage_total   = 0
        self._stage_dur_s   = 0.0
        self._stage_start_t: Optional[float] = None
        self._display_mode  = "warmup"
        self._instruction   = ""
        self._target_speed: Optional[float] = None
        self._gxt_target_lo: Optional[float] = None
        self._gxt_target_hi: Optional[float] = None

        # VT thresholds
        self._vt1: Optional[float] = None
        self._vt2: Optional[float] = None

        # Announce state (set by announce_stage, cleared by _on_start)
        self._waiting_for_start = False
        self._announce_name     = ""
        self._announce_sub      = ""

        # Rest display state
        self._rest_dur_s        = 0.0
        self._rest_start_t_disp: Optional[float] = None
        self._rest_next_stage   = ""

        # Wingate peak
        self._wingate_peak = 1.0

        # Isometric trial tracking
        self._iso_n_trials    = _N_ISO_BARS
        self._iso_results     = []
        self._iso_current_max = 0.0
        self._iso_max_scale   = 50.0

        # Latest scalars
        self._latest_power = 0.0
        self._latest_speed = 0.0
        self._latest_nm    = 0.0

        # Countdown overlay
        self._countdown_val = 0

        # Top chart mode tracking (speed / torque / power)
        self._top_chart_mode = ""

        self.fig_exp  = None
        self.fig_part = None
        self._timer   = None

    # ------------------------------------------------------------------
    # Session configuration
    # ------------------------------------------------------------------
    def configure_session(
        self,
        mode: str = "demo",
        total_active_s: float = 0.0,
        vt1: Optional[float] = None,
        vt2: Optional[float] = None,
    ):
        with self._lock:
            self._mode           = mode
            self._total_active_s = total_active_s
            self._vt1 = vt1
            self._vt2 = vt2

    # ------------------------------------------------------------------
    # Stage announcement — show splash screen, block until START pressed
    # ------------------------------------------------------------------
    def announce_stage(self, stage_name: str, subtitle: str = ""):
        """Switch participant display to announcement mode."""
        with self._lock:
            self._announce_name     = stage_name
            self._announce_sub      = subtitle
            self._waiting_for_start = True
            self._display_mode      = "announce"
            self._instruction       = ""

    def wait_for_start(self) -> bool:
        """Block worker thread until experimenter clicks START (or stop)."""
        while True:
            with self._lock:
                done = not self._waiting_for_start
            if done:
                break
            if self._stop_event.is_set():
                return False
            time.sleep(0.05)
        return not self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Stage state
    # ------------------------------------------------------------------
    def set_stage(
        self,
        stage_name: str,
        stage_num: int = 0,
        stage_total: int = 0,
        stage_dur_s: float = 0.0,
        display_mode: str = "warmup",
        instruction: str = "",
        target_speed: Optional[float] = None,
        gxt_target_lo: Optional[float] = None,
        gxt_target_hi: Optional[float] = None,
    ):
        with self._lock:
            self._stage_name    = stage_name
            self._stage_num     = stage_num
            self._stage_total   = stage_total
            self._stage_dur_s   = stage_dur_s
            self._stage_start_t = time.perf_counter()
            self._display_mode  = display_mode
            self._instruction   = instruction or _default_instruction(stage_name)
            self._target_speed  = target_speed
            self._gxt_target_lo = gxt_target_lo
            self._gxt_target_hi = gxt_target_hi

            if display_mode == "wingate":
                self._wingate_peak = 1.0
            if display_mode == "force":
                self._iso_n_trials    = stage_total or _N_ISO_BARS
                self._iso_current_max = 0.0

            # Clear scrolling buffers so each trial starts with a clean chart
            for buf in (self._t, self._torque_L, self._torque_R,
                        self._rpm_L, self._rpm_R,
                        self._power_L, self._power_R,
                        self._voltage, self._speed_L, self._speed_R):
                buf.clear()

    def set_vt(self, vt1: Optional[float], vt2: Optional[float]):
        with self._lock:
            self._vt1 = vt1
            self._vt2 = vt2

    # ------------------------------------------------------------------
    # Isometric trial recording
    # ------------------------------------------------------------------
    def record_iso_trial(self, trial_num: int, max_fiso_nm: float):
        with self._lock:
            while len(self._iso_results) < trial_num:
                self._iso_results.append(0.0)
            self._iso_results[trial_num - 1] = max_fiso_nm
            if max_fiso_nm > self._iso_max_scale * 0.85:
                self._iso_max_scale = max_fiso_nm * 1.25

    def reset_iso_results(self):
        with self._lock:
            self._iso_results     = []
            self._iso_current_max = 0.0
            self._iso_max_scale   = 50.0

    # ------------------------------------------------------------------
    # Countdown (worker thread)
    # ------------------------------------------------------------------
    def countdown(self, n: int = 3):
        for i in range(n, 0, -1):
            with self._lock:
                self._countdown_val = i
            time.sleep(0.97)
        with self._lock:
            self._countdown_val = 0
        time.sleep(0.1)

    # ------------------------------------------------------------------
    # Rest tracking + display
    # ------------------------------------------------------------------
    def start_rest(self, rest_s: float = 0.0, next_stage: str = ""):
        with self._lock:
            self._pause_start_t     = time.perf_counter()
            self._rest_dur_s        = rest_s
            self._rest_start_t_disp = time.perf_counter()
            self._rest_next_stage   = next_stage
            if rest_s > 0:
                self._display_mode  = "rest"
                self._stage_dur_s   = rest_s
                self._stage_start_t = time.perf_counter()

    def end_rest(self):
        with self._lock:
            if self._pause_start_t is not None:
                self._rest_s    += time.perf_counter() - self._pause_start_t
                self._rest_count += 1
                self._pause_start_t = None
            self._rest_start_t_disp = None

    # ------------------------------------------------------------------
    # Pause / Stop
    # ------------------------------------------------------------------
    @property
    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    @property
    def is_paused(self) -> bool:
        return self._paused

    def wait_if_paused(self) -> bool:
        while self._paused:
            if self._stop_event.is_set():
                return False
            time.sleep(0.02)
        return not self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Data push (worker thread)
    # ------------------------------------------------------------------
    def push(self, sample: dict):
        try:
            self._q.put_nowait(sample)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Start / Stop (main thread)
    # ------------------------------------------------------------------
    def start(self):
        self._running         = True
        self._session_start_t = time.perf_counter()
        self._build_experimenter_window()
        self._build_participant_window()
        self._timer = self.fig_exp.canvas.new_timer(interval=40)
        self._timer.add_callback(self._update)
        self._timer.start()
        plt.show()

    def stop(self):
        self._running = False
        self._stop_event.set()
        self._paused  = False
        with self._lock:
            self._waiting_for_start = False
        try:
            if self._timer:
                self._timer.stop()
        except Exception:
            pass
        try:
            plt.close("all")
        except Exception:
            pass

    # ==================================================================
    # Build: Experimenter Monitor
    # ==================================================================
    def _build_experimenter_window(self):
        fig = plt.figure("Experimenter Monitor", figsize=(13, 8))
        fig.patch.set_facecolor("#f4f6f9")

        outer = gridspec.GridSpec(
            3, 1, figure=fig,
            height_ratios=[1.4, 3, 3],
            hspace=0.50, left=0.07, right=0.97, top=0.95, bottom=0.08,
        )

        # ── Row 0: Session Status + Power Feedback ─────────────────────
        gs0 = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0], width_ratios=[1, 3.2], wspace=0.08,
        )
        ax_s = fig.add_subplot(gs0[0])
        ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1); ax_s.axis("off")
        ax_s.set_facecolor("#e8f0fe"); ax_s.patch.set_visible(True)
        for sp in ax_s.spines.values():
            sp.set_visible(True); sp.set_edgecolor("#3a6ea5"); sp.set_linewidth(1.5)
        ax_s.text(0.07, 0.91, "Session Status",
                  transform=ax_s.transAxes, fontsize=9.5, fontweight="bold",
                  color="#1a3a6a", va="top")
        self._txt_mode   = ax_s.text(0.07, 0.68, "Mode: —",
                                      transform=ax_s.transAxes, fontsize=8.5,
                                      color="#333333", va="top")
        self._txt_active = ax_s.text(0.07, 0.47, "Active: 00:00 / 00:00",
                                      transform=ax_s.transAxes, fontsize=8.5,
                                      color="#1a6a3a", va="top")
        self._txt_rest_s = ax_s.text(0.07, 0.25, "Rest: 00:00 (0 breaks)",
                                      transform=ax_s.transAxes, fontsize=8.5,
                                      color="#8b0000", va="top")

        ax_pb = fig.add_subplot(gs0[1])
        ax_pb.set_title("Power Feedback (VT1~VT2 band)", fontsize=9.5)
        ax_pb.set_xlim(0, 1); ax_pb.set_ylim(0, 1); ax_pb.axis("off")
        ax_pb.add_patch(Rectangle((0, 0.30), 1.0, 0.32,
                                   facecolor="#cccccc", edgecolor="#999"))
        self._vt_patch_exp = Rectangle((0.3, 0.30), 0.2, 0.32,
                                        facecolor="#88cc88", edgecolor="none", alpha=0.0)
        ax_pb.add_patch(self._vt_patch_exp)
        self._vt1_line_exp = ax_pb.axvline(0.3, color="#008800", lw=1.5, ls="--", alpha=0.0)
        self._vt2_line_exp = ax_pb.axvline(0.5, color="#006600", lw=1.5, ls="--", alpha=0.0)
        self._txt_vt1_exp  = ax_pb.text(0.3, 0.65, "VT1", fontsize=8, ha="center",
                                         va="bottom", color="#005500", alpha=0.0)
        self._txt_vt2_exp  = ax_pb.text(0.5, 0.65, "VT2", fontsize=8, ha="center",
                                         va="bottom", color="#003300", alpha=0.0)
        (self._marker_exp,) = ax_pb.plot([0.02], [0.46], "k^", ms=13, zorder=5)
        self._txt_pval_exp  = ax_pb.text(0.02, 0.08, "0 W", fontsize=8.5,
                                          ha="center", color="#333")

        # ── Row 1: Torque | RPM ────────────────────────────────────────
        gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.32)
        ax_t = fig.add_subplot(gs1[0]); ax_r = fig.add_subplot(gs1[1])
        for ax, title, ylabel in [
            (ax_t, "Torque (L/R)", "N·m"),
            (ax_r, "RPM (L/R)",    "rpm"),
        ]:
            ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)
        (self._line_tL,) = ax_t.plot([], [], "r-", lw=1.2, label="L")
        (self._line_tR,) = ax_t.plot([], [], "b-", lw=1.2, label="R")
        ax_t.legend(fontsize=7, loc="upper right")
        (self._line_rL,) = ax_r.plot([], [], "r-", lw=1.2, label="RPM_L")
        (self._line_rR,) = ax_r.plot([], [], "b-", lw=1.2, label="RPM_R")
        ax_r.legend(fontsize=7, loc="upper right")

        # ── Row 2: Power | Brake ───────────────────────────────────────
        gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.32)
        ax_p = fig.add_subplot(gs2[0]); ax_b = fig.add_subplot(gs2[1])
        ax_p.set_title("Mechanical Power", fontsize=9); ax_p.set_ylabel("W", fontsize=8)
        ax_p.set_xlabel("Time (s)", fontsize=8)
        ax_p.grid(True, alpha=0.3); ax_p.tick_params(labelsize=7)
        self._vt1_pline = ax_p.axhline(0, color="#008800", lw=1, ls="--", alpha=0)
        self._vt2_pline = ax_p.axhline(0, color="#006600", lw=1, ls="--", alpha=0)
        ax_b.set_title("Brake (voltage)", fontsize=9); ax_b.set_ylabel("V", fontsize=8)
        ax_b.set_xlabel("Time (s)", fontsize=8)
        ax_b.set_ylim(0, 10.5); ax_b.grid(True, alpha=0.3); ax_b.tick_params(labelsize=7)
        (self._line_pow,)  = ax_p.plot([], [], "k-",  lw=1.5, label="Total")
        (self._line_powL,) = ax_p.plot([], [], "r-",  lw=0.8, alpha=0.5, label="L")
        (self._line_powR,) = ax_p.plot([], [], "b-",  lw=0.8, alpha=0.5, label="R")
        ax_p.legend(fontsize=7, loc="upper right")
        (self._line_V,) = ax_b.plot([], [], color="#555555", lw=1.5)

        self._ax_t = ax_t; self._ax_r = ax_r
        self._ax_p = ax_p; self._ax_b = ax_b

        # ── Buttons ────────────────────────────────────────────────────
        ax_start_btn = fig.add_axes([0.22, 0.005, 0.12, 0.050])
        ax_pause_btn = fig.add_axes([0.37, 0.005, 0.12, 0.050])
        ax_stop_btn  = fig.add_axes([0.51, 0.005, 0.10, 0.050])

        self._btn_start = Button(ax_start_btn, "— START —",
                                  color="#cccccc", hovercolor="#99ff99")
        self._btn_pause = Button(ax_pause_btn, "PAUSE",
                                  color="#ffe0a0", hovercolor="#ffcc44")
        self._btn_stop  = Button(ax_stop_btn,  "STOP",
                                  color="#ff9999", hovercolor="#cc0000")

        self._btn_start.on_clicked(self._on_start)
        self._btn_pause.on_clicked(self._on_pause)
        self._btn_stop.on_clicked(self._on_stop)

        self.fig_exp = fig

    # ==================================================================
    # Build: Participant Display
    # ==================================================================
    def _build_participant_window(self):
        fig = plt.figure("Participant Display", figsize=(9, 7))
        fig.patch.set_facecolor("#0a0a1a")

        outer = gridspec.GridSpec(
            4, 1, figure=fig,
            height_ratios=[3, 2.2, 0.85, 1.2],
            hspace=0.55, top=0.94, bottom=0.05, left=0.10, right=0.95,
        )

        # ── Row 0: Speed chart ─────────────────────────────────────────
        ax_spd = fig.add_subplot(outer[0])
        ax_spd.set_facecolor("#111133")
        ax_spd.set_title("Your wheelchair Speed", fontsize=13, color="white", pad=5)
        ax_spd.set_ylabel("Speed (m/s)", fontsize=10, color="white")
        ax_spd.set_xlabel("Time (s)", fontsize=10, color="white")
        ax_spd.tick_params(colors="white", labelsize=9)
        for sp in ax_spd.spines.values():
            sp.set_color("#334466")
        ax_spd.grid(True, color="#223355", alpha=0.7)
        # Speed lines (warmup / sprint / gxt)
        (self._line_sL,) = ax_spd.plot([], [], color="#00ff88", lw=1.5, label="Speed L")
        (self._line_sR,) = ax_spd.plot([], [], color="#33dd33", lw=1.5, label="Speed R")
        self._spd_target_line = ax_spd.axhline(0, color="#ffff00", lw=1.5,
                                                 ls="--", alpha=0.0)
        # Torque lines (isometric)
        (self._line_tqL_top,) = ax_spd.plot([], [], color="#ff5555", lw=1.8,
                                              label="Torque L", visible=False)
        (self._line_tqR_top,) = ax_spd.plot([], [], color="#5599ff", lw=1.8,
                                              label="Torque R", visible=False)
        # Power line (wingate)
        (self._line_pow_top,) = ax_spd.plot([], [], color="#ffaa00", lw=2.2,
                                              label="Power", visible=False)
        self._ax_spd_legend = ax_spd.legend(
            fontsize=9, facecolor="#111133", labelcolor="white",
            edgecolor="#334466", loc="upper right",
        )
        self._ax_spd = ax_spd

        # ── Row 1: Feedback panel ──────────────────────────────────────
        ax_fb = fig.add_subplot(outer[1])
        ax_fb.set_facecolor("#0a0a1a"); ax_fb.axis("off")
        ax_fb.set_xlim(0, 1); ax_fb.set_ylim(0, 1)

        # General instruction text (warmup / sprint)
        self._txt_instr = ax_fb.text(
            0.5, 0.93, "", fontsize=14, color="white",
            ha="center", va="top", fontweight="bold",
        )

        # ── ANNOUNCE panel ─────────────────────────────────────────────
        self._txt_ann_name = ax_fb.text(
            0.5, 0.68, "", fontsize=34, color="#00ffcc",
            ha="center", va="center", fontweight="bold", visible=False,
        )
        self._txt_ann_sub = ax_fb.text(
            0.5, 0.42, "", fontsize=16, color="#aaaaff",
            ha="center", va="center", visible=False,
        )
        self._txt_ann_wait = ax_fb.text(
            0.5, 0.14, "Waiting for experimenter to press  START  ►",
            fontsize=10, color="#666688",
            ha="center", va="center", style="italic", visible=False,
        )

        # ── REST panel ─────────────────────────────────────────────────
        self._txt_rest_title = ax_fb.text(
            0.5, 0.80, "REST", fontsize=40, color="#88aaff",
            ha="center", va="center", fontweight="bold", visible=False,
        )
        self._txt_rest_timer = ax_fb.text(
            0.5, 0.52, "0 s", fontsize=28, color="#aaccff",
            ha="center", va="center", fontweight="bold", visible=False,
        )
        self._rest_prog_bg = Rectangle(
            (0.08, 0.28), 0.84, 0.12,
            facecolor="#1a1a3a", edgecolor="#445566", lw=1.2, visible=False,
        )
        ax_fb.add_patch(self._rest_prog_bg)
        self._rest_prog_fill = Rectangle(
            (0.08, 0.28), 0.84, 0.12,
            facecolor="#4488ff", edgecolor="none", visible=False,
        )
        ax_fb.add_patch(self._rest_prog_fill)
        self._txt_rest_next = ax_fb.text(
            0.5, 0.11, "", fontsize=11, color="#888888",
            ha="center", va="center", visible=False,
        )

        # ── VT band panel ──────────────────────────────────────────────
        self._vt_bg = Rectangle(
            (0, 0.28), 1.0, 0.30,
            facecolor="#1a1a3a", edgecolor="#444466", visible=False,
        )
        ax_fb.add_patch(self._vt_bg)
        self._vt_patch_part = Rectangle(
            (0.3, 0.28), 0.2, 0.30,
            facecolor="#00cc44", edgecolor="none", alpha=0.85, visible=False,
        )
        ax_fb.add_patch(self._vt_patch_part)
        self._txt_vt1_part = ax_fb.text(
            0.3, 0.60, "VT1", fontsize=8.5, ha="center",
            va="bottom", color="#aaffaa", visible=False,
        )
        self._txt_vt2_part = ax_fb.text(
            0.5, 0.60, "VT2", fontsize=8.5, ha="center",
            va="bottom", color="#aaffaa", visible=False,
        )
        (self._marker_vt,) = ax_fb.plot(
            [0.02], [0.43], "w^", ms=20, zorder=5, visible=False,
        )

        # ── Isometric trial bar chart ──────────────────────────────────
        _BAR_X   = [0.08, 0.40, 0.72]
        _BAR_W   = 0.22
        _BAR_BOT = 0.08
        _BAR_H   = 0.65

        self._iso_bar_bg    = []
        self._iso_bar_fill  = []
        self._iso_peak_line = []
        self._iso_lbl       = []
        self._iso_val       = []

        for i in range(_N_ISO_BARS):
            x = _BAR_X[i]
            bg = Rectangle((x, _BAR_BOT), _BAR_W, _BAR_H,
                            facecolor="#1a1a3a", edgecolor="#445566",
                            lw=1.2, visible=False)
            ax_fb.add_patch(bg)
            self._iso_bar_bg.append(bg)

            fill = Rectangle((x, _BAR_BOT), _BAR_W, 0.0,
                              facecolor="#4488ff", edgecolor="none", visible=False)
            ax_fb.add_patch(fill)
            self._iso_bar_fill.append(fill)

            (pk,) = ax_fb.plot(
                [x, x + _BAR_W], [_BAR_BOT, _BAR_BOT],
                color="#ffdd00", lw=2.5, ls="--", visible=False, zorder=6,
            )
            self._iso_peak_line.append(pk)

            vt = ax_fb.text(
                x + _BAR_W / 2, _BAR_BOT + _BAR_H + 0.02,
                "", fontsize=10, ha="center", va="bottom",
                color="white", fontweight="bold", visible=False,
            )
            self._iso_val.append(vt)

            lt = ax_fb.text(
                x + _BAR_W / 2, _BAR_BOT - 0.05,
                f"Trial {i+1}", fontsize=9, ha="center", va="top",
                color="#aaaaaa", visible=False,
            )
            self._iso_lbl.append(lt)

        self._iso_bar_x   = _BAR_X
        self._iso_bar_w   = _BAR_W
        self._iso_bar_bot = _BAR_BOT
        self._iso_bar_h   = _BAR_H

        # ── Wingate power bar ──────────────────────────────────────────
        self._wg_bg = Rectangle(
            (0.05, 0.40), 0.90, 0.18,
            facecolor="#1a1a3a", edgecolor="#556688", visible=False,
        )
        ax_fb.add_patch(self._wg_bg)
        self._wg_fill = Rectangle(
            (0.05, 0.40), 0.0, 0.18,
            facecolor="#00ff88", edgecolor="none", visible=False,
        )
        ax_fb.add_patch(self._wg_fill)
        self._txt_wg_power = ax_fb.text(
            0.50, 0.22, "0 W", fontsize=16, color="#00ff88",
            ha="center", va="bottom", fontweight="bold", visible=False,
        )
        self._txt_wg_pct = ax_fb.text(
            0.50, 0.09, "-- % of peak", fontsize=11, color="#aaffaa",
            ha="center", va="bottom", visible=False,
        )

        # ── GXT step label ─────────────────────────────────────────────
        self._txt_gxt_step = ax_fb.text(
            0.50, 0.50, "", fontsize=22, color="#00ccff",
            ha="center", va="center", fontweight="bold", visible=False,
        )
        self._gxt_lo_line = ax_spd.axhline(0, color="#00cc44", lw=1.5, ls="--", alpha=0.0)
        self._gxt_hi_line = ax_spd.axhline(0, color="#cc4400", lw=1.5, ls="--", alpha=0.0)

        # ── Countdown overlay ──────────────────────────────────────────
        self._txt_cd = ax_fb.text(
            0.5, 0.50, "", fontsize=110,
            color="#ffff00", ha="center", va="center",
            fontweight="bold", alpha=0.96, zorder=20,
        )

        # ── Row 2: Stage progress bar ──────────────────────────────────
        ax_sg = fig.add_subplot(outer[2])
        ax_sg.set_facecolor("#0a0a1a"); ax_sg.axis("off")
        ax_sg.set_xlim(0, 1); ax_sg.set_ylim(0, 1)
        ax_sg.add_patch(Rectangle((0, 0.15), 1.0, 0.70,
                                   facecolor="#1a1a3a", edgecolor="#334466", lw=1))
        self._stage_fill = Rectangle((0, 0.15), 0.0, 0.70,
                                      facecolor="#00cc88", edgecolor="none")
        ax_sg.add_patch(self._stage_fill)
        self._txt_stage = ax_sg.text(
            0.5, 0.50, "Stage 0 / 0", fontsize=10.5, color="#00cc88",
            ha="center", va="center", fontweight="bold",
        )

        # ── Row 3: Speed | Stage time ──────────────────────────────────
        ax_info = fig.add_subplot(outer[3])
        ax_info.set_facecolor("#0a0a1a"); ax_info.axis("off")
        ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
        ax_info.axvline(0.50, color="#223355", lw=1.2)
        self._txt_spd_val = ax_info.text(
            0.25, 0.70, "0.0 m/s", fontsize=22, color="#ff44bb",
            ha="center", va="center", fontweight="bold",
        )
        ax_info.text(0.25, 0.16, "Speed", fontsize=9, color="#888888",
                     ha="center", va="center")
        self._txt_time_val = ax_info.text(
            0.75, 0.70, "00:00 / 00:00", fontsize=20, color="#ffaa00",
            ha="center", va="center", fontweight="bold",
        )
        ax_info.text(0.75, 0.16, "Stage Time", fontsize=9, color="#888888",
                     ha="center", va="center")

        self._ax_fb   = ax_fb
        self.fig_part = fig

    # ==================================================================
    # Button callbacks (main thread)
    # ==================================================================
    def _on_start(self, _):
        with self._lock:
            if self._waiting_for_start:
                self._waiting_for_start = False

    def _on_pause(self, _):
        if self._paused:
            self._paused = False
            if self._pause_start_t is not None:
                with self._lock:
                    self._rest_s    += time.perf_counter() - self._pause_start_t
                    self._rest_count += 1
                    self._pause_start_t = None
            self._btn_pause.label.set_text("PAUSE")
            self._btn_pause.ax.set_facecolor("#ffe0a0")
        else:
            self._paused = True
            with self._lock:
                self._pause_start_t = time.perf_counter()
            self._btn_pause.label.set_text("RESUME")
            self._btn_pause.ax.set_facecolor("#aaffaa")

    def _on_stop(self, _):
        self.stop()

    # ==================================================================
    # Timer callback — main thread, 25 fps
    # ==================================================================
    def _update(self):
        if not self._running:
            return

        now = time.perf_counter()

        # Drain sample queue
        updated = False
        while True:
            try:
                s = self._q.get_nowait()
            except queue.Empty:
                break
            if s is None:
                self.stop(); return

            self._t.append(s.get("time", 0.0))
            self._torque_L.append(s.get("torque_L", 0.0))
            self._torque_R.append(s.get("torque_R", 0.0))
            self._rpm_L.append(s.get("rpm_L", 0.0))
            self._rpm_R.append(s.get("rpm_R", 0.0))
            self._power_L.append(s.get("power_L", 0.0))
            self._power_R.append(s.get("power_R", 0.0))
            self._voltage.append(s.get("voltage", 0.0))

            rL = s.get("rpm_L", 0.0); rR = s.get("rpm_R", 0.0)
            sL = rL * (2 * math.pi / 60) * (self._roller_r / self._wheel_r) * self._rim_r
            sR = rR * (2 * math.pi / 60) * (self._roller_r / self._wheel_r) * self._rim_r
            self._speed_L.append(sL); self._speed_R.append(sR)

            self._latest_power = s.get("power_L", 0.0) + s.get("power_R", 0.0)
            self._latest_speed = (sL + sR) / 2
            tL = s.get("torque_L", 0.0); tR = s.get("torque_R", 0.0)
            self._latest_nm    = (tL + tR) * 9.8 * 0.01

            with self._lock:
                dm = self._display_mode
            if dm == "wingate" and self._latest_power > self._wingate_peak:
                self._wingate_peak = self._latest_power
            if dm == "force" and self._latest_nm > self._iso_current_max:
                self._iso_current_max = self._latest_nm
            updated = True

        # START button appearance — glows green when waiting
        with self._lock:
            wfs = self._waiting_for_start
        if wfs:
            self._btn_start.label.set_text("START  ►")
            self._btn_start.ax.set_facecolor("#00ee44")
        else:
            self._btn_start.label.set_text("— running —")
            self._btn_start.ax.set_facecolor("#cccccc")

        # Session status labels
        elapsed   = now - self._session_start_t if self._session_start_t else 0
        cur_pause = (now - self._pause_start_t
                     if (self._paused and self._pause_start_t) else 0)
        active_s  = max(0.0, elapsed - self._rest_s - cur_pause)
        a_mm, a_ss = divmod(int(active_s), 60)
        t_mm, t_ss = divmod(int(self._total_active_s), 60)
        r_mm, r_ss = divmod(int(self._rest_s + cur_pause), 60)
        self._txt_mode.set_text(f"Mode: {self._mode}")
        self._txt_active.set_text(
            f"Active: {a_mm:02d}:{a_ss:02d} / {t_mm:02d}:{t_ss:02d}"
        )
        self._txt_rest_s.set_text(
            f"Rest: {r_mm:02d}:{r_ss:02d} ({self._rest_count} breaks)"
        )

        # Power feedback bar
        with self._lock:
            vt1 = self._vt1; vt2 = self._vt2
        pmax = max(300.0, (vt2 * 1.5) if vt2 else 300.0)
        p_frac = min(1.0, max(0.0, self._latest_power / pmax))
        self._marker_exp.set_xdata([p_frac])
        self._txt_pval_exp.set_text(f"{self._latest_power:.0f} W")
        self._txt_pval_exp.set_position((p_frac, 0.08))
        if vt1 and vt2:
            lo_f = vt1 / pmax; hi_f = vt2 / pmax
            self._vt_patch_exp.set_x(lo_f)
            self._vt_patch_exp.set_width(hi_f - lo_f)
            self._vt_patch_exp.set_alpha(0.7)
            self._vt1_line_exp.set_xdata([lo_f, lo_f])
            self._vt1_line_exp.set_alpha(0.7)
            self._vt2_line_exp.set_xdata([hi_f, hi_f])
            self._vt2_line_exp.set_alpha(0.7)
            self._txt_vt1_exp.set_position((lo_f, 0.65))
            self._txt_vt1_exp.set_alpha(1.0)
            self._txt_vt2_exp.set_position((hi_f, 0.65))
            self._txt_vt2_exp.set_alpha(1.0)
            self._vt1_pline.set_ydata([vt1, vt1])
            self._vt1_pline.set_alpha(0.65)
            self._vt2_pline.set_ydata([vt2, vt2])
            self._vt2_pline.set_alpha(0.65)

        # Scrolling charts
        if updated:
            t = list(self._t)
            self._line_tL.set_data(t, list(self._torque_L))
            self._line_tR.set_data(t, list(self._torque_R))
            self._ax_t.relim(); self._ax_t.autoscale_view()
            self._line_rL.set_data(t, list(self._rpm_L))
            self._line_rR.set_data(t, list(self._rpm_R))
            self._ax_r.relim(); self._ax_r.autoscale_view()
            pt = [L + R for L, R in zip(list(self._power_L), list(self._power_R))]
            self._line_pow.set_data(t, pt)
            self._line_powL.set_data(t, list(self._power_L))
            self._line_powR.set_data(t, list(self._power_R))
            self._ax_p.relim(); self._ax_p.autoscale_view()
            self._line_V.set_data(t, list(self._voltage))
            self._line_sL.set_data(t, list(self._speed_L))
            self._line_sR.set_data(t, list(self._speed_R))
            self._line_tqL_top.set_data(t, list(self._torque_L))
            self._line_tqR_top.set_data(t, list(self._torque_R))
            self._line_pow_top.set_data(t, pt)
            self._ax_spd.relim(); self._ax_spd.autoscale_view()

        self._update_participant(pmax, vt1, vt2, now)

        try:
            self.fig_exp.canvas.draw_idle()
            self.fig_part.canvas.draw_idle()
        except Exception:
            pass

    # ==================================================================
    # Participant display dispatch
    # ==================================================================
    def _update_participant(self, pmax, vt1, vt2, now):
        with self._lock:
            dm          = self._display_mode
            cdv         = self._countdown_val
            instr       = self._instruction
            sname       = self._stage_name
            snum        = self._stage_num
            stotal      = self._stage_total
            sdur        = self._stage_dur_s
            sstart      = self._stage_start_t
            tgt_spd     = self._target_speed
            gxt_lo      = self._gxt_target_lo
            gxt_hi      = self._gxt_target_hi
            wg_peak     = self._wingate_peak
            iso_res     = list(self._iso_results)
            iso_ntrials = self._iso_n_trials
            iso_curmax  = self._iso_current_max
            iso_scale   = self._iso_max_scale
            ann_name    = self._announce_name
            ann_sub     = self._announce_sub
            rest_dur    = self._rest_dur_s
            rest_start  = self._rest_start_t_disp
            rest_next   = self._rest_next_stage

        # Countdown overlay takes full priority
        if cdv > 0:
            self._txt_cd.set_text(str(cdv))
            self._txt_instr.set_alpha(0.0)
            self._hide_all_panels()
            return
        else:
            self._txt_cd.set_text("")
            self._txt_instr.set_alpha(1.0)

        self._txt_instr.set_text("" if dm in ("announce", "rest") else instr)
        self._hide_all_panels()

        # Switch top chart content (only on mode change; skip transient states)
        if dm not in ("announce", "rest"):
            self._switch_top_chart(dm)

        stage_elapsed = (now - sstart) if sstart else 0

        if dm == "announce":
            self._panel_announce(ann_name, ann_sub)
        elif dm == "rest":
            rest_elapsed = (now - rest_start) if rest_start else 0
            self._panel_rest(rest_dur, rest_elapsed, rest_next)
        elif dm == "vt_band":
            self._panel_vt_band(vt1, vt2, pmax)
        elif dm == "force":
            self._panel_iso_bars(snum, iso_ntrials, iso_res, iso_curmax, iso_scale)
        elif dm == "wingate":
            self._panel_wingate(wg_peak)
        elif dm == "gxt":
            self._panel_gxt(gxt_lo, gxt_hi, snum, stotal, sdur, stage_elapsed)
        # "warmup" / "sprint": speed chart + instruction text is enough

        # Target speed line on speed chart
        if tgt_spd and dm in ("gxt", "warmup"):
            self._spd_target_line.set_ydata([tgt_spd, tgt_spd])
            self._spd_target_line.set_alpha(0.7)
        else:
            self._spd_target_line.set_alpha(0.0)

        # Stage progress bar + label
        frac = min(1.0, stage_elapsed / sdur) if sdur > 0 else 0
        self._stage_fill.set_width(frac)
        if dm == "rest":
            self._txt_stage.set_text("REST")
            self._txt_stage.set_color("#88aaff")
            self._stage_fill.set_facecolor("#4488ff")
        elif dm == "announce":
            self._txt_stage.set_text(ann_name)
            self._txt_stage.set_color("#00ffcc")
            self._stage_fill.set_facecolor("#00ffcc")
        else:
            lbl = f"Stage {snum} / {stotal}" if stotal > 0 else sname
            self._txt_stage.set_text(lbl)
            self._txt_stage.set_color("#00cc88")
            self._stage_fill.set_facecolor("#00cc88")

        # Speed + time labels (bottom row)
        self._txt_spd_val.set_text(f"{self._latest_speed:.1f} m/s")
        if dm == "rest" and rest_start:
            rem = max(0.0, rest_dur - (now - rest_start))
        else:
            rem = max(0.0, sdur - stage_elapsed)
        r_mm, r_ss = divmod(int(rem), 60)
        d_mm, d_ss = divmod(int(sdur), 60)
        self._txt_time_val.set_text(f"{r_mm:02d}:{r_ss:02d} / {d_mm:02d}:{d_ss:02d}")

    # ------------------------------------------------------------------
    # Top chart mode switch (speed / torque / power)
    # ------------------------------------------------------------------
    def _switch_top_chart(self, dm: str):
        # Map display_mode to top chart type
        if dm == "force":
            mode = "torque"
        elif dm == "wingate":
            mode = "power"
        else:
            mode = "speed"

        if mode == self._top_chart_mode:
            return   # no change needed
        self._top_chart_mode = mode

        spd_vis = (mode == "speed")
        tq_vis  = (mode == "torque")
        pw_vis  = (mode == "power")

        self._line_sL.set_visible(spd_vis)
        self._line_sR.set_visible(spd_vis)
        self._line_tqL_top.set_visible(tq_vis)
        self._line_tqR_top.set_visible(tq_vis)
        self._line_pow_top.set_visible(pw_vis)
        self._spd_target_line.set_alpha(0.0)

        if mode == "torque":
            self._ax_spd.set_title("Torque   (L = red,  R = blue)",
                                    fontsize=12, color="white", pad=5)
            self._ax_spd.set_ylabel("N·m", fontsize=10, color="white")
        elif mode == "power":
            self._ax_spd.set_title("Power Output", fontsize=13, color="white", pad=5)
            self._ax_spd.set_ylabel("W", fontsize=10, color="white")
        else:
            self._ax_spd.set_title("Your wheelchair Speed",
                                    fontsize=13, color="white", pad=5)
            self._ax_spd.set_ylabel("Speed (m/s)", fontsize=10, color="white")

        # Rebuild legend from currently visible named lines
        visible_lines = [l for l in (
            self._line_sL, self._line_sR,
            self._line_tqL_top, self._line_tqR_top,
            self._line_pow_top,
        ) if l.get_visible()]
        self._ax_spd.legend(
            handles=visible_lines,
            fontsize=9, facecolor="#111133", labelcolor="white",
            edgecolor="#334466", loc="upper right",
        )

    # ------------------------------------------------------------------
    # Hide all panels
    # ------------------------------------------------------------------
    def _hide_all_panels(self):
        for obj in [
            self._txt_ann_name, self._txt_ann_sub, self._txt_ann_wait,
            self._txt_rest_title, self._txt_rest_timer,
            self._rest_prog_bg, self._rest_prog_fill, self._txt_rest_next,
            self._vt_bg, self._vt_patch_part,
            self._txt_vt1_part, self._txt_vt2_part, self._marker_vt,
            self._wg_bg, self._wg_fill, self._txt_wg_power, self._txt_wg_pct,
            self._txt_gxt_step,
        ]:
            obj.set_visible(False)
        for i in range(_N_ISO_BARS):
            self._iso_bar_bg[i].set_visible(False)
            self._iso_bar_fill[i].set_visible(False)
            self._iso_peak_line[i].set_visible(False)
            self._iso_val[i].set_visible(False)
            self._iso_lbl[i].set_visible(False)
        self._gxt_lo_line.set_alpha(0.0)
        self._gxt_hi_line.set_alpha(0.0)

    # ------------------------------------------------------------------
    # Panel: Announce
    # ------------------------------------------------------------------
    def _panel_announce(self, name: str, sub: str):
        self._txt_ann_name.set_text(name)
        self._txt_ann_name.set_visible(True)
        if sub:
            self._txt_ann_sub.set_text(sub)
            self._txt_ann_sub.set_visible(True)
        self._txt_ann_wait.set_visible(True)

    # ------------------------------------------------------------------
    # Panel: Rest
    # ------------------------------------------------------------------
    def _panel_rest(self, dur: float, elapsed: float, next_stage: str):
        # "REST" label — upper-center
        self._txt_rest_title.set_position((0.5, 0.68))
        self._txt_rest_title.set_visible(True)

        # "Next: ..." — prominent, lower-center
        if next_stage:
            self._txt_rest_next.set_text(f"Next:  {next_stage}")
            self._txt_rest_next.set_fontsize(20)
            self._txt_rest_next.set_position((0.5, 0.32))
            self._txt_rest_next.set_color("#aaccff")
            self._txt_rest_next.set_visible(True)

        # Timer and gauge bar are intentionally hidden —
        # the stage progress bar + time label in the bottom rows already cover this.

    # ------------------------------------------------------------------
    # Panel: VT band
    # ------------------------------------------------------------------
    def _panel_vt_band(self, vt1, vt2, pmax):
        self._vt_bg.set_visible(True)
        self._vt_patch_part.set_visible(True)
        self._marker_vt.set_visible(True)
        p_frac = min(1.0, max(0.0, self._latest_power / pmax))
        self._marker_vt.set_xdata([p_frac])
        if vt1 and vt2:
            lo_f = vt1 / pmax; hi_f = vt2 / pmax
            self._vt_patch_part.set_x(lo_f)
            self._vt_patch_part.set_width(hi_f - lo_f)
            self._txt_vt1_part.set_position((lo_f, 0.60))
            self._txt_vt1_part.set_visible(True)
            self._txt_vt2_part.set_position((hi_f, 0.60))
            self._txt_vt2_part.set_visible(True)

    # ------------------------------------------------------------------
    # Panel: Isometric bars
    # ------------------------------------------------------------------
    def _panel_iso_bars(self, current_trial, n_trials, iso_results, cur_max, scale):
        if scale < 1.0:
            scale = 50.0
        all_vals = list(iso_results) + [cur_max]
        max_val  = max(all_vals) if all_vals else 0
        if max_val > scale * 0.85:
            scale = max_val * 1.25

        hmax = self._iso_bar_h
        bot  = self._iso_bar_bot
        bw   = self._iso_bar_w

        for i in range(_N_ISO_BARS):
            show = (i < n_trials)
            self._iso_bar_bg[i].set_visible(show)
            self._iso_lbl[i].set_visible(show)
            if not show:
                continue
            x = self._iso_bar_x[i]
            trial_num = i + 1

            if trial_num < current_trial:
                val = iso_results[i] if i < len(iso_results) else 0.0
                h   = max(0.0, min((val / scale) * hmax if scale > 0 else 0, hmax))
                self._iso_bar_fill[i].set_height(h)
                self._iso_bar_fill[i].set_facecolor("#4488ff")
                self._iso_bar_fill[i].set_visible(True)
                self._iso_val[i].set_text(f"{val:.1f} N·m")
                self._iso_val[i].set_position((x + bw / 2, bot + h + 0.02))
                self._iso_val[i].set_color("#88bbff")
                self._iso_val[i].set_visible(True)
                self._iso_peak_line[i].set_visible(False)

            elif trial_num == current_trial:
                live_h = max(0.0, min((self._latest_nm / scale) * hmax, hmax))
                peak_h = max(0.0, min((cur_max / scale) * hmax, hmax))
                self._iso_bar_fill[i].set_height(live_h)
                self._iso_bar_fill[i].set_facecolor("#00ff88")
                self._iso_bar_fill[i].set_visible(True)
                peak_y = bot + peak_h
                self._iso_peak_line[i].set_ydata([peak_y, peak_y])
                self._iso_peak_line[i].set_xdata([x, x + bw])
                self._iso_peak_line[i].set_visible(True)
                self._iso_val[i].set_text(f"Peak: {cur_max:.1f} N·m")
                self._iso_val[i].set_position((x + bw / 2, bot + peak_h + 0.02))
                self._iso_val[i].set_color("#ffdd00")
                self._iso_val[i].set_visible(True)

            else:
                self._iso_bar_fill[i].set_height(0.0)
                self._iso_bar_fill[i].set_visible(False)
                self._iso_peak_line[i].set_visible(False)
                self._iso_val[i].set_text("")
                self._iso_val[i].set_visible(False)

    # ------------------------------------------------------------------
    # Panel: Wingate
    # ------------------------------------------------------------------
    def _panel_wingate(self, peak_power):
        self._wg_bg.set_visible(True)
        self._wg_fill.set_visible(True)
        self._txt_wg_power.set_visible(True)
        self._txt_wg_pct.set_visible(True)
        p = max(0.0, self._latest_power)
        if peak_power < 1.0:
            peak_power = max(1.0, p)
        pct = min(1.0, p / peak_power)
        self._wg_fill.set_width(pct * 0.90)
        c = "#00ff88" if pct >= 0.90 else ("#ffcc00" if pct >= 0.70 else "#ff4444")
        self._wg_fill.set_facecolor(c)
        self._txt_wg_power.set_text(f"{p:.0f} W")
        self._txt_wg_power.set_color(c)
        self._txt_wg_pct.set_text(
            f"{pct*100:.0f} % of peak  ({peak_power:.0f} W peak)"
        )

    # ------------------------------------------------------------------
    # Panel: GXT
    # ------------------------------------------------------------------
    def _panel_gxt(self, lo, hi, snum, stotal, sdur, stage_elapsed):
        self._txt_gxt_step.set_visible(True)
        rem = max(0.0, sdur - stage_elapsed)
        r_mm, r_ss = divmod(int(rem), 60)
        self._txt_gxt_step.set_text(
            f"Step {snum} / {stotal}\n{r_mm:02d}:{r_ss:02d} remaining"
        )
        if lo:
            self._gxt_lo_line.set_ydata([lo, lo])
            self._gxt_lo_line.set_alpha(0.75)
        if hi:
            self._gxt_hi_line.set_ydata([hi, hi])
            self._gxt_hi_line.set_alpha(0.75)


# ------------------------------------------------------------------
def _default_instruction(stage_name: str) -> str:
    m = {
        "Warmup":    "Comfortably Propulsion!",
        "Isometric": "Push as hard as you can!",
        "Sprint":    "Sprint at full speed!",
        "Wingate":   "Maintain your power output!",
    }
    if stage_name.startswith("GXT"):
        return "Maintain your pace!"
    return m.get(stage_name, stage_name)
