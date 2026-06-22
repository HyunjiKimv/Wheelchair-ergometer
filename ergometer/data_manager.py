"""
CSV data logger for ergometer sessions.

Records one row per sample, auto-creates output directory.
"""

from __future__ import annotations
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


COLUMNS = [
    "time",
    "remaining_s",
    "stage",
    "test_name",
    "status",
    "voltage",

    "torque_L",
    "rpm_L",
    "power_L",

    "torque_R",
    "rpm_R",
    "power_R",

    "total_power",
    "total_rpm",
]


class DataLogger:
    """Write calibrated samples to CSV, one row per call."""

    def __init__(
        self,
        output_dir: str = "results",
        participant_id: str = "unknown",
        stage: str = "",
    ):
        self._stage = stage
        self._fh: Optional[object] = None
        self._writer: Optional[csv.DictWriter] = None
        self._path: Optional[Path] = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{participant_id}_{stage}_{timestamp}.csv" if stage else f"{participant_id}_{timestamp}.csv"
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self._path = out_path / fname

        self._fh = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS, extrasaction="ignore")
        self._writer.writeheader()

    @property
    def path(self) -> Path:
        return self._path

    def write(self, sample: dict) -> None:
        if self._writer is None:
            return

        row = dict(sample)
        row.setdefault("stage", self._stage)

        self._writer.writerow(row)

    def close(self) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SessionSummary:
    """Collects per-stage result objects and exports a summary CSV."""

    def __init__(self, participant_id: str, output_dir: str = "results"):
        self._pid = participant_id
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict] = []

    def add(self, stage: str, result_obj) -> None:
        row = {"participant": self._pid, "stage": stage}
        # Flatten dataclass fields into row
        for k, v in result_obj.__dict__.items():
            if not isinstance(v, list):
                row[k] = v
        self._rows.append(row)

    def save(self) -> Path:
        if not self._rows:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._dir / f"{self._pid}_summary_{timestamp}.csv"
        keys = list(dict.fromkeys(k for r in self._rows for k in r))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._rows)
        print(f"Summary saved → {path}")
        return path
