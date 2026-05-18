from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ImuConfig


@dataclass(frozen=True)
class ImuMeasurement:
    timestamp_sec: float
    gyro_rad_s: np.ndarray
    accel_m_s2: np.ndarray


class CsvImuSource:
    def __init__(self, config: ImuConfig) -> None:
        self.config = config

    def load(self) -> list[ImuMeasurement]:
        if self.config.csv_path is None:
            return []
        path = Path(self.config.csv_path)
        if not path.exists():
            raise FileNotFoundError(path)

        measurements: list[ImuMeasurement] = []
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                measurements.append(
                    ImuMeasurement(
                        timestamp_sec=float(row[self.config.timestamp_column]),
                        gyro_rad_s=np.array(
                            [float(row[column]) for column in self.config.gyro_columns],
                            dtype=np.float64,
                        ),
                        accel_m_s2=np.array(
                            [float(row[column]) for column in self.config.accel_columns],
                            dtype=np.float64,
                        ),
                    )
                )
        return measurements
