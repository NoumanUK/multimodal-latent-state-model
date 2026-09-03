from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


EXPECTED_MODALITIES = (
    "A",
    "B",
    "C",
)

DEFAULT_HISTORY_LENGTH = 20

DEFAULT_FORECAST_HORIZONS = (
    1,
    5,
    10,
)


class LorenzMultimodalDataset(Dataset):
    """
    Lazy temporal dataset for multimodal Lorenz-96 data.

    No temporal windows are materialized in advance.

    Each sample contains:

        A observations:
            (history_length, 20)

        B observations:
            (history_length, 20)

        C observations:
            (history_length, 20)

        A/B/C binary masks:
            (history_length, 20)

        current_truth:
            (40,)

        future_truth:
            (num_horizons, 40)

        trajectory_index:
            scalar

        time_index:
            scalar

    For current time t, input history is:

        t-history_length+1 ... t

    Forecast targets are:

        t+h

    for every requested forecast horizon h.
    """

    def __init__(
        self,
        modalities: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        truth: np.ndarray,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        forecast_horizons: tuple[int, ...] = (
            DEFAULT_FORECAST_HORIZONS
        ),
    ) -> None:
        super().__init__()

        self.history_length = int(
            history_length
        )

        self.forecast_horizons = tuple(
            int(h)
            for h in forecast_horizons
        )

        self._validate_configuration()

        self.truth = np.asarray(
            truth,
            dtype=np.float32,
        )

        self.modalities = {
            name: np.asarray(
                modalities[name],
                dtype=np.float32,
            )
            for name in EXPECTED_MODALITIES
        }

        self.masks = {
            name: np.asarray(
                masks[name],
                dtype=np.float32,
            )
            for name in EXPECTED_MODALITIES
        }

        self._validate_data()

        self.num_trajectories = (
            self.truth.shape[0]
        )

        self.num_time_steps = (
            self.truth.shape[1]
        )

        self.max_horizon = max(
            self.forecast_horizons
        )

        self.windows_per_trajectory = (
            self.num_time_steps
            - self.history_length
            - self.max_horizon
            + 1
        )

        if self.windows_per_trajectory <= 0:
            raise ValueError(
                "Not enough time steps for requested "
                "history length and forecast horizons."
            )

        self.total_samples = (
            self.num_trajectories
            * self.windows_per_trajectory
        )

    def _validate_configuration(
        self,
    ) -> None:
        if self.history_length <= 0:
            raise ValueError(
                "history_length must be positive."
            )

        if not self.forecast_horizons:
            raise ValueError(
                "forecast_horizons cannot be empty."
            )

        if any(
            horizon <= 0
            for horizon in self.forecast_horizons
        ):
            raise ValueError(
                "All forecast horizons must be positive."
            )

        if (
            len(set(self.forecast_horizons))
            != len(self.forecast_horizons)
        ):
            raise ValueError(
                "forecast_horizons contains duplicates."
            )

    def _validate_data(
        self,
    ) -> None:
        if self.truth.ndim != 3:
            raise ValueError(
                "truth must have shape "
                "(trajectories, time, 40)."
            )

        if self.truth.shape[-1] != 40:
            raise ValueError(
                f"Truth must contain 40 state "
                f"variables, received "
                f"{self.truth.shape[-1]}."
            )

        if not np.all(
            np.isfinite(self.truth)
        ):
            raise ValueError(
                "truth contains NaN or Inf."
            )

        for name in EXPECTED_MODALITIES:
            values = self.modalities[name]
            mask = self.masks[name]

            if values.ndim != 3:
                raise ValueError(
                    f"Modality {name} must have shape "
                    f"(trajectories, time, 20)."
                )

            if values.shape[-1] != 20:
                raise ValueError(
                    f"Modality {name} must contain "
                    f"20 features."
                )

            if values.shape[:2] != (
                self.truth.shape[:2]
            ):
                raise ValueError(
                    f"Temporal dimensions of modality "
                    f"{name} do not match truth."
                )

            if values.shape != mask.shape:
                raise ValueError(
                    f"Values and mask shape mismatch "
                    f"for modality {name}."
                )

            if not np.all(
                np.isfinite(values)
            ):
                raise ValueError(
                    f"Modality {name} contains "
                    f"NaN or Inf."
                )

            valid_mask = np.logical_or(
                mask == 0.0,
                mask == 1.0,
            )

            if not np.all(valid_mask):
                raise ValueError(
                    f"Mask for modality {name} "
                    f"must contain only 0 and 1."
                )

    def __len__(
        self,
    ) -> int:
        return self.total_samples

    def _decode_index(
        self,
        index: int,
    ) -> tuple[int, int]:
        """
        Convert global sample index into:

            trajectory_index
            current_time
        """
        if index < 0:
            index += self.total_samples

        if (
            index < 0
            or index >= self.total_samples
        ):
            raise IndexError(
                f"Dataset index {index} out of range."
            )

        trajectory_index = (
            index
            // self.windows_per_trajectory
        )

        local_window_index = (
            index
            % self.windows_per_trajectory
        )

        current_time = (
            self.history_length
            - 1
            + local_window_index
        )

        return (
            trajectory_index,
            current_time,
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        trajectory_index, current_time = (
            self._decode_index(index)
        )

        start_time = (
            current_time
            - self.history_length
            + 1
        )

        end_time = (
            current_time + 1
        )

        modality_windows = {}

        mask_windows = {}

        for name in EXPECTED_MODALITIES:
            modality_window = (
                self.modalities[name][
                    trajectory_index,
                    start_time:end_time,
                    :
                ]
            )

            mask_window = (
                self.masks[name][
                    trajectory_index,
                    start_time:end_time,
                    :
                ]
            )

            modality_windows[name] = (
                torch.from_numpy(
                    modality_window.copy()
                )
            )

            mask_windows[name] = (
                torch.from_numpy(
                    mask_window.copy()
                )
            )

        current_truth = torch.from_numpy(
            self.truth[
                trajectory_index,
                current_time,
                :
            ].copy()
        )

        future_states = np.stack(
            [
                self.truth[
                    trajectory_index,
                    current_time + horizon,
                    :
                ]
                for horizon
                in self.forecast_horizons
            ],
            axis=0,
        )

        future_truth = torch.from_numpy(
            future_states.astype(
                np.float32,
                copy=False,
            )
        )

        return {
            "modalities": modality_windows,
            "masks": mask_windows,
            "current_truth": current_truth,
            "future_truth": future_truth,
            "trajectory_index": torch.tensor(
                trajectory_index,
                dtype=torch.long,
            ),
            "time_index": torch.tensor(
                current_time,
                dtype=torch.long,
            ),
        }