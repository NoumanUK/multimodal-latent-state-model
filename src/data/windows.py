from __future__ import annotations

from typing import Mapping

import numpy as np


EXPECTED_MODALITIES = ("A", "B", "C")

HISTORY_LENGTH = 20

FORECAST_HORIZONS = (1, 5, 10)


def _validate_inputs(
    modalities: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    truth: np.ndarray,
) -> None:
    """
    Validate temporal dataset inputs.

    Expected:
        modality: (trajectories, time, 20)
        mask:     (trajectories, time, 20)
        truth:    (trajectories, time, 40)
    """
    truth = np.asarray(truth)

    if truth.ndim != 3:
        raise ValueError(
            "truth must have shape "
            "(trajectories, time, state_features)."
        )

    if truth.shape[-1] != 40:
        raise ValueError(
            f"Expected truth dimension 40, "
            f"received {truth.shape[-1]}."
        )

    if not np.all(np.isfinite(truth)):
        raise ValueError(
            "truth contains NaN or infinite values."
        )

    for name in EXPECTED_MODALITIES:
        if name not in modalities:
            raise ValueError(
                f"Missing modality {name}."
            )

        if name not in masks:
            raise ValueError(
                f"Missing mask for modality {name}."
            )

        values = np.asarray(
            modalities[name]
        )

        mask = np.asarray(
            masks[name]
        )

        if values.ndim != 3:
            raise ValueError(
                f"Modality {name} must have shape "
                f"(trajectories, time, features)."
            )

        if values.shape[-1] != 20:
            raise ValueError(
                f"Modality {name} must contain "
                f"20 features."
            )

        if values.shape != mask.shape:
            raise ValueError(
                f"Values/mask shape mismatch "
                f"for modality {name}."
            )

        if values.shape[:2] != truth.shape[:2]:
            raise ValueError(
                f"Temporal dimensions for modality "
                f"{name} do not match truth."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Modality {name} contains "
                f"NaN or infinite values."
            )

        valid_mask = np.logical_or(
            mask == 0,
            mask == 1,
        )

        if not np.all(valid_mask):
            raise ValueError(
                f"Mask {name} must contain "
                f"only 0 and 1."
            )


def count_windows(
    num_trajectories: int,
    num_time_steps: int,
    history_length: int = HISTORY_LENGTH,
    forecast_horizons: tuple[int, ...] = FORECAST_HORIZONS,
) -> int:
    """
    Calculate total number of valid temporal windows.
    """
    if num_trajectories <= 0:
        raise ValueError(
            "num_trajectories must be positive."
        )

    if num_time_steps <= 0:
        raise ValueError(
            "num_time_steps must be positive."
        )

    if history_length <= 0:
        raise ValueError(
            "history_length must be positive."
        )

    if not forecast_horizons:
        raise ValueError(
            "At least one forecast horizon is required."
        )

    if any(
        horizon <= 0
        for horizon in forecast_horizons
    ):
        raise ValueError(
            "Forecast horizons must be positive."
        )

    max_horizon = max(
        forecast_horizons
    )

    windows_per_trajectory = (
        num_time_steps
        - history_length
        - max_horizon
        + 1
    )

    if windows_per_trajectory <= 0:
        raise ValueError(
            "Trajectory is too short for requested "
            "history length and forecast horizons."
        )

    return (
        num_trajectories
        * windows_per_trajectory
    )


def build_temporal_windows(
    modalities: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    truth: np.ndarray,
    history_length: int = HISTORY_LENGTH,
    forecast_horizons: tuple[int, ...] = FORECAST_HORIZONS,
) -> dict[str, object]:
    """
    Build temporal windows.

    For a window ending at time t:

        input:
            observations from
            t-history_length+1 ... t

        current target:
            truth X_t

        forecast targets:
            X_(t+h)

        for every h in forecast_horizons.

    Returns:
        {
            "modalities": {
                "A": (samples, history, 20),
                "B": (samples, history, 20),
                "C": (samples, history, 20),
            },

            "masks": {
                "A": (samples, history, 20),
                ...
            },

            "current_truth":
                (samples, 40),

            "future_truth":
                (samples, num_horizons, 40),

            "trajectory_index":
                (samples,),

            "time_index":
                (samples,),

            "forecast_horizons":
                array([1, 5, 10]),
        }
    """
    _validate_inputs(
        modalities=modalities,
        masks=masks,
        truth=truth,
    )

    if history_length <= 0:
        raise ValueError(
            "history_length must be positive."
        )

    if not forecast_horizons:
        raise ValueError(
            "forecast_horizons cannot be empty."
        )

    if any(
        horizon <= 0
        for horizon in forecast_horizons
    ):
        raise ValueError(
            "All forecast horizons must be positive."
        )

    if len(set(forecast_horizons)) != len(
        forecast_horizons
    ):
        raise ValueError(
            "forecast_horizons contains duplicates."
        )

    truth = np.asarray(
        truth,
        dtype=np.float32,
    )

    num_trajectories = truth.shape[0]
    num_time_steps = truth.shape[1]

    max_horizon = max(
        forecast_horizons
    )

    windows_per_trajectory = (
        num_time_steps
        - history_length
        - max_horizon
        + 1
    )

    if windows_per_trajectory <= 0:
        raise ValueError(
            "Not enough time steps for the "
            "requested temporal configuration."
        )

    total_samples = (
        num_trajectories
        * windows_per_trajectory
    )

    modality_windows = {
        name: np.empty(
            (
                total_samples,
                history_length,
                20,
            ),
            dtype=np.float32,
        )
        for name in EXPECTED_MODALITIES
    }

    mask_windows = {
        name: np.empty(
            (
                total_samples,
                history_length,
                20,
            ),
            dtype=np.float32,
        )
        for name in EXPECTED_MODALITIES
    }

    current_truth = np.empty(
        (
            total_samples,
            40,
        ),
        dtype=np.float32,
    )

    future_truth = np.empty(
        (
            total_samples,
            len(forecast_horizons),
            40,
        ),
        dtype=np.float32,
    )

    trajectory_index = np.empty(
        total_samples,
        dtype=np.int32,
    )

    time_index = np.empty(
        total_samples,
        dtype=np.int32,
    )

    sample_index = 0

    for trajectory in range(
        num_trajectories
    ):
        first_current_time = (
            history_length - 1
        )

        last_current_time = (
            num_time_steps
            - max_horizon
            - 1
        )

        for current_time in range(
            first_current_time,
            last_current_time + 1,
        ):
            start_time = (
                current_time
                - history_length
                + 1
            )

            end_time = (
                current_time + 1
            )

            for name in EXPECTED_MODALITIES:
                modality_windows[name][
                    sample_index
                ] = modalities[name][
                    trajectory,
                    start_time:end_time,
                    :
                ]

                mask_windows[name][
                    sample_index
                ] = masks[name][
                    trajectory,
                    start_time:end_time,
                    :
                ]

            current_truth[
                sample_index
            ] = truth[
                trajectory,
                current_time,
                :
            ]

            for horizon_index, horizon in enumerate(
                forecast_horizons
            ):
                future_truth[
                    sample_index,
                    horizon_index,
                    :
                ] = truth[
                    trajectory,
                    current_time + horizon,
                    :
                ]

            trajectory_index[
                sample_index
            ] = trajectory

            time_index[
                sample_index
            ] = current_time

            sample_index += 1

    if sample_index != total_samples:
        raise RuntimeError(
            "Internal temporal-window count mismatch."
        )

    return {
        "modalities": modality_windows,
        "masks": mask_windows,
        "current_truth": current_truth,
        "future_truth": future_truth,
        "trajectory_index": trajectory_index,
        "time_index": time_index,
        "forecast_horizons": np.asarray(
            forecast_horizons,
            dtype=np.int32,
        ),
    }