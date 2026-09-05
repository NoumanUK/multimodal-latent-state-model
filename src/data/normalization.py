from __future__ import annotations

from typing import Mapping

import numpy as np


EXPECTED_MODALITIES = (
    "A",
    "B",
    "C",
)


def compute_normalization_stats(
    train_modalities: Mapping[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute feature-wise mean and standard deviation
    using CLEAN TRAINING MODALITIES ONLY.

    Expected modality shape:
        (trajectories, time, 20)

    Returns:
        {
            "A": {
                "mean": (20,),
                "std":  (20,),
            },
            ...
        }

    Validation and test data must never be used to
    calculate these statistics.
    """
    stats: dict[
        str,
        dict[str, np.ndarray],
    ] = {}

    for name in EXPECTED_MODALITIES:
        if name not in train_modalities:
            raise ValueError(
                f"Missing training modality {name}."
            )

        values = np.asarray(
            train_modalities[name],
            dtype=np.float64,
        )

        if values.ndim < 2:
            raise ValueError(
                f"Modality {name} must have at least "
                f"2 dimensions."
            )

        if values.shape[-1] != 20:
            raise ValueError(
                f"Expected 20 features for modality "
                f"{name}, received {values.shape[-1]}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Training modality {name} contains "
                f"NaN or infinite values."
            )

        flattened = values.reshape(
            -1,
            values.shape[-1],
        )

        mean = np.mean(
            flattened,
            axis=0,
        )

        std = np.std(
            flattened,
            axis=0,
            ddof=0,
        )

        if not np.all(np.isfinite(mean)):
            raise ValueError(
                f"Non-finite mean found for "
                f"modality {name}."
            )

        if not np.all(np.isfinite(std)):
            raise ValueError(
                f"Non-finite standard deviation found "
                f"for modality {name}."
            )

        if np.any(std <= 0.0):
            indices = np.where(
                std <= 0.0
            )[0]

            raise ValueError(
                f"Zero-variance features found in "
                f"modality {name}: "
                f"{indices.tolist()}"
            )

        stats[name] = {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }

    return stats


def normalize_observations(
    observations: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Feature-wise standardization.

        normalized = (x - training_mean)
                     / training_std

    If a binary mask is supplied:

        mask = 1 -> normalized observation retained
        mask = 0 -> normalized value forced to 0

    This ensures missing observations use zero only
    as a placeholder after normalization.
    """
    observations = np.asarray(
        observations,
        dtype=np.float32,
    )

    mean = np.asarray(
        mean,
        dtype=np.float32,
    )

    std = np.asarray(
        std,
        dtype=np.float32,
    )

    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError(
            "mean and std must be one-dimensional."
        )

    if mean.shape != std.shape:
        raise ValueError(
            "mean and std must have identical shapes."
        )

    if observations.shape[-1] != mean.shape[0]:
        raise ValueError(
            "Normalization statistics do not match "
            "the observation feature dimension."
        )

    if not np.all(np.isfinite(observations)):
        raise ValueError(
            "observations contains NaN or Inf."
        )

    if not np.all(np.isfinite(mean)):
        raise ValueError(
            "mean contains NaN or Inf."
        )

    if not np.all(np.isfinite(std)):
        raise ValueError(
            "std contains NaN or Inf."
        )

    if np.any(std <= 0.0):
        raise ValueError(
            "All standard deviations must be positive."
        )

    normalized = (
        observations - mean
    ) / std

    if mask is not None:
        mask = np.asarray(
            mask,
            dtype=np.float32,
        )

        if mask.shape != observations.shape:
            raise ValueError(
                "mask and observations must have "
                "identical shapes."
            )

        valid_mask = np.logical_or(
            mask == 0.0,
            mask == 1.0,
        )

        if not np.all(valid_mask):
            raise ValueError(
                "mask must contain only 0 and 1."
            )

        normalized = np.where(
            mask == 1.0,
            normalized,
            0.0,
        )

    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            "Normalization produced NaN or Inf."
        )

    return normalized.astype(
        np.float32,
        copy=False,
    )


def normalize_modalities(
    modalities: Mapping[str, np.ndarray],
    stats: Mapping[
        str,
        Mapping[str, np.ndarray],
    ],
    masks: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """
    Normalize A/B/C using training-derived statistics.

    The same stats object must be used for:
        training
        validation
        test

    Optional masks preserve missing observations as
    zero after normalization.
    """
    normalized: dict[
        str,
        np.ndarray,
    ] = {}

    for name in EXPECTED_MODALITIES:
        if name not in modalities:
            raise ValueError(
                f"Missing modality {name}."
            )

        if name not in stats:
            raise ValueError(
                f"Missing normalization statistics "
                f"for modality {name}."
            )

        if (
            "mean" not in stats[name]
            or "std" not in stats[name]
        ):
            raise ValueError(
                f"Statistics for modality {name} "
                f"must contain mean and std."
            )

        mask = None

        if masks is not None:
            if name not in masks:
                raise ValueError(
                    f"Missing mask for modality {name}."
                )

            mask = masks[name]

        normalized[name] = (
            normalize_observations(
                observations=modalities[name],
                mean=stats[name]["mean"],
                std=stats[name]["std"],
                mask=mask,
            )
        )

    return normalized


def denormalize_observations(
    normalized: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Convert normalized observations back into their
    original measurement scale.

        x = normalized * training_std
            + training_mean
    """
    normalized = np.asarray(
        normalized,
        dtype=np.float32,
    )

    mean = np.asarray(
        mean,
        dtype=np.float32,
    )

    std = np.asarray(
        std,
        dtype=np.float32,
    )

    if normalized.shape[-1] != mean.shape[0]:
        raise ValueError(
            "Feature dimension does not match mean."
        )

    if mean.shape != std.shape:
        raise ValueError(
            "mean and std must have identical shapes."
        )

    restored = (
        normalized * std
        + mean
    )

    return restored.astype(
        np.float32,
        copy=False,
    )

def compute_state_normalization_stats(
    train_states: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute feature-wise normalization statistics
    from TRAINING TRUTH ONLY.

    Expected shape:
        (trajectories, time, 40)
    """
    states = np.asarray(
        train_states,
        dtype=np.float64,
    )

    if states.ndim != 3:
        raise ValueError(
            "train_states must have shape "
            "(trajectories, time, 40)."
        )

    if states.shape[-1] != 40:
        raise ValueError(
            f"Expected 40 state variables, received "
            f"{states.shape[-1]}."
        )

    if not np.all(np.isfinite(states)):
        raise ValueError(
            "train_states contains NaN or Inf."
        )

    flattened = states.reshape(
        -1,
        states.shape[-1],
    )

    mean = np.mean(
        flattened,
        axis=0,
    )

    std = np.std(
        flattened,
        axis=0,
        ddof=0,
    )

    if not np.all(np.isfinite(mean)):
        raise ValueError(
            "State mean contains NaN or Inf."
        )

    if not np.all(np.isfinite(std)):
        raise ValueError(
            "State std contains NaN or Inf."
        )

    if np.any(std <= 0.0):
        raise ValueError(
            "All state standard deviations "
            "must be positive."
        )

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def normalize_states(
    states: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Normalize Lorenz-96 states using training
    statistics.
    """
    states = np.asarray(
        states,
        dtype=np.float32,
    )

    mean = np.asarray(
        mean,
        dtype=np.float32,
    )

    std = np.asarray(
        std,
        dtype=np.float32,
    )

    if states.shape[-1] != 40:
        raise ValueError(
            "Expected state dimension 40."
        )

    if mean.shape != (40,):
        raise ValueError(
            "State mean must have shape (40,)."
        )

    if std.shape != (40,):
        raise ValueError(
            "State std must have shape (40,)."
        )

    if np.any(std <= 0.0):
        raise ValueError(
            "State standard deviations must "
            "be positive."
        )

    normalized = (
        states - mean
    ) / std

    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            "State normalization produced "
            "NaN or Inf."
        )

    return normalized.astype(
        np.float32,
        copy=False,
    )


def denormalize_states(
    normalized_states: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Convert normalized Lorenz-96 states back
    to original physical scale.
    """
    normalized_states = np.asarray(
        normalized_states,
        dtype=np.float32,
    )

    mean = np.asarray(
        mean,
        dtype=np.float32,
    )

    std = np.asarray(
        std,
        dtype=np.float32,
    )

    if normalized_states.shape[-1] != 40:
        raise ValueError(
            "Expected state dimension 40."
        )

    restored = (
        normalized_states * std
        + mean
    )

    if not np.all(np.isfinite(restored)):
        raise ValueError(
            "State denormalization produced "
            "NaN or Inf."
        )

    return restored.astype(
        np.float32,
        copy=False,
    )