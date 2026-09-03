from __future__ import annotations

from typing import Mapping

import numpy as np


SUPPORTED_NOISE_LEVELS = (
    0.0,
    0.05,
    0.10,
    0.20,
)

EXPECTED_MODALITIES = (
    "A",
    "B",
    "C",
)


def _validate_modalities(
    modalities: Mapping[str, np.ndarray],
) -> None:
    """
    Validate a dictionary of clean/noisy modality arrays.

    Expected structure:
        {
            "A": (..., 20),
            "B": (..., 20),
            "C": (..., 20),
        }
    """
    missing = [
        name
        for name in EXPECTED_MODALITIES
        if name not in modalities
    ]

    if missing:
        raise ValueError(
            f"Missing modalities: {missing}"
        )

    for name in EXPECTED_MODALITIES:
        values = np.asarray(
            modalities[name]
        )

        if values.ndim < 2:
            raise ValueError(
                f"Modality {name} must have at least "
                f"2 dimensions, received {values.shape}."
            )

        if values.shape[-1] != 20:
            raise ValueError(
                f"Modality {name} must have 20 features, "
                f"received shape {values.shape}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Modality {name} contains NaN "
                f"or infinite values."
            )


def compute_training_feature_stds(
    train_modalities: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Compute one standard deviation per feature for each
    modality using TRAINING DATA ONLY.

    For training arrays shaped:

        (trajectories, time, features)

    all axes except the final feature axis are collapsed.

    Returns:
        {
            "A": (20,),
            "B": (20,),
            "C": (20,),
        }

    These standard deviations define the scale of
    observation noise for train, validation and test data.
    """
    _validate_modalities(
        train_modalities
    )

    feature_stds: dict[
        str,
        np.ndarray,
    ] = {}

    for name in EXPECTED_MODALITIES:
        values = np.asarray(
            train_modalities[name],
            dtype=np.float64,
        )

        flattened = values.reshape(
            -1,
            values.shape[-1],
        )

        stds = np.std(
            flattened,
            axis=0,
            ddof=0,
        )

        if np.any(~np.isfinite(stds)):
            raise ValueError(
                f"Non-finite feature standard deviation "
                f"found for modality {name}."
            )

        if np.any(stds <= 0.0):
            bad_indices = np.where(
                stds <= 0.0
            )[0]

            raise ValueError(
                f"Modality {name} contains zero-variance "
                f"features at indices "
                f"{bad_indices.tolist()}."
            )

        feature_stds[name] = (
            stds.astype(np.float32)
        )

    return feature_stds


def add_gaussian_noise(
    observations: np.ndarray,
    feature_stds: np.ndarray,
    noise_level: float,
    seed: int,
) -> np.ndarray:
    """
    Add independent Gaussian observation noise.

    For feature j:

        epsilon_j ~ N(
            0,
            (noise_level * feature_std_j)^2
        )

        y_noisy = y_clean + epsilon

    Example:
        noise_level = 0.10

    means the noise standard deviation for each
    observation feature is 10% of that feature's
    TRAINING standard deviation.
    """
    observations = np.asarray(
        observations,
        dtype=np.float32,
    )

    feature_stds = np.asarray(
        feature_stds,
        dtype=np.float32,
    )

    if observations.ndim < 2:
        raise ValueError(
            "observations must have at least "
            "2 dimensions."
        )

    if feature_stds.ndim != 1:
        raise ValueError(
            "feature_stds must be one-dimensional."
        )

    if (
        observations.shape[-1]
        != feature_stds.shape[0]
    ):
        raise ValueError(
            "Number of feature standard deviations "
            "does not match observation dimension."
        )

    if noise_level < 0.0:
        raise ValueError(
            "noise_level cannot be negative."
        )

    if not np.all(
        np.isfinite(observations)
    ):
        raise ValueError(
            "observations contains NaN "
            "or infinite values."
        )

    if not np.all(
        np.isfinite(feature_stds)
    ):
        raise ValueError(
            "feature_stds contains NaN "
            "or infinite values."
        )

    if np.any(feature_stds <= 0.0):
        raise ValueError(
            "All feature standard deviations "
            "must be positive."
        )

    # At zero noise, return an independent copy
    # without consuming random numbers.
    if noise_level == 0.0:
        return observations.copy()

    rng = np.random.default_rng(
        seed
    )

    noise_scale = (
        noise_level * feature_stds
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_scale,
        size=observations.shape,
    )

    noisy_observations = (
        observations.astype(
            np.float64,
            copy=False,
        )
        + noise
    )

    if not np.all(
        np.isfinite(noisy_observations)
    ):
        raise ValueError(
            "Noise generation produced NaN "
            "or infinite values."
        )

    return noisy_observations.astype(
        np.float32
    )


def add_noise_to_modalities(
    modalities: Mapping[str, np.ndarray],
    training_feature_stds: Mapping[
        str,
        np.ndarray,
    ],
    noise_level: float,
    base_seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Add Gaussian noise to A, B and C.

    Separate deterministic random streams are used
    for each modality:

        A -> base_seed + 0
        B -> base_seed + 1
        C -> base_seed + 2

    The supplied standard deviations MUST have been
    calculated from clean TRAINING observations only.
    """
    _validate_modalities(
        modalities
    )

    if noise_level not in SUPPORTED_NOISE_LEVELS:
        raise ValueError(
            f"Unsupported noise level {noise_level}. "
            f"Supported levels are "
            f"{SUPPORTED_NOISE_LEVELS}."
        )

    for name in EXPECTED_MODALITIES:
        if name not in training_feature_stds:
            raise ValueError(
                f"Missing training feature standard "
                f"deviations for modality {name}."
            )

    noisy: dict[
        str,
        np.ndarray,
    ] = {}

    for offset, name in enumerate(
        EXPECTED_MODALITIES
    ):
        noisy[name] = add_gaussian_noise(
            observations=modalities[name],
            feature_stds=training_feature_stds[
                name
            ],
            noise_level=noise_level,
            seed=base_seed + offset,
        )

    return noisy


def generate_noise_conditions(
    modalities: Mapping[str, np.ndarray],
    training_feature_stds: Mapping[
        str,
        np.ndarray,
    ],
    base_seed: int = 42,
) -> dict[
    float,
    dict[str, np.ndarray],
]:
    """
    Generate all frozen observation-noise conditions:

        0%
        5%
        10%
        20%

    Returns:
        {
            0.00: {"A": ..., "B": ..., "C": ...},
            0.05: {"A": ..., "B": ..., "C": ...},
            0.10: {"A": ..., "B": ..., "C": ...},
            0.20: {"A": ..., "B": ..., "C": ...},
        }

    Each condition is reproducible.

    Different noise levels use separate deterministic
    random streams so that experiment conditions do
    not accidentally share identical standardized
    noise samples.
    """
    _validate_modalities(
        modalities
    )

    conditions: dict[
        float,
        dict[str, np.ndarray],
    ] = {}

    for level_index, level in enumerate(
        SUPPORTED_NOISE_LEVELS
    ):
        condition_seed = (
            base_seed
            + level_index * 1000
        )

        conditions[level] = (
            add_noise_to_modalities(
                modalities=modalities,
                training_feature_stds=(
                    training_feature_stds
                ),
                noise_level=level,
                base_seed=condition_seed,
            )
        )

    return conditions