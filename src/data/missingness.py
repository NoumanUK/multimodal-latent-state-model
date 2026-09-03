from __future__ import annotations


from typing import Mapping

import numpy as np


EXPECTED_MODALITIES = (
    "A",
    "B",
    "C",
)

SUPPORTED_MISSING_RATES = (
    0.0,
    0.10,
    0.30,
    0.50,
)

SUPPORTED_MODALITY_COMBINATIONS = (
    ("A", "B", "C"),
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("A",),
    ("B",),
    ("C",),
)


def _validate_modalities(
    modalities: Mapping[str, np.ndarray],
) -> None:
    """
    Validate modality arrays.

    Expected:
        A: (..., 20)
        B: (..., 20)
        C: (..., 20)
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

    reference_shape = None

    for name in EXPECTED_MODALITIES:
        values = np.asarray(
            modalities[name]
        )

        if values.ndim < 2:
            raise ValueError(
                f"Modality {name} must have at least "
                f"2 dimensions. Received {values.shape}."
            )

        if values.shape[-1] != 20:
            raise ValueError(
                f"Modality {name} must have 20 features. "
                f"Received {values.shape}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Modality {name} contains NaN "
                f"or infinite values."
            )

        if reference_shape is None:
            reference_shape = values.shape
        elif values.shape != reference_shape:
            raise ValueError(
                "All modalities must have the same shape. "
                f"Expected {reference_shape}, "
                f"but {name} has {values.shape}."
            )


def _validate_missing_rate(
    missing_rate: float,
) -> None:
    """
    Validate that missing_rate is one of the frozen
    experimental conditions.
    """
    if missing_rate not in SUPPORTED_MISSING_RATES:
        raise ValueError(
            f"Unsupported missing rate {missing_rate}. "
            f"Supported rates are "
            f"{SUPPORTED_MISSING_RATES}."
        )


def create_element_mask(
    shape: tuple[int, ...],
    missing_rate: float,
    seed: int,
) -> np.ndarray:
    """
    Create an element-wise binary observation mask.

    mask = 1:
        observation is available

    mask = 0:
        observation is missing

    Missingness is MCAR:
        Missing Completely At Random.

    Each element is independently unavailable with
    probability equal to missing_rate.

    Returns:
        float32 array containing only 0.0 and 1.0.
    """
    _validate_missing_rate(
        missing_rate
    )

    if len(shape) < 1:
        raise ValueError(
            "shape must contain at least one dimension."
        )

    if any(
        dimension <= 0
        for dimension in shape
    ):
        raise ValueError(
            f"Invalid shape: {shape}"
        )

    if missing_rate == 0.0:
        return np.ones(
            shape,
            dtype=np.float32,
        )

    rng = np.random.default_rng(
        seed
    )

    available = rng.random(
        shape
    ) >= missing_rate

    return available.astype(
        np.float32
    )


def apply_mask(
    observations: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Apply an observation mask.

    Available:
        retain original measurement.

    Missing:
        replace with fill_value.

    The mask MUST later accompany these observations
    into the neural network.
    """
    observations = np.asarray(
        observations,
        dtype=np.float32,
    )

    mask = np.asarray(
        mask,
        dtype=np.float32,
    )

    if observations.shape != mask.shape:
        raise ValueError(
            "observations and mask must have "
            f"identical shapes. Received "
            f"{observations.shape} and {mask.shape}."
        )

    if not np.all(
        np.isfinite(observations)
    ):
        raise ValueError(
            "observations contains NaN "
            "or infinite values."
        )

    valid_mask = np.logical_or(
        mask == 0.0,
        mask == 1.0,
    )

    if not np.all(valid_mask):
        raise ValueError(
            "mask must contain only 0 and 1."
        )

    masked = np.where(
        mask == 1.0,
        observations,
        np.float32(fill_value),
    )

    return masked.astype(
        np.float32,
        copy=False,
    )


def apply_element_missingness(
    modalities: Mapping[str, np.ndarray],
    missing_rate: float,
    base_seed: int = 42,
    fill_value: float = 0.0,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """
    Apply independent element-level missingness
    to all three modalities.

    Separate deterministic random streams:

        A -> base_seed + 0
        B -> base_seed + 1
        C -> base_seed + 2

    Returns:
        masked_modalities,
        masks
    """
    _validate_modalities(
        modalities
    )

    _validate_missing_rate(
        missing_rate
    )

    masked_modalities: dict[
        str,
        np.ndarray,
    ] = {}

    masks: dict[
        str,
        np.ndarray,
    ] = {}

    for offset, name in enumerate(
        EXPECTED_MODALITIES
    ):
        values = np.asarray(
            modalities[name],
            dtype=np.float32,
        )

        mask = create_element_mask(
            shape=values.shape,
            missing_rate=missing_rate,
            seed=base_seed + offset,
        )

        masked = apply_mask(
            observations=values,
            mask=mask,
            fill_value=fill_value,
        )

        masks[name] = mask
        masked_modalities[name] = masked

    return (
        masked_modalities,
        masks,
    )


def apply_modality_dropout(
    modalities: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    available_modalities: tuple[str, ...],
    fill_value: float = 0.0,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """
    Remove entire modalities.

    Example:

        available_modalities = ("A", "C")

    means:
        A remains available
        B becomes completely unavailable
        C remains available

    Existing element-level masks are preserved for
    modalities that remain available.

    Dropped modalities receive:
        values = fill_value
        mask   = 0 everywhere
    """
    _validate_modalities(
        modalities
    )

    if not available_modalities:
        raise ValueError(
            "At least one modality must remain available."
        )

    available_set = set(
        available_modalities
    )

    unknown = available_set.difference(
        EXPECTED_MODALITIES
    )

    if unknown:
        raise ValueError(
            f"Unknown modalities: {sorted(unknown)}"
        )

    canonical_combination = tuple(
        name
        for name in EXPECTED_MODALITIES
        if name in available_set
    )

    if (
        canonical_combination
        not in SUPPORTED_MODALITY_COMBINATIONS
    ):
        raise ValueError(
            f"Unsupported modality combination: "
            f"{canonical_combination}"
        )

    output_modalities: dict[
        str,
        np.ndarray,
    ] = {}

    output_masks: dict[
        str,
        np.ndarray,
    ] = {}

    for name in EXPECTED_MODALITIES:
        values = np.asarray(
            modalities[name],
            dtype=np.float32,
        )

        if name not in masks:
            raise ValueError(
                f"Missing mask for modality {name}."
            )

        mask = np.asarray(
            masks[name],
            dtype=np.float32,
        )

        if values.shape != mask.shape:
            raise ValueError(
                f"Values and mask shape mismatch "
                f"for modality {name}."
            )

        if name in available_set:
            output_modalities[name] = (
                values.copy()
            )

            output_masks[name] = (
                mask.copy()
            )

        else:
            output_modalities[name] = (
                np.full_like(
                    values,
                    fill_value,
                    dtype=np.float32,
                )
            )

            output_masks[name] = (
                np.zeros_like(
                    mask,
                    dtype=np.float32,
                )
            )

    return (
        output_modalities,
        output_masks,
    )


def generate_missingness_conditions(
    modalities: Mapping[str, np.ndarray],
    base_seed: int = 42,
    fill_value: float = 0.0,
) -> dict[
    float,
    dict[str, dict[str, np.ndarray]],
]:
    """
    Generate the four frozen element-missingness
    conditions:

        0%
        10%
        30%
        50%

    Returns:

        {
            0.00: {
                "values": {...},
                "masks": {...},
            },

            0.10: {
                "values": {...},
                "masks": {...},
            },

            ...
        }

    Each missingness level receives a separate,
    deterministic random stream.
    """
    _validate_modalities(
        modalities
    )

    conditions = {}

    for level_index, level in enumerate(
        SUPPORTED_MISSING_RATES
    ):
        condition_seed = (
            base_seed
            + level_index * 1000
        )

        values, masks = (
            apply_element_missingness(
                modalities=modalities,
                missing_rate=level,
                base_seed=condition_seed,
                fill_value=fill_value,
            )
        )

        conditions[level] = {
            "values": values,
            "masks": masks,
        }

    return conditions


def get_modality_combinations(
) -> tuple[tuple[str, ...], ...]:
    """
    Return the seven non-empty modality combinations.
    """
    return SUPPORTED_MODALITY_COMBINATIONS