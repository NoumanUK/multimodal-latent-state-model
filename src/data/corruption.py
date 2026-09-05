from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from src.data.noise import add_noise_to_modalities

from src.data.missingness import (
    apply_element_missingness,
    apply_modality_dropout,
)

from src.data.normalization import normalize_modalities


MODALITY_NAMES = (
    "A",
    "B",
    "C",
)


# =========================================================
# Corruption condition
# =========================================================

@dataclass(frozen=True)
class CorruptionCondition:
    """
    Observation corruption configuration.

    noise_level:
        Gaussian noise relative to clean-training
        feature standard deviations.

    missing_rate:
        Elementwise MCAR missingness probability.

    available_modalities:
        Modalities retained after whole-modality dropout.
    """

    noise_level: float
    missing_rate: float
    available_modalities: tuple[str, ...]


# =========================================================
# Validation
# =========================================================

def _validate_condition(
    condition: CorruptionCondition,
) -> None:

    if not 0.0 <= condition.noise_level <= 1.0:
        raise ValueError(
            "noise_level must be between 0 and 1."
        )

    if not 0.0 <= condition.missing_rate <= 1.0:
        raise ValueError(
            "missing_rate must be between 0 and 1."
        )

    available = tuple(
        condition.available_modalities
    )

    if len(available) == 0:
        raise ValueError(
            "At least one modality must remain available."
        )

    if len(set(available)) != len(available):
        raise ValueError(
            "available_modalities contains duplicates."
        )

    unknown = (
        set(available)
        - set(MODALITY_NAMES)
    )

    if unknown:
        raise ValueError(
            f"Unknown modalities: {sorted(unknown)}"
        )


def _validate_inputs(
    clean_modalities: Mapping[
        str,
        np.ndarray,
    ],
    training_feature_stds: Mapping[
        str,
        np.ndarray,
    ],
    normalization_stats,
) -> None:

    for name in MODALITY_NAMES:

        if name not in clean_modalities:
            raise KeyError(
                f"Missing clean modality: {name}"
            )

        if name not in training_feature_stds:
            raise KeyError(
                f"Missing training standard deviations "
                f"for modality {name}"
            )

        if name not in normalization_stats:
            raise KeyError(
                f"Missing normalization statistics "
                f"for modality {name}"
            )

        values = np.asarray(
            clean_modalities[name]
        )

        if values.shape[-1] != 20:
            raise ValueError(
                f"{name}: expected feature dimension 20, "
                f"got {values.shape[-1]}."
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                f"{name}: clean observations contain "
                "non-finite values."
            )


# =========================================================
# Main corruption pipeline
# =========================================================

def corrupt_modalities(
    clean_modalities: dict[
        str,
        np.ndarray,
    ],
    training_feature_stds: dict[
        str,
        np.ndarray,
    ],
    normalization_stats,
    condition: CorruptionCondition,
    seed: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """
    Apply the frozen corruption pipeline.

    Order
    -----
    1. Gaussian observation noise
    2. Elementwise MCAR missingness
    3. Whole-modality dropout
    4. Clean-training normalization
    5. Missing normalized entries remain exactly zero

    Returns
    -------
    normalized_modalities
        Dictionary containing A, B and C.

    masks
        Binary dictionaries with:
            1 = observed
            0 = missing
    """

    _validate_condition(
        condition
    )

    _validate_inputs(
        clean_modalities=clean_modalities,
        training_feature_stds=(
            training_feature_stds
        ),
        normalization_stats=(
            normalization_stats
        ),
    )

    # =====================================================
    # 1. Gaussian observation noise
    # =====================================================

    noisy_modalities = add_noise_to_modalities(
        modalities=clean_modalities,
        training_feature_stds=(
            training_feature_stds
        ),
        noise_level=(
            condition.noise_level
        ),
        base_seed=seed,
    )

    # =====================================================
    # 2. Elementwise MCAR missingness
    # =====================================================

    (
        corrupted_modalities,
        masks,
    ) = apply_element_missingness(
        modalities=noisy_modalities,
        missing_rate=(
            condition.missing_rate
        ),
        base_seed=(
            seed + 100
        ),
        fill_value=0.0,
    )

    # =====================================================
    # 3. Whole-modality dropout
    # =====================================================

    (
        corrupted_modalities,
        masks,
    ) = apply_modality_dropout(
        modalities=corrupted_modalities,
        masks=masks,
        available_modalities=(
            condition.available_modalities
        ),
        fill_value=0.0,
    )

    # =====================================================
    # 4. Normalize
    # =====================================================

    normalized_modalities = normalize_modalities(
        modalities=corrupted_modalities,
        stats=normalization_stats,
        masks=masks,
    )

    # =====================================================
    # 5. Final safety checks
    # =====================================================

    available_set = set(
        condition.available_modalities
    )

    for name in MODALITY_NAMES:

        values = np.asarray(
            normalized_modalities[name]
        )

        mask = np.asarray(
            masks[name]
        )

        clean_values = np.asarray(
            clean_modalities[name]
        )

        if values.shape != clean_values.shape:
            raise RuntimeError(
                f"{name}: corrupted shape "
                f"{values.shape} does not match clean "
                f"shape {clean_values.shape}."
            )

        if mask.shape != values.shape:
            raise RuntimeError(
                f"{name}: mask shape "
                f"{mask.shape} does not match values "
                f"shape {values.shape}."
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise RuntimeError(
                f"{name}: normalized observations "
                "contain non-finite values."
            )

        binary_mask = np.logical_or(
            mask == 0,
            mask == 1,
        )

        if not np.all(
            binary_mask
        ):
            raise RuntimeError(
                f"{name}: mask contains values other "
                "than 0 or 1."
            )

        missing_locations = (
            mask == 0
        )

        if np.any(
            values[
                missing_locations
            ]
            != 0.0
        ):
            raise RuntimeError(
                f"{name}: missing normalized entries "
                "are not exactly zero."
            )

        if name not in available_set:

            if not np.all(
                mask == 0
            ):
                raise RuntimeError(
                    f"{name}: dropped modality mask "
                    "is not completely zero."
                )

            if not np.all(
                values == 0.0
            ):
                raise RuntimeError(
                    f"{name}: dropped modality values "
                    "are not completely zero."
                )

    return (
        normalized_modalities,
        masks,
    )


# =========================================================
# Clean condition
# =========================================================

def clean_condition() -> CorruptionCondition:

    return CorruptionCondition(
        noise_level=0.0,
        missing_rate=0.0,
        available_modalities=(
            "A",
            "B",
            "C",
        ),
    )


# =========================================================
# Noise-only evaluation conditions
# =========================================================

def noise_conditions() -> tuple[
    CorruptionCondition,
    ...
]:

    levels = (
        0.00,
        0.05,
        0.10,
        0.20,
    )

    return tuple(
        CorruptionCondition(
            noise_level=level,
            missing_rate=0.0,
            available_modalities=(
                "A",
                "B",
                "C",
            ),
        )
        for level in levels
    )


# =========================================================
# Missingness-only evaluation conditions
# =========================================================

def missingness_conditions() -> tuple[
    CorruptionCondition,
    ...
]:

    rates = (
        0.00,
        0.10,
        0.30,
        0.50,
    )

    return tuple(
        CorruptionCondition(
            noise_level=0.0,
            missing_rate=rate,
            available_modalities=(
                "A",
                "B",
                "C",
            ),
        )
        for rate in rates
    )


# =========================================================
# Modality-dropout evaluation conditions
# =========================================================

def modality_conditions() -> tuple[
    CorruptionCondition,
    ...
]:

    combinations = (
        (
            "A",
            "B",
            "C",
        ),
        (
            "A",
            "B",
        ),
        (
            "A",
            "C",
        ),
        (
            "B",
            "C",
        ),
        (
            "A",
        ),
        (
            "B",
        ),
        (
            "C",
        ),
    )

    return tuple(
        CorruptionCondition(
            noise_level=0.0,
            missing_rate=0.0,
            available_modalities=(
                combination
            ),
        )
        for combination in combinations
    )


# =========================================================
# Full factorial evaluation conditions
# =========================================================

def combined_conditions() -> tuple[
    CorruptionCondition,
    ...
]:

    noise_levels = (
        0.00,
        0.05,
        0.10,
        0.20,
    )

    missing_rates = (
        0.00,
        0.10,
        0.30,
        0.50,
    )

    modality_combinations = (
        (
            "A",
            "B",
            "C",
        ),
        (
            "A",
            "B",
        ),
        (
            "A",
            "C",
        ),
        (
            "B",
            "C",
        ),
        (
            "A",
        ),
        (
            "B",
        ),
        (
            "C",
        ),
    )

    conditions: list[
        CorruptionCondition
    ] = []

    for noise_level in noise_levels:

        for missing_rate in missing_rates:

            for available_modalities in (
                modality_combinations
            ):

                condition = CorruptionCondition(
                    noise_level=(
                        noise_level
                    ),
                    missing_rate=(
                        missing_rate
                    ),
                    available_modalities=(
                        available_modalities
                    ),
                )

                _validate_condition(
                    condition
                )

                conditions.append(
                    condition
                )

    return tuple(
        conditions
    )


# =========================================================
# Robust training sampler
# =========================================================

class TrainingCorruptionSampler:
    """
    Random corruption-condition sampler used during
    robust Temporal MVAE training.
    """

    def __init__(
        self,
        noise_levels: Iterable[
            float
        ] = (
            0.00,
            0.05,
            0.10,
            0.20,
        ),
        missing_rates: Iterable[
            float
        ] = (
            0.00,
            0.10,
            0.30,
            0.50,
        ),
        modality_combinations: Iterable[
            tuple[str, ...]
        ] = (
            (
                "A",
                "B",
                "C",
            ),
            (
                "A",
                "B",
            ),
            (
                "A",
                "C",
            ),
            (
                "B",
                "C",
            ),
            (
                "A",
            ),
            (
                "B",
            ),
            (
                "C",
            ),
        ),
        seed: int = 42,
    ) -> None:

        self.noise_levels = tuple(
            float(x)
            for x in noise_levels
        )

        self.missing_rates = tuple(
            float(x)
            for x in missing_rates
        )

        self.modality_combinations = tuple(
            tuple(x)
            for x in modality_combinations
        )

        if len(
            self.noise_levels
        ) == 0:
            raise ValueError(
                "noise_levels cannot be empty."
            )

        if len(
            self.missing_rates
        ) == 0:
            raise ValueError(
                "missing_rates cannot be empty."
            )

        if len(
            self.modality_combinations
        ) == 0:
            raise ValueError(
                "modality_combinations cannot be empty."
            )

        for noise_level in self.noise_levels:

            for missing_rate in self.missing_rates:

                for available_modalities in (
                    self.modality_combinations
                ):

                    _validate_condition(
                        CorruptionCondition(
                            noise_level=(
                                noise_level
                            ),
                            missing_rate=(
                                missing_rate
                            ),
                            available_modalities=(
                                available_modalities
                            ),
                        )
                    )

        self.rng = np.random.default_rng(
            seed
        )

    def sample(
        self,
    ) -> CorruptionCondition:

        noise_index = int(
            self.rng.integers(
                0,
                len(
                    self.noise_levels
                ),
            )
        )

        missing_index = int(
            self.rng.integers(
                0,
                len(
                    self.missing_rates
                ),
            )
        )

        modality_index = int(
            self.rng.integers(
                0,
                len(
                    self.modality_combinations
                ),
            )
        )

        condition = CorruptionCondition(
            noise_level=(
                self.noise_levels[
                    noise_index
                ]
            ),
            missing_rate=(
                self.missing_rates[
                    missing_index
                ]
            ),
            available_modalities=(
                self.modality_combinations[
                    modality_index
                ]
            ),
        )

        _validate_condition(
            condition
        )

        return condition