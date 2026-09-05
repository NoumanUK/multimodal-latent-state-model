from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import LorenzMultimodalDataset

from src.data.observation_operators import (
    generate_clean_modalities,
)

from src.data.normalization import (
    compute_normalization_stats,
    compute_state_normalization_stats,
    normalize_states,
)

from src.data.noise import (
    compute_training_feature_stds,
)

from src.data.corruption import (
    CorruptionCondition,
    corrupt_modalities,
)

from src.models.temporal_mvae import (
    TemporalMVAE,
)


# =========================================================
# Configuration
# =========================================================

SEED = 42

DATA_PATH = Path(
    "data/raw/lorenz96_clean.npz"
)

CHECKPOINT_PATH = Path(
    "experiments/checkpoints/"
    "temporal_mvae_robust_best.pt"
)

RESULT_DIR = Path(
    "experiments/results"
)

JSON_PATH = (
    RESULT_DIR
    / "temporal_uncertainty_results.json"
)

CSV_PATH = (
    RESULT_DIR
    / "temporal_uncertainty_results.csv"
)

BATCH_SIZE = 256

LATENT_DIM = 16

HISTORY_LENGTH = 20

FORECAST_HORIZONS = (
    1,
    5,
    10,
)

# Number of posterior samples used for each test example.
#
# 100 gives reasonable empirical 95% interval estimates
# without making the experiment unnecessarily enormous.
MC_SAMPLES = 100

MODALITY_NAMES = (
    "A",
    "B",
    "C",
)


# =========================================================
# Reproducibility
# =========================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# =========================================================
# Evaluation conditions
# =========================================================

def build_conditions():

    """
    Selected uncertainty conditions.

    We intentionally do not evaluate every robustness
    condition here.

    These six conditions represent progressively poorer
    information availability.
    """

    return (
        (
            "clean_ABC",
            CorruptionCondition(
                noise_level=0.0,
                missing_rate=0.0,
                available_modalities=(
                    "A",
                    "B",
                    "C",
                ),
            ),
        ),
        (
            "missing_30",
            CorruptionCondition(
                noise_level=0.0,
                missing_rate=0.30,
                available_modalities=(
                    "A",
                    "B",
                    "C",
                ),
            ),
        ),
        (
            "missing_50",
            CorruptionCondition(
                noise_level=0.0,
                missing_rate=0.50,
                available_modalities=(
                    "A",
                    "B",
                    "C",
                ),
            ),
        ),
        (
            "modalities_AB",
            CorruptionCondition(
                noise_level=0.0,
                missing_rate=0.0,
                available_modalities=(
                    "A",
                    "B",
                ),
            ),
        ),
        (
            "modalities_A",
            CorruptionCondition(
                noise_level=0.0,
                missing_rate=0.0,
                available_modalities=(
                    "A",
                ),
            ),
        ),
        (
            "modalities_C",
            CorruptionCondition(
                noise_level=0.0,
                missing_rate=0.0,
                available_modalities=(
                    "C",
                ),
            ),
        ),
    )


# =========================================================
# Dataset
# =========================================================

def make_loader(
    modalities,
    masks,
    truth,
):

    dataset = LorenzMultimodalDataset(
        modalities=modalities,
        masks=masks,
        truth=truth,
        history_length=HISTORY_LENGTH,
        forecast_horizons=(
            FORECAST_HORIZONS
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    return (
        dataset,
        loader,
    )


# =========================================================
# Model
# =========================================================

def load_model(
    device,
):

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = TemporalMVAE(
        modality_dim=20,
        latent_dim=LATENT_DIM,
        state_dim=40,
        encoder_hidden_dim=64,
        transition_hidden_dim=64,
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        checkpoint,
    )


# =========================================================
# Accumulator
# =========================================================

def empty_accumulator():

    return {
        "sse": 0.0,
        "elements": 0,
        "covered": 0,
        "interval_width_sum": 0.0,
        "predictive_std_sum": 0.0,
    }


def update_accumulator(
    accumulator,
    mean_prediction,
    lower,
    upper,
    predictive_std,
    truth,
):

    error = (
        mean_prediction
        - truth
    )

    accumulator[
        "sse"
    ] += torch.sum(
        error ** 2
    ).item()

    accumulator[
        "elements"
    ] += (
        truth.numel()
    )

    covered = (
        (truth >= lower)
        &
        (truth <= upper)
    )

    accumulator[
        "covered"
    ] += torch.sum(
        covered
    ).item()

    accumulator[
        "interval_width_sum"
    ] += torch.sum(
        upper - lower
    ).item()

    accumulator[
        "predictive_std_sum"
    ] += torch.sum(
        predictive_std
    ).item()


def finalize_accumulator(
    accumulator,
):

    mse = (
        accumulator[
            "sse"
        ]
        /
        accumulator[
            "elements"
        ]
    )

    return {
        "mse": float(
            mse
        ),
        "rmse": float(
            math.sqrt(
                mse
            )
        ),
        "coverage_95": float(
            accumulator[
                "covered"
            ]
            /
            accumulator[
                "elements"
            ]
        ),
        "mean_interval_width_95": float(
            accumulator[
                "interval_width_sum"
            ]
            /
            accumulator[
                "elements"
            ]
        ),
        "mean_predictive_std": float(
            accumulator[
                "predictive_std_sum"
            ]
            /
            accumulator[
                "elements"
            ]
        ),
    }


# =========================================================
# Monte Carlo evaluation
# =========================================================

@torch.no_grad()
def evaluate_condition(
    model,
    loader,
    device,
):

    """
    Repeatedly sample the latent posterior.

    Each model call uses:

        deterministic=False

    Therefore:

        z_t ~ q(z_t | observation history)

    The latent transition and decoder are deterministic,
    so dispersion in future predictions comes entirely
    from posterior uncertainty in z_t.

    This should therefore be described as:

        latent-posterior-induced predictive uncertainty

    rather than complete predictive uncertainty.
    """

    model.eval()

    accumulators = {
        "current": (
            empty_accumulator()
        ),
        "h1": (
            empty_accumulator()
        ),
        "h5": (
            empty_accumulator()
        ),
        "h10": (
            empty_accumulator()
        ),
    }

    batch_index = 0

    for batch in loader:

        batch_index += 1

        modalities = {
            name:
            batch[
                "modalities"
            ][name].to(
                device,
                non_blocking=True,
            )
            for name
            in MODALITY_NAMES
        }

        masks = {
            name:
            batch[
                "masks"
            ][name].to(
                device,
                non_blocking=True,
            )
            for name
            in MODALITY_NAMES
        }

        current_truth = (
            batch[
                "current_truth"
            ].to(
                device,
                non_blocking=True,
            )
        )

        future_truth = (
            batch[
                "future_truth"
            ].to(
                device,
                non_blocking=True,
            )
        )

        current_samples = []

        forecast_samples = []

        # -------------------------------------------------
        # Monte Carlo posterior sampling
        # -------------------------------------------------

        for _ in range(
            MC_SAMPLES
        ):

            output = model(
                modalities=modalities,
                masks=masks,
                available_modalities=(
                    "A",
                    "B",
                    "C",
                ),
                horizons=(
                    FORECAST_HORIZONS
                ),
                deterministic=False,
            )

            current_samples.append(
                output[
                    "reconstruction"
                ]
            )

            forecast_samples.append(
                output[
                    "forecast"
                ]
            )

        # [S, B, 40]
        current_samples = torch.stack(
            current_samples,
            dim=0,
        )

        # [S, B, 3, 40]
        forecast_samples = torch.stack(
            forecast_samples,
            dim=0,
        )

        # -------------------------------------------------
        # Current state uncertainty
        # -------------------------------------------------

        current_mean = torch.mean(
            current_samples,
            dim=0,
        )

        current_std = torch.std(
            current_samples,
            dim=0,
            unbiased=False,
        )

        current_lower = torch.quantile(
            current_samples,
            q=0.025,
            dim=0,
        )

        current_upper = torch.quantile(
            current_samples,
            q=0.975,
            dim=0,
        )

        update_accumulator(
            accumulator=(
                accumulators[
                    "current"
                ]
            ),
            mean_prediction=(
                current_mean
            ),
            lower=(
                current_lower
            ),
            upper=(
                current_upper
            ),
            predictive_std=(
                current_std
            ),
            truth=(
                current_truth
            ),
        )

        # -------------------------------------------------
        # Forecast horizons
        # -------------------------------------------------

        for horizon_index, horizon in enumerate(
            FORECAST_HORIZONS
        ):

            horizon_samples = (
                forecast_samples[
                    :,
                    :,
                    horizon_index,
                    :
                ]
            )

            horizon_mean = torch.mean(
                horizon_samples,
                dim=0,
            )

            horizon_std = torch.std(
                horizon_samples,
                dim=0,
                unbiased=False,
            )

            horizon_lower = torch.quantile(
                horizon_samples,
                q=0.025,
                dim=0,
            )

            horizon_upper = torch.quantile(
                horizon_samples,
                q=0.975,
                dim=0,
            )

            horizon_truth = (
                future_truth[
                    :,
                    horizon_index,
                    :
                ]
            )

            update_accumulator(
                accumulator=(
                    accumulators[
                        f"h{horizon}"
                    ]
                ),
                mean_prediction=(
                    horizon_mean
                ),
                lower=(
                    horizon_lower
                ),
                upper=(
                    horizon_upper
                ),
                predictive_std=(
                    horizon_std
                ),
                truth=(
                    horizon_truth
                ),
            )

        if (
            batch_index % 20
            == 0
        ):

            print(
                f"    processed "
                f"{batch_index} batches"
            )

    return {
        name:
        finalize_accumulator(
            accumulator
        )
        for name, accumulator
        in accumulators.items()
    }


# =========================================================
# Saving
# =========================================================

def save_results(
    results,
):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    fieldnames = (
        "condition",
        "target",
        "noise_level",
        "missing_rate",
        "available_modalities",
        "mse",
        "rmse",
        "coverage_95",
        "mean_interval_width_95",
        "mean_predictive_std",
    )

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for condition_result in results:

            for target in (
                "current",
                "h1",
                "h5",
                "h10",
            ):

                metrics = (
                    condition_result[
                        target
                    ]
                )

                writer.writerow(
                    {
                        "condition": (
                            condition_result[
                                "condition"
                            ]
                        ),
                        "target": (
                            target
                        ),
                        "noise_level": (
                            condition_result[
                                "noise_level"
                            ]
                        ),
                        "missing_rate": (
                            condition_result[
                                "missing_rate"
                            ]
                        ),
                        "available_modalities": (
                            condition_result[
                                "available_modalities"
                            ]
                        ),
                        "mse": (
                            metrics[
                                "mse"
                            ]
                        ),
                        "rmse": (
                            metrics[
                                "rmse"
                            ]
                        ),
                        "coverage_95": (
                            metrics[
                                "coverage_95"
                            ]
                        ),
                        "mean_interval_width_95": (
                            metrics[
                                "mean_interval_width_95"
                            ]
                        ),
                        "mean_predictive_std": (
                            metrics[
                                "mean_predictive_std"
                            ]
                        ),
                    }
                )


# =========================================================
# Printing
# =========================================================

def print_condition_result(
    result,
):

    print()

    print(
        f"Condition: "
        f"{result['condition']}"
    )

    print(
        "-" * 100
    )

    for target in (
        "current",
        "h1",
        "h5",
        "h10",
    ):

        metrics = (
            result[
                target
            ]
        )

        print(
            f"{target:8s}"
            f" | RMSE "
            f"{metrics['rmse']:.6f}"
            f" | coverage95 "
            f"{metrics['coverage_95'] * 100:6.2f}%"
            f" | width95 "
            f"{metrics['mean_interval_width_95']:.6f}"
            f" | pred std "
            f"{metrics['mean_predictive_std']:.6f}"
        )


# =========================================================
# Main
# =========================================================

def main():

    set_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "Monte Carlo samples:",
        MC_SAMPLES,
    )

    # =====================================================
    # Data
    # =====================================================

    print()
    print(
        "Loading Lorenz-96 data..."
    )

    data = np.load(
        DATA_PATH
    )

    train_truth_raw = (
        data[
            "train_states"
        ].astype(
            np.float32
        )
    )

    test_truth_raw = (
        data[
            "test_states"
        ].astype(
            np.float32
        )
    )

    print(
        "Train shape:",
        train_truth_raw.shape,
    )

    print(
        "Test shape:",
        test_truth_raw.shape,
    )

    # =====================================================
    # Observations
    # =====================================================

    print()
    print(
        "Generating clean modalities..."
    )

    clean_train_modalities = (
        generate_clean_modalities(
            train_truth_raw
        )
    )

    clean_test_modalities = (
        generate_clean_modalities(
            test_truth_raw
        )
    )

    observation_stats = (
        compute_normalization_stats(
            clean_train_modalities
        )
    )

    training_feature_stds = (
        compute_training_feature_stds(
            clean_train_modalities
        )
    )

    state_stats = (
        compute_state_normalization_stats(
            train_truth_raw
        )
    )

    test_truth = normalize_states(
        test_truth_raw,
        state_stats[
            "mean"
        ],
        state_stats[
            "std"
        ],
    )

    # =====================================================
    # Frozen robust temporal model
    # =====================================================

    print()
    print(
        "Loading robust Temporal MVAE..."
    )

    (
        model,
        checkpoint,
    ) = load_model(
        device
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    # =====================================================
    # Evaluate
    # =====================================================

    conditions = (
        build_conditions()
    )

    results = []

    print()
    print(
        "=" * 100
    )

    print(
        "TEMPORAL MVAE UNCERTAINTY EVALUATION"
    )

    print(
        "=" * 100
    )

    for condition_index, (
        condition_name,
        condition,
    ) in enumerate(
        conditions
    ):

        print()

        print(
            f"Evaluating "
            f"{condition_name}..."
        )

        condition_seed = (
            SEED
            + 2_000_000
            + condition_index * 10_000
        )

        (
            test_modalities,
            test_masks,
        ) = corrupt_modalities(
            clean_modalities=(
                clean_test_modalities
            ),
            training_feature_stds=(
                training_feature_stds
            ),
            normalization_stats=(
                observation_stats
            ),
            condition=condition,
            seed=condition_seed,
        )

        (
            dataset,
            loader,
        ) = make_loader(
            modalities=(
                test_modalities
            ),
            masks=test_masks,
            truth=test_truth,
        )

        print(
            "    test windows:",
            len(dataset),
        )

        condition_metrics = (
            evaluate_condition(
                model=model,
                loader=loader,
                device=device,
            )
        )

        result = {
            "condition": (
                condition_name
            ),
            "noise_level": float(
                condition.noise_level
            ),
            "missing_rate": float(
                condition.missing_rate
            ),
            "available_modalities": (
                "".join(
                    condition.available_modalities
                )
            ),
            **condition_metrics,
        }

        results.append(
            result
        )

        print_condition_result(
            result
        )

    # =====================================================
    # Save
    # =====================================================

    save_results(
        results
    )

    # =====================================================
    # Compact uncertainty summary
    # =====================================================

    print()

    print(
        "=" * 100
    )

    print(
        "CURRENT-STATE UNCERTAINTY SUMMARY"
    )

    print(
        "=" * 100
    )

    for result in results:

        metrics = (
            result[
                "current"
            ]
        )

        print(
            f"{result['condition']:16s}"
            f" | RMSE "
            f"{metrics['rmse']:.6f}"
            f" | coverage "
            f"{metrics['coverage_95'] * 100:6.2f}%"
            f" | width "
            f"{metrics['mean_interval_width_95']:.6f}"
            f" | std "
            f"{metrics['mean_predictive_std']:.6f}"
        )

    print()

    print(
        "Evaluation finished."
    )

    print(
        "JSON:",
        JSON_PATH,
    )

    print(
        "CSV:",
        CSV_PATH,
    )


if __name__ == "__main__":
    main()