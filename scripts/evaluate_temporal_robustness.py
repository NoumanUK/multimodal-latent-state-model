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
from src.data.observation_operators import generate_clean_modalities

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

from src.models.temporal_mvae import TemporalMVAE


# =========================================================
# Configuration
# =========================================================

SEED = 42

DATA_PATH = Path(
    "data/raw/lorenz96_clean.npz"
)

CLEAN_CHECKPOINT = Path(
    "experiments/checkpoints/"
    "temporal_mvae_clean_best.pt"
)

ROBUST_CHECKPOINT = Path(
    "experiments/checkpoints/"
    "temporal_mvae_robust_best.pt"
)

RESULT_DIR = Path(
    "experiments/results"
)

JSON_PATH = (
    RESULT_DIR
    / "temporal_robustness_results.json"
)

CSV_PATH = (
    RESULT_DIR
    / "temporal_robustness_results.csv"
)


BATCH_SIZE = 256

LATENT_DIM = 16

HISTORY_LENGTH = 20

FORECAST_HORIZONS = (
    1,
    5,
    10,
)

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
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Conditions
# =========================================================

def build_conditions():
    """
    Build the three isolated evaluation families.

    Important:
    We do NOT evaluate the complete 112-condition
    factorial here.

    We isolate:
        1. observation noise
        2. elementwise missingness
        3. whole-modality dropout
    """

    conditions = []

    # -----------------------------------------------------
    # Noise-only
    # -----------------------------------------------------

    for noise_level in (
        0.00,
        0.05,
        0.10,
        0.20,
    ):

        conditions.append(
            (
                "noise",
                f"noise_{int(noise_level * 100)}",
                CorruptionCondition(
                    noise_level=noise_level,
                    missing_rate=0.0,
                    available_modalities=(
                        "A",
                        "B",
                        "C",
                    ),
                ),
            )
        )

    # -----------------------------------------------------
    # Missingness-only
    # -----------------------------------------------------

    for missing_rate in (
        0.00,
        0.10,
        0.30,
        0.50,
    ):

        conditions.append(
            (
                "missingness",
                f"missing_{int(missing_rate * 100)}",
                CorruptionCondition(
                    noise_level=0.0,
                    missing_rate=missing_rate,
                    available_modalities=(
                        "A",
                        "B",
                        "C",
                    ),
                ),
            )
        )

    # -----------------------------------------------------
    # Modality dropout
    # -----------------------------------------------------

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

    for combination in combinations:

        name = "".join(
            combination
        )

        conditions.append(
            (
                "modalities",
                f"modalities_{name}",
                CorruptionCondition(
                    noise_level=0.0,
                    missing_rate=0.0,
                    available_modalities=combination,
                ),
            )
        )

    return conditions


# =========================================================
# Model creation
# =========================================================

def create_model(
    device,
):

    model = TemporalMVAE(
        modality_dim=20,
        latent_dim=LATENT_DIM,
        state_dim=40,
        encoder_hidden_dim=64,
        transition_hidden_dim=64,
    ).to(device)

    return model


# =========================================================
# Checkpoint loading
# =========================================================

def load_checkpoint(
    checkpoint_path: Path,
    device,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            f"{checkpoint_path} does not contain "
            "'model_state_dict'."
        )

    model = create_model(
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
# Dataset
# =========================================================

def make_test_loader(
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
# Evaluation
# =========================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
):

    model.eval()

    current_sse = 0.0
    current_sae = 0.0
    current_elements = 0

    horizon_sse = {
        horizon: 0.0
        for horizon
        in FORECAST_HORIZONS
    }

    horizon_sae = {
        horizon: 0.0
        for horizon
        in FORECAST_HORIZONS
    }

    horizon_elements = {
        horizon: 0
        for horizon
        in FORECAST_HORIZONS
    }

    forecast_sse = 0.0
    forecast_sae = 0.0
    forecast_elements = 0

    for batch in loader:

        modalities = {
            name:
            batch[
                "modalities"
            ][name].to(
                device,
                non_blocking=True,
            )
            for name in MODALITY_NAMES
        }

        masks = {
            name:
            batch[
                "masks"
            ][name].to(
                device,
                non_blocking=True,
            )
            for name in MODALITY_NAMES
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

        # Deterministic evaluation:
        # z_t = posterior mean.
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
            deterministic=True,
        )

        reconstruction = (
            output[
                "reconstruction"
            ]
        )

        forecast = (
            output[
                "forecast"
            ]
        )

        # -------------------------------------------------
        # Current state
        # -------------------------------------------------

        current_error = (
            reconstruction
            - current_truth
        )

        current_sse += (
            torch.sum(
                current_error ** 2
            ).item()
        )

        current_sae += (
            torch.sum(
                torch.abs(
                    current_error
                )
            ).item()
        )

        current_elements += (
            current_error.numel()
        )

        # -------------------------------------------------
        # All forecasts
        # -------------------------------------------------

        forecast_error = (
            forecast
            - future_truth
        )

        forecast_sse += (
            torch.sum(
                forecast_error ** 2
            ).item()
        )

        forecast_sae += (
            torch.sum(
                torch.abs(
                    forecast_error
                )
            ).item()
        )

        forecast_elements += (
            forecast_error.numel()
        )

        # -------------------------------------------------
        # Individual horizons
        # -------------------------------------------------

        for index, horizon in enumerate(
            FORECAST_HORIZONS
        ):

            error = (
                forecast[
                    :,
                    index,
                    :
                ]
                -
                future_truth[
                    :,
                    index,
                    :
                ]
            )

            horizon_sse[
                horizon
            ] += (
                torch.sum(
                    error ** 2
                ).item()
            )

            horizon_sae[
                horizon
            ] += (
                torch.sum(
                    torch.abs(
                        error
                    )
                ).item()
            )

            horizon_elements[
                horizon
            ] += (
                error.numel()
            )

    # =====================================================
    # Aggregate
    # =====================================================

    current_mse = (
        current_sse
        / current_elements
    )

    current_mae = (
        current_sae
        / current_elements
    )

    forecast_mse = (
        forecast_sse
        / forecast_elements
    )

    forecast_mae = (
        forecast_sae
        / forecast_elements
    )

    metrics = {
        "current_mse": (
            current_mse
        ),
        "current_rmse": (
            math.sqrt(
                current_mse
            )
        ),
        "current_mae": (
            current_mae
        ),
        "forecast_mse": (
            forecast_mse
        ),
        "forecast_rmse": (
            math.sqrt(
                forecast_mse
            )
        ),
        "forecast_mae": (
            forecast_mae
        ),
    }

    for horizon in FORECAST_HORIZONS:

        mse = (
            horizon_sse[
                horizon
            ]
            /
            horizon_elements[
                horizon
            ]
        )

        mae = (
            horizon_sae[
                horizon
            ]
            /
            horizon_elements[
                horizon
            ]
        )

        metrics[
            f"h{horizon}_mse"
        ] = mse

        metrics[
            f"h{horizon}_rmse"
        ] = math.sqrt(
            mse
        )

        metrics[
            f"h{horizon}_mae"
        ] = mae

    return metrics


# =========================================================
# Printing
# =========================================================

def print_result(
    model_name,
    family,
    condition_name,
    metrics,
):

    print(
        f"{model_name:14s}"
        f" | {family:11s}"
        f" | {condition_name:18s}"
        f" | current MSE "
        f"{metrics['current_mse']:.6f}"
        f" | RMSE "
        f"{metrics['current_rmse']:.6f}"
        f" | h1 "
        f"{metrics['h1_mse']:.6f}"
        f" | h5 "
        f"{metrics['h5_mse']:.6f}"
        f" | h10 "
        f"{metrics['h10_mse']:.6f}"
    )


# =========================================================
# Save results
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
        "model",
        "family",
        "condition",
        "noise_level",
        "missing_rate",
        "available_modalities",
        "current_mse",
        "current_rmse",
        "current_mae",
        "forecast_mse",
        "forecast_rmse",
        "forecast_mae",
        "h1_mse",
        "h1_rmse",
        "h1_mae",
        "h5_mse",
        "h5_rmse",
        "h5_mae",
        "h10_mse",
        "h10_rmse",
        "h10_mae",
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

        for row in results:

            writer.writerow(
                {
                    key:
                    row[key]
                    for key in fieldnames
                }
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

    # =====================================================
    # Load untouched data
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
    # Generate CLEAN observations
    # =====================================================

    print()
    print(
        "Generating clean observations..."
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

    # =====================================================
    # Frozen TRAIN statistics
    # =====================================================

    print(
        "Computing training-only statistics..."
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

    # =====================================================
    # Normalize TEST truth
    # =====================================================

    test_truth = normalize_states(
        test_truth_raw,
        state_stats["mean"],
        state_stats["std"],
    )

    # =====================================================
    # Load models
    # =====================================================

    print()
    print(
        "Loading clean-trained checkpoint..."
    )

    (
        clean_model,
        clean_checkpoint,
    ) = load_checkpoint(
        CLEAN_CHECKPOINT,
        device,
    )

    print(
        "Clean checkpoint epoch:",
        clean_checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "Loading robust-trained checkpoint..."
    )

    (
        robust_model,
        robust_checkpoint,
    ) = load_checkpoint(
        ROBUST_CHECKPOINT,
        device,
    )

    print(
        "Robust checkpoint epoch:",
        robust_checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    # =====================================================
    # Sanity checks
    # =====================================================

    for checkpoint_name, checkpoint in (
        (
            "clean",
            clean_checkpoint,
        ),
        (
            "robust",
            robust_checkpoint,
        ),
    ):

        checkpoint_latent = (
            checkpoint.get(
                "latent_dim"
            )
        )

        if (
            checkpoint_latent
            is not None
            and int(
                checkpoint_latent
            )
            != LATENT_DIM
        ):

            raise RuntimeError(
                f"{checkpoint_name} checkpoint "
                f"latent_dim={checkpoint_latent}, "
                f"expected {LATENT_DIM}."
            )

        checkpoint_history = (
            checkpoint.get(
                "history_length"
            )
        )

        if (
            checkpoint_history
            is not None
            and int(
                checkpoint_history
            )
            != HISTORY_LENGTH
        ):

            raise RuntimeError(
                f"{checkpoint_name} checkpoint "
                f"history_length="
                f"{checkpoint_history}, "
                f"expected {HISTORY_LENGTH}."
            )

    # =====================================================
    # Evaluation
    # =====================================================

    conditions = (
        build_conditions()
    )

    results = []

    models = (
        (
            "clean_trained",
            clean_model,
        ),
        (
            "robust_trained",
            robust_model,
        ),
    )

    print()
    print(
        "=" * 120
    )

    print(
        "TEMPORAL MVAE ROBUSTNESS EVALUATION"
    )

    print(
        "=" * 120
    )

    # Fixed condition-specific seeds.
    #
    # Both models receive EXACTLY the same corrupted
    # observations for every condition.

    for condition_index, (
        family,
        condition_name,
        condition,
    ) in enumerate(
        conditions
    ):

        condition_seed = (
            SEED
            + 1_000_000
            + condition_index * 10_000
        )

        test_modalities, test_masks = (
            corrupt_modalities(
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
        )

        (
            test_dataset,
            test_loader,
        ) = make_test_loader(
            test_modalities,
            test_masks,
            test_truth,
        )

        for (
            model_name,
            model,
        ) in models:

            metrics = evaluate_model(
                model=model,
                loader=test_loader,
                device=device,
            )

            row = {
                "model": (
                    model_name
                ),
                "family": (
                    family
                ),
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
            }

            row.update(
                {
                    key: float(value)
                    for key, value
                    in metrics.items()
                }
            )

            results.append(
                row
            )

            print_result(
                model_name=(
                    model_name
                ),
                family=family,
                condition_name=(
                    condition_name
                ),
                metrics=metrics,
            )

        print(
            "-" * 120
        )

    # =====================================================
    # Save
    # =====================================================

    save_results(
        results
    )

    # =====================================================
    # Direct comparison summary
    # =====================================================

    print()
    print(
        "=" * 120
    )

    print(
        "CURRENT-STATE MSE COMPARISON"
    )

    print(
        "=" * 120
    )

    clean_lookup = {
        (
            row["family"],
            row["condition"],
        ):
        row
        for row in results
        if row["model"]
        == "clean_trained"
    }

    robust_lookup = {
        (
            row["family"],
            row["condition"],
        ):
        row
        for row in results
        if row["model"]
        == "robust_trained"
    }

    for (
        family,
        condition_name,
        _,
    ) in conditions:

        key = (
            family,
            condition_name,
        )

        clean_mse = (
            clean_lookup[
                key
            ][
                "current_mse"
            ]
        )

        robust_mse = (
            robust_lookup[
                key
            ][
                "current_mse"
            ]
        )

        difference = (
            robust_mse
            - clean_mse
        )

        if robust_mse < clean_mse:

            winner = (
                "ROBUST"
            )

            improvement = (
                (
                    clean_mse
                    - robust_mse
                )
                /
                clean_mse
                * 100.0
            )

        elif clean_mse < robust_mse:

            winner = (
                "CLEAN"
            )

            improvement = (
                (
                    robust_mse
                    - clean_mse
                )
                /
                robust_mse
                * 100.0
            )

        else:

            winner = (
                "TIE"
            )

            improvement = 0.0

        print(
            f"{family:11s}"
            f" | {condition_name:18s}"
            f" | clean "
            f"{clean_mse:.6f}"
            f" | robust "
            f"{robust_mse:.6f}"
            f" | diff "
            f"{difference:+.6f}"
            f" | winner "
            f"{winner}"
            f" | advantage "
            f"{improvement:.2f}%"
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