from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import LorenzMultimodalDataset
from src.data.normalization import (
    normalize_modalities,
    normalize_states,
    denormalize_states,
)
from src.data.observation_operators import (
    generate_clean_modalities,
)
from src.models.autoencoder import Autoencoder


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATCH_SIZE = 512

NUM_WORKERS = 0

DATA_PATH = Path(
    "data/raw/lorenz96_clean.npz"
)

CHECKPOINT_PATH = Path(
    "experiments/checkpoints/ae_clean_best.pt"
)

RESULTS_DIR = Path(
    "experiments/results"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "ae_clean_test.json"
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def compute_global_r2(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Global R² over all test samples and all
    40 state dimensions.
    """

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    residual_sum_squares = np.sum(
        (target - prediction) ** 2
    )

    total_sum_squares = np.sum(
        (
            target
            - np.mean(target)
        ) ** 2
    )

    if total_sum_squares <= 0.0:
        raise ValueError(
            "Cannot compute R² because target "
            "variance is zero."
        )

    return float(
        1.0
        - residual_sum_squares
        / total_sum_squares
    )


def compute_feature_r2(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Compute one R² score for each of the
    40 Lorenz-96 state variables.
    """

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    target_mean = np.mean(
        target,
        axis=0,
        keepdims=True,
    )

    residual_sum_squares = np.sum(
        (target - prediction) ** 2,
        axis=0,
    )

    total_sum_squares = np.sum(
        (target - target_mean) ** 2,
        axis=0,
    )

    r2 = np.full(
        target.shape[-1],
        np.nan,
        dtype=np.float64,
    )

    valid = (
        total_sum_squares > 0.0
    )

    r2[valid] = (
        1.0
        - residual_sum_squares[valid]
        / total_sum_squares[valid]
    )

    return r2


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

@torch.no_grad()
def main():

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # -----------------------------------------------------
    # Check checkpoint
    # -----------------------------------------------------

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    print(
        "Loading checkpoint:",
        CHECKPOINT_PATH,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    print(
        "Best checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Checkpoint validation MSE:",
        checkpoint["val_loss"],
    )

    observation_stats = checkpoint[
        "observation_stats"
    ]

    state_stats = checkpoint[
        "state_stats"
    ]

    latent_dim = int(
        checkpoint.get(
            "latent_dim",
            16,
        )
    )

    # -----------------------------------------------------
    # Load untouched TEST truth
    # -----------------------------------------------------

    print()
    print(
        "Loading untouched test data..."
    )

    data = np.load(
        DATA_PATH
    )

    test_truth_raw = data[
        "test_states"
    ].astype(np.float32)

    print(
        "Raw test truth shape:",
        test_truth_raw.shape,
    )

    # -----------------------------------------------------
    # Generate clean test observations
    # -----------------------------------------------------

    test_clean = (
        generate_clean_modalities(
            test_truth_raw
        )
    )

    # Clean experiment:
    # every observation is available.
    test_masks = {
        name: np.ones_like(
            values,
            dtype=np.float32,
        )
        for name, values
        in test_clean.items()
    }

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # Reuse TRAINING normalization statistics stored
    # inside the checkpoint.
    #
    # We do NOT fit anything using test data.
    # -----------------------------------------------------

    test_observations = (
        normalize_modalities(
            test_clean,
            observation_stats,
            test_masks,
        )
    )

    test_truth_normalized = (
        normalize_states(
            test_truth_raw,
            state_stats["mean"],
            state_stats["std"],
        )
    )

    # -----------------------------------------------------
    # Construct same lazy dataset definition used
    # during training.
    # -----------------------------------------------------

    test_dataset = (
        LorenzMultimodalDataset(
            modalities=test_observations,
            masks=test_masks,
            truth=test_truth_normalized,
        )
    )

    print(
        "Test samples:",
        len(test_dataset),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = Autoencoder(
        input_dim=60,
        latent_dim=latent_dim,
        output_dim=40,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "Model parameters:",
        parameter_count,
    )

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    predictions_normalized = []
    targets_normalized = []

    print()
    print(
        "Running test inference..."
    )

    for batch in test_loader:

        a = batch[
            "modalities"
        ]["A"][:, -1, :]

        b = batch[
            "modalities"
        ]["B"][:, -1, :]

        c = batch[
            "modalities"
        ]["C"][:, -1, :]

        observations = torch.cat(
            [a, b, c],
            dim=-1,
        ).to(
            device,
            non_blocking=True,
        )

        target = batch[
            "current_truth"
        ].to(
            device,
            non_blocking=True,
        )

        output = model(
            observations
        )

        prediction = output[
            "reconstruction"
        ]

        predictions_normalized.append(
            prediction.cpu().numpy()
        )

        targets_normalized.append(
            target.cpu().numpy()
        )

    predictions_normalized = (
        np.concatenate(
            predictions_normalized,
            axis=0,
        )
    )

    targets_normalized = (
        np.concatenate(
            targets_normalized,
            axis=0,
        )
    )

    print(
        "Prediction shape:",
        predictions_normalized.shape,
    )

    print(
        "Target shape:",
        targets_normalized.shape,
    )

    if not np.all(
        np.isfinite(
            predictions_normalized
        )
    ):
        raise ValueError(
            "Predictions contain NaN or Inf."
        )

    # -----------------------------------------------------
    # Normalized-space metrics
    # -----------------------------------------------------

    normalized_errors = (
        predictions_normalized
        - targets_normalized
    )

    normalized_mse = float(
        np.mean(
            normalized_errors ** 2
        )
    )

    normalized_rmse = float(
        np.sqrt(
            normalized_mse
        )
    )

    normalized_mae = float(
        np.mean(
            np.abs(
                normalized_errors
            )
        )
    )

    # -----------------------------------------------------
    # Convert predictions + targets back to original
    # Lorenz-96 scale.
    # -----------------------------------------------------

    predictions_original = (
        denormalize_states(
            predictions_normalized,
            state_stats["mean"],
            state_stats["std"],
        )
    )

    targets_original = (
        denormalize_states(
            targets_normalized,
            state_stats["mean"],
            state_stats["std"],
        )
    )

    # -----------------------------------------------------
    # Original-scale metrics
    # -----------------------------------------------------

    original_errors = (
        predictions_original
        - targets_original
    )

    original_mse = float(
        np.mean(
            original_errors ** 2
        )
    )

    original_rmse = float(
        np.sqrt(
            original_mse
        )
    )

    original_mae = float(
        np.mean(
            np.abs(
                original_errors
            )
        )
    )

    global_r2 = compute_global_r2(
        predictions_original,
        targets_original,
    )

    feature_r2 = compute_feature_r2(
        predictions_original,
        targets_original,
    )

    mean_feature_r2 = float(
        np.nanmean(
            feature_r2
        )
    )

    # -----------------------------------------------------
    # Per-state metrics
    # -----------------------------------------------------

    feature_rmse = np.sqrt(
        np.mean(
            original_errors ** 2,
            axis=0,
        )
    )

    feature_mae = np.mean(
        np.abs(
            original_errors
        ),
        axis=0,
    )

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "AE CLEAN TEST RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Best training epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation MSE: "
        f"{checkpoint['val_loss']:.6f}"
    )

    print()

    print(
        f"Normalized MSE:  "
        f"{normalized_mse:.6f}"
    )

    print(
        f"Normalized RMSE: "
        f"{normalized_rmse:.6f}"
    )

    print(
        f"Normalized MAE:  "
        f"{normalized_mae:.6f}"
    )

    print()

    print(
        f"Original MSE:    "
        f"{original_mse:.6f}"
    )

    print(
        f"Original RMSE:   "
        f"{original_rmse:.6f}"
    )

    print(
        f"Original MAE:    "
        f"{original_mae:.6f}"
    )

    print()

    print(
        f"Global R2:       "
        f"{global_r2:.6f}"
    )

    print(
        f"Mean feature R2: "
        f"{mean_feature_r2:.6f}"
    )

    print()

    print(
        f"Test samples:     "
        f"{len(test_dataset)}"
    )

    print(
        f"Parameters:       "
        f"{parameter_count}"
    )

    print(
        "=" * 60
    )

    # -----------------------------------------------------
    # Save reproducible results
    # -----------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "model": (
            "deterministic_static_encoder_decoder"
        ),
        "experiment": (
            "clean_ABC_current_state_reconstruction"
        ),
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "best_epoch": int(
            checkpoint["epoch"]
        ),
        "validation_mse": float(
            checkpoint["val_loss"]
        ),
        "test_samples": int(
            len(test_dataset)
        ),
        "parameter_count": int(
            parameter_count
        ),
        "latent_dim": int(
            latent_dim
        ),
        "normalized_metrics": {
            "mse": normalized_mse,
            "rmse": normalized_rmse,
            "mae": normalized_mae,
        },
        "original_scale_metrics": {
            "mse": original_mse,
            "rmse": original_rmse,
            "mae": original_mae,
            "global_r2": global_r2,
            "mean_feature_r2": (
                mean_feature_r2
            ),
        },
        "per_feature_original_scale": {
            "rmse": (
                feature_rmse
                .astype(float)
                .tolist()
            ),
            "mae": (
                feature_mae
                .astype(float)
                .tolist()
            ),
            "r2": (
                feature_r2
                .astype(float)
                .tolist()
            ),
        },
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print()
    print(
        "Results saved to:",
        RESULTS_PATH,
    )


if __name__ == "__main__":
    main()