from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    LorenzMultimodalDataset,
)
from src.data.normalization import (
    normalize_modalities,
    normalize_states,
    denormalize_states,
)
from src.data.observation_operators import (
    generate_clean_modalities,
)
from src.models.vae import VAE


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATCH_SIZE = 512
NUM_WORKERS = 0

DATA_PATH = Path(
    "data/raw/lorenz96_clean.npz"
)

CHECKPOINT_PATH = Path(
    "experiments/checkpoints/vae_clean_best.pt"
)

RESULTS_DIR = Path(
    "experiments/results"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "vae_clean_test.json"
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def compute_global_r2(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    sse = np.sum(
        (target - prediction) ** 2
    )

    target_mean = np.mean(
        target
    )

    sst = np.sum(
        (target - target_mean) ** 2
    )

    if sst <= 0.0:
        raise ValueError(
            "Cannot compute R2 because "
            "target variance is zero."
        )

    return float(
        1.0 - sse / sst
    )


def compute_feature_r2(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:

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

    sse = np.sum(
        (target - prediction) ** 2,
        axis=0,
    )

    sst = np.sum(
        (target - target_mean) ** 2,
        axis=0,
    )

    result = np.full(
        target.shape[-1],
        np.nan,
        dtype=np.float64,
    )

    valid = sst > 0.0

    result[valid] = (
        1.0
        - sse[valid]
        / sst[valid]
    )

    return result


def compute_kl_per_sample(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    KL averaged over latent dimensions,
    returning one value per sample.

    Shape:
        mu/logvar: [B, latent_dim]
        output:    [B]
    """

    kl = -0.5 * (
        1.0
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    return kl.mean(
        dim=1
    )


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
    # Checkpoint
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
        "Checkpoint validation "
        "reconstruction MSE:",
        checkpoint[
            "val_reconstruction"
        ],
    )

    print(
        "Checkpoint beta:",
        checkpoint["beta"],
    )

    observation_stats = checkpoint[
        "observation_stats"
    ]

    state_stats = checkpoint[
        "state_stats"
    ]

    latent_dim = int(
        checkpoint[
            "latent_dim"
        ]
    )

    # -----------------------------------------------------
    # Untouched test truth
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
    ].astype(
        np.float32
    )

    print(
        "Raw test truth shape:",
        test_truth_raw.shape,
    )

    # -----------------------------------------------------
    # Clean observations
    #
    # 0% noise
    # 0% missingness
    # ABC available
    # -----------------------------------------------------

    test_clean = (
        generate_clean_modalities(
            test_truth_raw
        )
    )

    test_masks = {
        name: np.ones_like(
            values,
            dtype=np.float32,
        )
        for name, values
        in test_clean.items()
    }

    # -----------------------------------------------------
    # Reuse TRAIN normalization statistics.
    #
    # Absolutely no statistics are fitted on TEST.
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
    # Dataset
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

    model = VAE(
        input_dim=60,
        latent_dim=latent_dim,
        output_dim=40,
    ).to(
        device
    )

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
    #
    # IMPORTANT:
    #
    # We do NOT call model(observations) for the primary
    # reconstruction metric because forward() samples z.
    #
    # Instead:
    #
    #     mu, logvar = encode(y)
    #     prediction = decode(mu)
    #
    # This gives a deterministic posterior-mean latent
    # point estimate.
    # -----------------------------------------------------

    predictions_normalized = []
    targets_normalized = []

    latent_means = []
    latent_logvars = []

    kl_values = []

    print()
    print(
        "Running deterministic "
        "posterior-mean test inference..."
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
            [
                a,
                b,
                c,
            ],
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

        mu, logvar = model.encode(
            observations
        )

        # ---------------------------------------------
        # Deterministic point estimate:
        #
        # z = posterior mean
        # ---------------------------------------------

        prediction = model.decode(
            mu
        )

        batch_kl = (
            compute_kl_per_sample(
                mu,
                logvar,
            )
        )

        predictions_normalized.append(
            prediction.cpu().numpy()
        )

        targets_normalized.append(
            target.cpu().numpy()
        )

        latent_means.append(
            mu.cpu().numpy()
        )

        latent_logvars.append(
            logvar.cpu().numpy()
        )

        kl_values.append(
            batch_kl.cpu().numpy()
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

    latent_means = np.concatenate(
        latent_means,
        axis=0,
    )

    latent_logvars = np.concatenate(
        latent_logvars,
        axis=0,
    )

    kl_values = np.concatenate(
        kl_values,
        axis=0,
    )

    print(
        "Prediction shape:",
        predictions_normalized.shape,
    )

    print(
        "Target shape:",
        targets_normalized.shape,
    )

    print(
        "Latent mean shape:",
        latent_means.shape,
    )

    # -----------------------------------------------------
    # Safety checks
    # -----------------------------------------------------

    arrays_to_check = {
        "predictions": (
            predictions_normalized
        ),
        "targets": (
            targets_normalized
        ),
        "latent means": (
            latent_means
        ),
        "latent logvars": (
            latent_logvars
        ),
        "KL": (
            kl_values
        ),
    }

    for name, array in (
        arrays_to_check.items()
    ):

        if not np.all(
            np.isfinite(array)
        ):

            raise ValueError(
                f"{name} contains "
                f"NaN or Inf."
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
    # Original Lorenz-96 scale
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
    # Latent / KL statistics
    # -----------------------------------------------------

    mean_kl = float(
        np.mean(
            kl_values
        )
    )

    std_kl = float(
        np.std(
            kl_values
        )
    )

    mean_abs_mu = float(
        np.mean(
            np.abs(
                latent_means
            )
        )
    )

    mean_logvar = float(
        np.mean(
            latent_logvars
        )
    )

    mean_posterior_std = float(
        np.mean(
            np.exp(
                0.5
                * latent_logvars
            )
        )
    )

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    print()
    print(
        "=" * 64
    )

    print(
        "VAE CLEAN TEST RESULTS"
    )

    print(
        "=" * 64
    )

    print(
        f"Best training epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation reconstruction MSE "
        f"(training log): "
        f"{checkpoint['val_reconstruction']:.6f}"
    )

    print(
        f"Beta: "
        f"{checkpoint['beta']:.5f}"
    )

    print()

    print(
        "POINT ESTIMATE: decoder(mu)"
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
        "LATENT STATISTICS"
    )

    print(
        f"Mean KL/sample:       "
        f"{mean_kl:.6f}"
    )

    print(
        f"Std KL/sample:        "
        f"{std_kl:.6f}"
    )

    print(
        f"Mean |mu|:            "
        f"{mean_abs_mu:.6f}"
    )

    print(
        f"Mean log-variance:     "
        f"{mean_logvar:.6f}"
    )

    print(
        f"Mean posterior std:    "
        f"{mean_posterior_std:.6f}"
    )

    print()

    print(
        f"Test samples:          "
        f"{len(test_dataset)}"
    )

    print(
        f"Parameters:            "
        f"{parameter_count}"
    )

    print(
        "=" * 64
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "model": (
            "static_variational_latent_state_estimator"
        ),
        "experiment": (
            "clean_ABC_current_state_reconstruction"
        ),
        "point_estimate": (
            "decoder_posterior_mean"
        ),
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "best_epoch": int(
            checkpoint["epoch"]
        ),
        "validation_reconstruction_mse_training_log": (
            float(
                checkpoint[
                    "val_reconstruction"
                ]
            )
        ),
        "beta": float(
            checkpoint["beta"]
        ),
        "beta_max": float(
            checkpoint[
                "beta_max"
            ]
        ),
        "kl_warmup_epochs": int(
            checkpoint[
                "kl_warmup_epochs"
            ]
        ),
        "latent_dim": int(
            latent_dim
        ),
        "test_samples": int(
            len(test_dataset)
        ),
        "parameter_count": int(
            parameter_count
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
        "latent_statistics": {
            "mean_kl_per_sample": (
                mean_kl
            ),
            "std_kl_per_sample": (
                std_kl
            ),
            "mean_absolute_mu": (
                mean_abs_mu
            ),
            "mean_logvar": (
                mean_logvar
            ),
            "mean_posterior_std": (
                mean_posterior_std
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