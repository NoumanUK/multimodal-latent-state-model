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
from src.models.mvae import MVAE


# =========================================================
# Configuration
# =========================================================

BATCH_SIZE = 512
NUM_WORKERS = 0

DATA_PATH = Path(
    "data/raw/lorenz96_clean.npz"
)

CHECKPOINT_PATH = Path(
    "experiments/checkpoints/mvae_clean_best.pt"
)

RESULTS_DIR = Path(
    "experiments/results"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "mvae_clean_test.json"
)


# =========================================================
# Modality conditions
# =========================================================

MODALITY_CONDITIONS = {
    "ABC": (
        "A",
        "B",
        "C",
    ),
    "AB": (
        "A",
        "B",
    ),
    "AC": (
        "A",
        "C",
    ),
    "BC": (
        "B",
        "C",
    ),
    "A": (
        "A",
    ),
    "B": (
        "B",
    ),
    "C": (
        "C",
    ),
}


# =========================================================
# Metrics
# =========================================================

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

    valid = (
        sst > 0.0
    )

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
    KL averaged across latent dimensions.

    Returns:
        [batch]
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


# =========================================================
# Evaluate one modality condition
# =========================================================

@torch.no_grad()
def evaluate_condition(
    model,
    loader,
    device,
    available_modalities,
    state_stats,
):

    model.eval()

    predictions_normalized = []
    targets_normalized = []

    latent_means = []
    latent_logvars = []
    kl_values = []

    for batch in loader:

        modalities = {
            name: batch[
                "modalities"
            ][name][:, -1, :].to(
                device,
                non_blocking=True,
            )
            for name in (
                "A",
                "B",
                "C",
            )
        }

        target = batch[
            "current_truth"
        ].to(
            device,
            non_blocking=True,
        )

        # -------------------------------------------------
        # Deterministic posterior-mean inference.
        #
        # MVAE internally uses only the experts listed in
        # available_modalities.
        # -------------------------------------------------

        output = model(
            modalities,
            available_modalities=(
                available_modalities
            ),
            deterministic=True,
        )

        prediction = output[
            "reconstruction"
        ]

        mu = output[
            "mu"
        ]

        logvar = output[
            "logvar"
        ]

        kl = compute_kl_per_sample(
            mu,
            logvar,
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
            kl.cpu().numpy()
        )

    predictions_normalized = np.concatenate(
        predictions_normalized,
        axis=0,
    )

    targets_normalized = np.concatenate(
        targets_normalized,
        axis=0,
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

    # -----------------------------------------------------
    # Safety checks
    # -----------------------------------------------------

    arrays = {
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

    for name, array in arrays.items():

        if not np.all(
            np.isfinite(array)
        ):

            raise ValueError(
                f"{name} contains NaN "
                f"or Inf."
            )

    # -----------------------------------------------------
    # Normalized metrics
    # -----------------------------------------------------

    normalized_error = (
        predictions_normalized
        - targets_normalized
    )

    normalized_mse = float(
        np.mean(
            normalized_error ** 2
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
                normalized_error
            )
        )
    )

    # -----------------------------------------------------
    # Original Lorenz-96 units
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

    original_error = (
        predictions_original
        - targets_original
    )

    original_mse = float(
        np.mean(
            original_error ** 2
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
                original_error
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
    # Latent statistics
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

    return {
        "normalized_mse": (
            normalized_mse
        ),
        "normalized_rmse": (
            normalized_rmse
        ),
        "normalized_mae": (
            normalized_mae
        ),
        "original_mse": (
            original_mse
        ),
        "original_rmse": (
            original_rmse
        ),
        "original_mae": (
            original_mae
        ),
        "global_r2": (
            global_r2
        ),
        "mean_feature_r2": (
            mean_feature_r2
        ),
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
        "per_feature_r2": (
            feature_r2
            .astype(float)
            .tolist()
        ),
    }


# =========================================================
# Main
# =========================================================

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
    # Test truth
    # -----------------------------------------------------

    print()
    print(
        "Loading test data..."
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
    # Generate CLEAN observations.
    #
    # No observation noise.
    # No element missingness.
    #
    # Whole-modality subsets are controlled directly
    # through the MVAE expert selection.
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
    # Reuse TRAIN observation normalization statistics.
    # -----------------------------------------------------

    test_observations = (
        normalize_modalities(
            test_clean,
            observation_stats,
            test_masks,
        )
    )

    # -----------------------------------------------------
    # Reuse TRAIN state normalization statistics.
    # -----------------------------------------------------

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

    model = MVAE(
        modality_dim=20,
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

    print()
    print(
        "Evaluating modality subsets..."
    )

    # -----------------------------------------------------
    # Evaluate all conditions
    # -----------------------------------------------------

    results_by_condition = {}

    for condition_name, modalities in (
        MODALITY_CONDITIONS.items()
    ):

        print()
        print(
            f"Condition: {condition_name}"
        )

        print(
            "Available modalities:",
            modalities,
        )

        metrics = evaluate_condition(
            model=model,
            loader=test_loader,
            device=device,
            available_modalities=modalities,
            state_stats=state_stats,
        )

        results_by_condition[
            condition_name
        ] = metrics

        print(
            f"Normalized MSE:  "
            f"{metrics['normalized_mse']:.6f}"
        )

        print(
            f"Normalized RMSE: "
            f"{metrics['normalized_rmse']:.6f}"
        )

        print(
            f"Original RMSE:   "
            f"{metrics['original_rmse']:.6f}"
        )

        print(
            f"Original MAE:    "
            f"{metrics['original_mae']:.6f}"
        )

        print(
            f"Global R2:       "
            f"{metrics['global_r2']:.6f}"
        )

        print(
            f"Mean KL:         "
            f"{metrics['mean_kl_per_sample']:.6f}"
        )

        print(
            f"Posterior std:   "
            f"{metrics['mean_posterior_std']:.6f}"
        )

    # -----------------------------------------------------
    # Summary table
    # -----------------------------------------------------

    print()
    print(
        "=" * 86
    )

    print(
        "MVAE CLEAN TEST — MODALITY SUBSET SUMMARY"
    )

    print(
        "=" * 86
    )

    print(
        f"{'Condition':<12}"
        f"{'Norm MSE':>12}"
        f"{'Norm RMSE':>12}"
        f"{'Orig RMSE':>12}"
        f"{'R2':>12}"
        f"{'KL':>12}"
    )

    print(
        "-" * 86
    )

    for condition_name in (
        MODALITY_CONDITIONS
    ):

        metrics = (
            results_by_condition[
                condition_name
            ]
        )

        print(
            f"{condition_name:<12}"
            f"{metrics['normalized_mse']:>12.6f}"
            f"{metrics['normalized_rmse']:>12.6f}"
            f"{metrics['original_rmse']:>12.6f}"
            f"{metrics['global_r2']:>12.6f}"
            f"{metrics['mean_kl_per_sample']:>12.6f}"
        )

    print(
        "=" * 86
    )

    # -----------------------------------------------------
    # Missing-modality degradation relative to ABC
    # -----------------------------------------------------

    abc_mse = (
        results_by_condition[
            "ABC"
        ][
            "normalized_mse"
        ]
    )

    degradation = {}

    print()
    print(
        "DEGRADATION RELATIVE TO ABC"
    )

    for condition_name in (
        MODALITY_CONDITIONS
    ):

        condition_mse = (
            results_by_condition[
                condition_name
            ][
                "normalized_mse"
            ]
        )

        percentage = (
            (
                condition_mse
                - abc_mse
            )
            / abc_mse
            * 100.0
        )

        degradation[
            condition_name
        ] = float(
            percentage
        )

        print(
            f"{condition_name:<4}: "
            f"{percentage:+.2f}% MSE"
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "model": (
            "static_multimodal_variational_latent_state_estimator"
        ),
        "fusion": (
            "gaussian_product_of_experts"
        ),
        "experiment": (
            "clean_zero_shot_modality_subset_reconstruction"
        ),
        "training_modalities": [
            "A",
            "B",
            "C",
        ],
        "training_modality_dropout": (
            False
        ),
        "interpretation": (
            "subset results are zero-shot missing-modality inference "
            "because the model was trained only with ABC available"
        ),
        "point_estimate": (
            "decoder_fused_posterior_mean"
        ),
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
        "best_epoch": int(
            checkpoint["epoch"]
        ),
        "validation_reconstruction_mse": float(
            checkpoint[
                "val_reconstruction"
            ]
        ),
        "beta": float(
            checkpoint["beta"]
        ),
        "latent_dim": (
            latent_dim
        ),
        "parameter_count": (
            parameter_count
        ),
        "test_samples": int(
            len(test_dataset)
        ),
        "conditions": (
            results_by_condition
        ),
        "normalized_mse_degradation_percent_vs_ABC": (
            degradation
        ),
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