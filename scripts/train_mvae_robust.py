from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    LorenzMultimodalDataset,
)

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
    TrainingCorruptionSampler,
    corrupt_modalities,
    clean_condition,
)

from src.models.robust_mvae import (
    RobustMVAE,
)

from src.training.losses import (
    vae_loss,
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
    "mvae_robust_best.pt"
)

LOG_PATH = Path(
    "experiments/logs/"
    "mvae_robust_training.json"
)

BATCH_SIZE = 256

LEARNING_RATE = 1e-3

MAX_EPOCHS = 500

PATIENCE = 10

LATENT_DIM = 16

HISTORY_LENGTH = 20

FORECAST_HORIZONS = (
    1,
    5,
    10,
)

BETA_MAX = 0.01

KL_WARMUP_EPOCHS = 50


# =========================================================
# Reproducibility
# =========================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# =========================================================
# Beta schedule
# =========================================================

def beta_for_epoch(
    epoch: int,
) -> float:

    if epoch >= KL_WARMUP_EPOCHS:

        return BETA_MAX

    return (
        BETA_MAX
        * epoch
        / KL_WARMUP_EPOCHS
    )


# =========================================================
# Dataset
# =========================================================

def make_dataset(
    modalities,
    masks,
    truth,
):

    return LorenzMultimodalDataset(
        modalities=modalities,
        masks=masks,
        truth=truth,
        history_length=(
            HISTORY_LENGTH
        ),
        forecast_horizons=(
            FORECAST_HORIZONS
        ),
    )


# =========================================================
# Prepare current timestep only
# =========================================================

def prepare_batch(
    batch,
    device,
):

    """
    Static control:

    Dataset still provides a 20-step history so that
    it uses exactly the same valid current timestamps
    as the Temporal MVAE.

    However, this model deliberately sees ONLY the
    final timestamp t.
    """

    modalities = {
        name:
        batch[
            "modalities"
        ][name][
            :,
            -1,
            :
        ].to(
            device,
            non_blocking=True,
        )
        for name in (
            "A",
            "B",
            "C",
        )
    }

    masks = {
        name:
        batch[
            "masks"
        ][name][
            :,
            -1,
            :
        ].to(
            device,
            non_blocking=True,
        )
        for name in (
            "A",
            "B",
            "C",
        )
    }

    target = (
        batch[
            "current_truth"
        ].to(
            device,
            non_blocking=True,
        )
    )

    return (
        modalities,
        masks,
        target,
    )


# =========================================================
# Training
# =========================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    beta,
):

    model.train()

    total_loss = 0.0

    total_reconstruction = 0.0

    total_kl = 0.0

    total_samples = 0

    for batch in loader:

        (
            modalities,
            masks,
            target,
        ) = prepare_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            modalities=modalities,
            masks=masks,
            deterministic=False,
        )

        losses = vae_loss(
            prediction=(
                output[
                    "reconstruction"
                ]
            ),
            target=target,
            mu=output[
                "mu"
            ],
            logvar=output[
                "logvar"
            ],
            beta=beta,
        )

        losses[
            "loss"
        ].backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        batch_size = (
            target.shape[0]
        )

        total_samples += (
            batch_size
        )

        total_loss += (
            losses[
                "loss"
            ].item()
            * batch_size
        )

        total_reconstruction += (
            losses[
                "reconstruction"
            ].item()
            * batch_size
        )

        total_kl += (
            losses[
                "kl"
            ].item()
            * batch_size
        )

    return {
        "total": (
            total_loss
            / total_samples
        ),
        "reconstruction": (
            total_reconstruction
            / total_samples
        ),
        "kl": (
            total_kl
            / total_samples
        ),
    }


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    beta,
):

    model.eval()

    total_loss = 0.0

    total_reconstruction = 0.0

    total_kl = 0.0

    total_samples = 0

    for batch in loader:

        (
            modalities,
            masks,
            target,
        ) = prepare_batch(
            batch,
            device,
        )

        output = model(
            modalities=modalities,
            masks=masks,
            deterministic=True,
        )

        losses = vae_loss(
            prediction=(
                output[
                    "reconstruction"
                ]
            ),
            target=target,
            mu=output[
                "mu"
            ],
            logvar=output[
                "logvar"
            ],
            beta=beta,
        )

        batch_size = (
            target.shape[0]
        )

        total_samples += (
            batch_size
        )

        total_loss += (
            losses[
                "loss"
            ].item()
            * batch_size
        )

        total_reconstruction += (
            losses[
                "reconstruction"
            ].item()
            * batch_size
        )

        total_kl += (
            losses[
                "kl"
            ].item()
            * batch_size
        )

    return {
        "total": (
            total_loss
            / total_samples
        ),
        "reconstruction": (
            total_reconstruction
            / total_samples
        ),
        "kl": (
            total_kl
            / total_samples
        ),
    }


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
    # Load clean Lorenz-96
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

    val_truth_raw = (
        data[
            "val_states"
        ].astype(
            np.float32
        )
    )

    # =====================================================
    # Generate clean observations
    # =====================================================

    print(
        "Generating clean modalities..."
    )

    clean_train_modalities = (
        generate_clean_modalities(
            train_truth_raw
        )
    )

    clean_val_modalities = (
        generate_clean_modalities(
            val_truth_raw
        )
    )

    # =====================================================
    # TRAIN-only statistics
    # =====================================================

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

    train_truth = normalize_states(
        train_truth_raw,
        state_stats[
            "mean"
        ],
        state_stats[
            "std"
        ],
    )

    val_truth = normalize_states(
        val_truth_raw,
        state_stats[
            "mean"
        ],
        state_stats[
            "std"
        ],
    )

    # =====================================================
    # Fixed clean validation
    # =====================================================

    val_modalities, val_masks = (
        corrupt_modalities(
            clean_modalities=(
                clean_val_modalities
            ),
            training_feature_stds=(
                training_feature_stds
            ),
            normalization_stats=(
                observation_stats
            ),
            condition=(
                clean_condition()
            ),
            seed=(
                SEED
                + 999999
            ),
        )
    )

    val_dataset = make_dataset(
        modalities=(
            val_modalities
        ),
        masks=val_masks,
        truth=val_truth,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    # =====================================================
    # Model
    # =====================================================

    model = RobustMVAE(
        modality_dim=20,
        latent_dim=(
            LATENT_DIM
        ),
        output_dim=40,
    ).to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "Model parameters:",
        parameter_count,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    sampler = (
        TrainingCorruptionSampler(
            seed=SEED
        )
    )

    # =====================================================
    # Training
    # =====================================================

    print()
    print(
        "Starting ROBUST STATIC MVAE training..."
    )

    print(
        "Input history used by model: "
        "CURRENT TIMESTAMP ONLY"
    )

    print(
        "Noise levels:",
        (
            0.00,
            0.05,
            0.10,
            0.20,
        ),
    )

    print(
        "Missingness:",
        (
            0.00,
            0.10,
            0.30,
            0.50,
        ),
    )

    print(
        "Modality combinations:",
        (
            "ABC",
            "AB",
            "AC",
            "BC",
            "A",
            "B",
            "C",
        ),
    )

    print(
        "Checkpoint validation: "
        "fixed clean ABC"
    )

    print(
        "Checkpoint criterion: "
        "validation reconstruction MSE"
    )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation = float(
        "inf"
    )

    best_epoch = 0

    epochs_without_improvement = 0

    logs = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        beta = beta_for_epoch(
            epoch
        )

        # =================================================
        # EXACT SAME style of corruption sampling used by
        # robust Temporal MVAE:
        # one condition per epoch.
        # =================================================

        condition = (
            sampler.sample()
        )

        (
            train_modalities,
            train_masks,
        ) = corrupt_modalities(
            clean_modalities=(
                clean_train_modalities
            ),
            training_feature_stds=(
                training_feature_stds
            ),
            normalization_stats=(
                observation_stats
            ),
            condition=condition,
            seed=(
                SEED
                + epoch * 1000
            ),
        )

        train_dataset = make_dataset(
            modalities=(
                train_modalities
            ),
            masks=train_masks,
            truth=train_truth,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=(
                torch.cuda.is_available()
            ),
        )

        # =================================================
        # Train
        # =================================================

        train_metrics = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                beta=beta,
            )
        )

        # =================================================
        # Fixed clean validation
        # =================================================

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            beta=beta,
        )

        validation_task = (
            val_metrics[
                "reconstruction"
            ]
        )

        improved = False

        # Same rule as temporal:
        # don't select/early-stop until KL warm-up ends.
        if (
            epoch
            >= KL_WARMUP_EPOCHS
            and validation_task
            < best_validation
        ):

            best_validation = (
                validation_task
            )

            best_epoch = (
                epoch
            )

            epochs_without_improvement = (
                0
            )

            improved = True

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "best_validation_reconstruction": (
                        float(
                            best_validation
                        )
                    ),
                    "condition": {
                        "noise_level": (
                            condition.noise_level
                        ),
                        "missing_rate": (
                            condition.missing_rate
                        ),
                        "available_modalities": (
                            condition.available_modalities
                        ),
                    },
                    "latent_dim": (
                        LATENT_DIM
                    ),
                    "history_used": (
                        1
                    ),
                    "dataset_history_length": (
                        HISTORY_LENGTH
                    ),
                    "beta_max": (
                        BETA_MAX
                    ),
                    "parameter_count": (
                        parameter_count
                    ),
                    "observation_stats": (
                        observation_stats
                    ),
                    "state_stats": (
                        state_stats
                    ),
                    "seed": (
                        SEED
                    ),
                },
                CHECKPOINT_PATH,
            )

        elif (
            epoch
            >= KL_WARMUP_EPOCHS
        ):

            epochs_without_improvement += (
                1
            )

        condition_text = (
            f"noise="
            f"{int(condition.noise_level * 100)}% "
            f"missing="
            f"{int(condition.missing_rate * 100)}% "
            f"modalities="
            f"{''.join(condition.available_modalities)}"
        )

        log_entry = {
            "epoch": epoch,
            "beta": beta,
            "condition": {
                "noise": (
                    condition.noise_level
                ),
                "missing": (
                    condition.missing_rate
                ),
                "modalities": (
                    "".join(
                        condition.available_modalities
                    )
                ),
            },
            "train": (
                train_metrics
            ),
            "validation": (
                val_metrics
            ),
            "best_validation": (
                best_validation
            ),
        }

        logs.append(
            log_entry
        )

        with open(
            LOG_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                logs,
                file,
                indent=4,
            )

        marker = (
            "  <-- best"
            if improved
            else ""
        )

        print(
            f"Epoch {epoch:03d}"
            f" | {condition_text}"
            f" | beta {beta:.5f}"
            f" | train total "
            f"{train_metrics['total']:.6f}"
            f" | train recon "
            f"{train_metrics['reconstruction']:.6f}"
            f" | val recon "
            f"{val_metrics['reconstruction']:.6f}"
            f" | val KL "
            f"{val_metrics['kl']:.6f}"
            f"{marker}"
        )

        if (
            epoch
            >= KL_WARMUP_EPOCHS
            and epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping."
            )

            break

    print()

    print(
        "Robust Static MVAE training finished."
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation reconstruction:",
        best_validation,
    )

    print(
        "Best checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Training log:",
        LOG_PATH,
    )


if __name__ == "__main__":
    main()