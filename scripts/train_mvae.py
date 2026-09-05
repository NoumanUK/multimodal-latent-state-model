from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    LorenzMultimodalDataset,
)
from src.data.normalization import (
    compute_normalization_stats,
    normalize_modalities,
    compute_state_normalization_stats,
    normalize_states,
)
from src.data.observation_operators import (
    generate_clean_modalities,
)
from src.models.mvae import MVAE
from src.training.losses import (
    vae_loss,
)


# =========================================================
# Configuration
# =========================================================

SEED = 42

BATCH_SIZE = 256
LEARNING_RATE = 1e-3

MAX_EPOCHS = 500
PATIENCE = 10

LATENT_DIM = 16

BETA_MAX = 0.01
KL_WARMUP_EPOCHS = 50

NUM_WORKERS = 0

DATA_PATH = Path(
    "data/raw/lorenz96_clean.npz"
)

CHECKPOINT_DIR = Path(
    "experiments/checkpoints"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "mvae_clean_best.pt"
)


# =========================================================
# Reproducibility
# =========================================================

def set_seed(
    seed: int,
):

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

def get_beta(
    epoch: int,
) -> float:
    """
    Linear KL warm-up.

    Epoch 1:
        beta = beta_max / warmup

    Epoch >= warmup:
        beta = beta_max
    """

    fraction = min(
        1.0,
        epoch
        / KL_WARMUP_EPOCHS,
    )

    return (
        BETA_MAX
        * fraction
    )


# =========================================================
# Prepare one batch
# =========================================================

def prepare_batch(
    batch,
    device,
):
    """
    MVAE is static in this experiment.

    The dataset supplies a temporal window, but this
    baseline deliberately uses ONLY the current
    observation at the final timestamp.

    A_t, B_t, C_t:
        each [B, 20]

    target X_t:
        [B, 40]
    """

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

    return (
        modalities,
        target,
    )


# =========================================================
# Training epoch
# =========================================================

def train_epoch(
    model,
    loader,
    optimizer,
    device,
    beta,
):

    model.train()

    total_loss_sum = 0.0
    reconstruction_sum = 0.0
    kl_sum = 0.0

    sample_count = 0

    for batch in loader:

        modalities, target = (
            prepare_batch(
                batch,
                device,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # -------------------------------------------------
        # Stochastic training.
        #
        # All three modalities are available for this
        # CLEAN ABC baseline.
        # -------------------------------------------------

        output = model(
            modalities,
            available_modalities=(
                "A",
                "B",
                "C",
            ),
            deterministic=False,
        )

        losses = vae_loss(
            prediction=output[
                "reconstruction"
            ],
            target=target,
            mu=output[
                "mu"
            ],
            logvar=output[
                "logvar"
            ],
            beta=beta,
        )

        loss = losses[
            "loss"
        ]

        loss.backward()

        optimizer.step()

        batch_size = (
            target.shape[0]
        )

        total_loss_sum += (
            losses[
                "loss"
            ].item()
            * batch_size
        )

        reconstruction_sum += (
            losses[
                "reconstruction"
            ].item()
            * batch_size
        )

        kl_sum += (
            losses[
                "kl"
            ].item()
            * batch_size
        )

        sample_count += (
            batch_size
        )

    return {
        "total": (
            total_loss_sum
            / sample_count
        ),
        "reconstruction": (
            reconstruction_sum
            / sample_count
        ),
        "kl": (
            kl_sum
            / sample_count
        ),
    }


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    beta,
):
    """
    IMPORTANT:

    Validation reconstruction uses the deterministic
    fused posterior mean.

        z = fused_mu

    This prevents random latent samples from affecting
    checkpoint selection.

    KL is still computed from the fused posterior.
    """

    model.eval()

    total_loss_sum = 0.0
    reconstruction_sum = 0.0
    kl_sum = 0.0

    sample_count = 0

    for batch in loader:

        modalities, target = (
            prepare_batch(
                batch,
                device,
            )
        )

        # -------------------------------------------------
        # deterministic=True means:
        #
        # z = fused posterior mean
        # -------------------------------------------------

        output = model(
            modalities,
            available_modalities=(
                "A",
                "B",
                "C",
            ),
            deterministic=True,
        )

        losses = vae_loss(
            prediction=output[
                "reconstruction"
            ],
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

        total_loss_sum += (
            losses[
                "loss"
            ].item()
            * batch_size
        )

        reconstruction_sum += (
            losses[
                "reconstruction"
            ].item()
            * batch_size
        )

        kl_sum += (
            losses[
                "kl"
            ].item()
            * batch_size
        )

        sample_count += (
            batch_size
        )

    return {
        "total": (
            total_loss_sum
            / sample_count
        ),
        "reconstruction": (
            reconstruction_sum
            / sample_count
        ),
        "kl": (
            kl_sum
            / sample_count
        ),
    }


# =========================================================
# Main
# =========================================================

def main():

    set_seed(
        SEED
    )

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
            torch.cuda.get_device_name(
                0
            ),
        )

    # -----------------------------------------------------
    # Load CLEAN Lorenz-96 truth.
    # -----------------------------------------------------

    print()
    print(
        "Loading Lorenz-96 data..."
    )

    data = np.load(
        DATA_PATH
    )

    train_truth_raw = data[
        "train_states"
    ].astype(
        np.float32
    )

    val_truth_raw = data[
        "val_states"
    ].astype(
        np.float32
    )

    # -----------------------------------------------------
    # Generate clean modalities.
    #
    # IMPORTANT:
    #
    # No noise.
    # No missingness.
    # All ABC modalities available.
    # -----------------------------------------------------

    train_clean = (
        generate_clean_modalities(
            train_truth_raw
        )
    )

    val_clean = (
        generate_clean_modalities(
            val_truth_raw
        )
    )

    # -----------------------------------------------------
    # Observation normalization.
    #
    # Statistics fitted ONLY on clean TRAIN observations.
    # -----------------------------------------------------

    observation_stats = (
        compute_normalization_stats(
            train_clean
        )
    )

    train_masks = {
        name: np.ones_like(
            values,
            dtype=np.float32,
        )
        for name, values
        in train_clean.items()
    }

    val_masks = {
        name: np.ones_like(
            values,
            dtype=np.float32,
        )
        for name, values
        in val_clean.items()
    }

    train_observations = (
        normalize_modalities(
            train_clean,
            observation_stats,
            train_masks,
        )
    )

    val_observations = (
        normalize_modalities(
            val_clean,
            observation_stats,
            val_masks,
        )
    )

    # -----------------------------------------------------
    # State normalization.
    #
    # Fit ONLY on TRAIN truth.
    # -----------------------------------------------------

    state_stats = (
        compute_state_normalization_stats(
            train_truth_raw
        )
    )

    train_truth = (
        normalize_states(
            train_truth_raw,
            state_stats[
                "mean"
            ],
            state_stats[
                "std"
            ],
        )
    )

    val_truth = (
        normalize_states(
            val_truth_raw,
            state_stats[
                "mean"
            ],
            state_stats[
                "std"
            ],
        )
    )

    # -----------------------------------------------------
    # Lazy datasets
    # -----------------------------------------------------

    train_dataset = (
        LorenzMultimodalDataset(
            modalities=train_observations,
            masks=train_masks,
            truth=train_truth,
        )
    )

    val_dataset = (
        LorenzMultimodalDataset(
            modalities=val_observations,
            masks=val_masks,
            truth=val_truth,
        )
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    # -----------------------------------------------------
    # Data loaders
    # -----------------------------------------------------

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
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
        latent_dim=LATENT_DIM,
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

    # -----------------------------------------------------
    # Training state
    # -----------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_reconstruction = (
        float("inf")
    )

    best_epoch = 0

    epochs_without_improvement = 0

    print()
    print(
        "Starting MVAE training..."
    )

    print(
        "Available modalities: ABC"
    )

    print(
        "Noise: 0%"
    )

    print(
        "Missingness: 0%"
    )

    print(
        "Beta max:",
        BETA_MAX,
    )

    print(
        "KL warm-up epochs:",
        KL_WARMUP_EPOCHS,
    )

    print(
        "Validation point estimate: "
        "decoder(fused_mu)"
    )

    print()

    # -----------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        beta = get_beta(
            epoch
        )

        train_metrics = (
            train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                beta=beta,
            )
        )

        val_metrics = (
            validate(
                model=model,
                loader=val_loader,
                device=device,
                beta=beta,
            )
        )

        improved = (
            val_metrics[
                "reconstruction"
            ]
            < best_val_reconstruction
        )

        if improved:

            best_val_reconstruction = (
                val_metrics[
                    "reconstruction"
                ]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            checkpoint = {
                "epoch": (
                    epoch
                ),
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "train_total": (
                    train_metrics[
                        "total"
                    ]
                ),
                "train_reconstruction": (
                    train_metrics[
                        "reconstruction"
                    ]
                ),
                "train_kl": (
                    train_metrics[
                        "kl"
                    ]
                ),
                "val_total": (
                    val_metrics[
                        "total"
                    ]
                ),
                "val_reconstruction": (
                    val_metrics[
                        "reconstruction"
                    ]
                ),
                "val_kl": (
                    val_metrics[
                        "kl"
                    ]
                ),
                "latent_dim": (
                    LATENT_DIM
                ),
                "seed": (
                    SEED
                ),
                "beta": (
                    beta
                ),
                "beta_max": (
                    BETA_MAX
                ),
                "kl_warmup_epochs": (
                    KL_WARMUP_EPOCHS
                ),
                "learning_rate": (
                    LEARNING_RATE
                ),
                "batch_size": (
                    BATCH_SIZE
                ),
                "parameter_count": (
                    parameter_count
                ),
                "available_modalities": (
                    (
                        "A",
                        "B",
                        "C",
                    )
                ),
                "noise_level": (
                    0.0
                ),
                "missing_rate": (
                    0.0
                ),
                "validation_point_estimate": (
                    "fused_posterior_mean"
                ),
                "observation_stats": (
                    observation_stats
                ),
                "state_stats": (
                    state_stats
                ),
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH,
            )

        else:

            # ---------------------------------------------
            # Do not use early stopping during KL warm-up.
            # ---------------------------------------------

            if epoch > (
                KL_WARMUP_EPOCHS
            ):

                epochs_without_improvement += 1

        marker = (
            "  <-- best"
            if improved
            else ""
        )

        print(
            f"Epoch {epoch:03d} | "
            f"beta: {beta:.5f} | "
            f"train total: "
            f"{train_metrics['total']:.6f} | "
            f"train recon: "
            f"{train_metrics['reconstruction']:.6f} | "
            f"train KL: "
            f"{train_metrics['kl']:.6f} | "
            f"val total: "
            f"{val_metrics['total']:.6f} | "
            f"val recon: "
            f"{val_metrics['reconstruction']:.6f} | "
            f"val KL: "
            f"{val_metrics['kl']:.6f}"
            f"{marker}"
        )

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        if (
            epoch
            > KL_WARMUP_EPOCHS
            and epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

    print()
    print(
        "MVAE training finished."
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation reconstruction "
        f"MSE: "
        f"{best_val_reconstruction:.6f}"
    )

    print(
        "Best checkpoint:",
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()