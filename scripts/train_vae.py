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
    compute_state_normalization_stats,
    normalize_modalities,
    normalize_states,
)
from src.data.observation_operators import (
    generate_clean_modalities,
)
from src.models.vae import VAE
from src.training.losses import (
    vae_loss,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 42

BATCH_SIZE = 256

LEARNING_RATE = 1e-3

MAX_EPOCHS = 500

PATIENCE = 10

LATENT_DIM = 16

NUM_WORKERS = 0

BETA_MAX = 0.01

KL_WARMUP_EPOCHS = 50

DATA_PATH = Path(
    "data/raw/lorenz96_clean.npz"
)

CHECKPOINT_DIR = Path(
    "experiments/checkpoints"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "vae_clean_best.pt"
)


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# Beta schedule
# ---------------------------------------------------------

def get_beta(
    epoch: int,
) -> float:
    """
    Linear KL warm-up.

    beta grows from approximately zero to
    BETA_MAX during the first
    KL_WARMUP_EPOCHS epochs.
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


# ---------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------

def prepare_datasets():

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
    # CLEAN observations only.
    #
    # 0% noise
    # 0% missingness
    # A+B+C available
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
    # TRAIN statistics only.
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
    # Truth normalization.
    #
    # TRAIN statistics only.
    # -----------------------------------------------------

    state_stats = (
        compute_state_normalization_stats(
            train_truth_raw
        )
    )

    train_truth = (
        normalize_states(
            train_truth_raw,
            state_stats["mean"],
            state_stats["std"],
        )
    )

    val_truth = (
        normalize_states(
            val_truth_raw,
            state_stats["mean"],
            state_stats["std"],
        )
    )

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

    return (
        train_dataset,
        val_dataset,
        observation_stats,
        state_stats,
    )


# ---------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------

def prepare_batch(
    batch,
    device,
):

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

    return (
        observations,
        target,
    )


# ---------------------------------------------------------
# Training epoch
# ---------------------------------------------------------

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

        observations, target = (
            prepare_batch(
                batch,
                device,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            observations
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

        total_samples += (
            batch_size
        )

    return {
        "loss": (
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


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

@torch.no_grad()
def validate(
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

        observations, target = (
            prepare_batch(
                batch,
                device,
            )
        )

        output = model(
            observations
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

        total_samples += (
            batch_size
        )

    return {
        "loss": (
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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

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
        f"Device: {device}"
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    (
        train_dataset,
        val_dataset,
        observation_stats,
        state_stats,
    ) = prepare_datasets()

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
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

    model = VAE(
        input_dim=60,
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

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Fresh VAE training
    #
    # IMPORTANT:
    # We deliberately do NOT resume from the AE.
    # -----------------------------------------------------

    best_val_reconstruction = float(
        "inf"
    )

    epochs_without_improvement = 0

    print()
    print(
        "Starting VAE training..."
    )

    print(
        "Beta max:",
        BETA_MAX,
    )

    print(
        "KL warm-up epochs:",
        KL_WARMUP_EPOCHS,
    )

    print()

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        beta = get_beta(
            epoch
        )

        train_metrics = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                beta=beta,
            )
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            beta=beta,
        )

        # -------------------------------------------------
        # Model selection:
        #
        # select according to validation reconstruction
        # MSE because state reconstruction is the task
        # metric being compared with deterministic AE.
        #
        # KL remains part of optimization.
        # -------------------------------------------------

        val_reconstruction = (
            val_metrics[
                "reconstruction"
            ]
        )

        improved = (
            val_reconstruction
            < best_val_reconstruction
        )

        marker = ""

        if improved:

            best_val_reconstruction = (
                val_reconstruction
            )

            epochs_without_improvement = 0

            marker = "  <-- best"

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "train_metrics": (
                    train_metrics
                ),
                "val_metrics": (
                    val_metrics
                ),
                "val_reconstruction": (
                    val_reconstruction
                ),
                "beta": beta,
                "beta_max": BETA_MAX,
                "kl_warmup_epochs": (
                    KL_WARMUP_EPOCHS
                ),
                "latent_dim": LATENT_DIM,
                "seed": SEED,
                "observation_stats": (
                    observation_stats
                ),
                "state_stats": (
                    state_stats
                ),
                "parameter_count": (
                    parameter_count
                ),
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH,
            )

        else:

            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"beta: {beta:.5f} | "
            f"train total: "
            f"{train_metrics['loss']:.6f} | "
            f"train recon: "
            f"{train_metrics['reconstruction']:.6f} | "
            f"train KL: "
            f"{train_metrics['kl']:.6f} | "
            f"val total: "
            f"{val_metrics['loss']:.6f} | "
            f"val recon: "
            f"{val_metrics['reconstruction']:.6f} | "
            f"val KL: "
            f"{val_metrics['kl']:.6f}"
            f"{marker}"
        )

        # -------------------------------------------------
        # Do not allow early stopping during KL warm-up.
        #
        # Otherwise the model could stop while beta is
        # still changing.
        # -------------------------------------------------

        if (
            epoch
            >= KL_WARMUP_EPOCHS
            and
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    print()
    print(
        "VAE training finished."
    )

    print(
        f"Best validation reconstruction MSE: "
        f"{best_val_reconstruction:.6f}"
    )

    print(
        "Best checkpoint:",
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()