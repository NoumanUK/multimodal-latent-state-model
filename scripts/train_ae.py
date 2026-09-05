from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import LorenzMultimodalDataset
from src.data.normalization import (
    compute_normalization_stats,
    compute_state_normalization_stats,
    normalize_modalities,
    normalize_states,
)
from src.data.observation_operators import (
    generate_clean_modalities,
)
from src.models.autoencoder import Autoencoder


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

DATA_PATH = Path(
    "data/raw/lorenz96_clean.npz"
)

CHECKPOINT_DIR = Path(
    "experiments/checkpoints"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "ae_clean_best.pt"
)


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_seed(
    seed: int,
) -> None:
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------

def prepare_datasets():
    print("Loading Lorenz-96 data...")

    data = np.load(
        DATA_PATH
    )

    train_truth_raw = data[
        "train_states"
    ].astype(np.float32)

    val_truth_raw = data[
        "val_states"
    ].astype(np.float32)

    # -----------------------------------------------------
    # Generate CLEAN observations.
    #
    # AE-clean baseline:
    #
    #   0% noise
    #   0% missingness
    #   A+B+C available
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
    # Observation normalization:
    # fit TRAIN ONLY.
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
    # Hidden-state normalization:
    # fit TRAIN ONLY.
    # -----------------------------------------------------

    state_stats = (
        compute_state_normalization_stats(
            train_truth_raw
        )
    )

    train_truth = normalize_states(
        train_truth_raw,
        state_stats["mean"],
        state_stats["std"],
    )

    val_truth = normalize_states(
        val_truth_raw,
        state_stats["mean"],
        state_stats["std"],
    )

    # -----------------------------------------------------
    # Lazy datasets.
    #
    # AE consumes only the final timestamp of each
    # temporal window.
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
    device: torch.device,
):
    """
    Deterministic AE baseline uses only current time t.

    Dataset modality shape:
        (batch, history, 20)

    Current observation:
        [:, -1, :]

    A + B + C:
        20 + 20 + 20 = 60 features.
    """

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

    return observations, target


# ---------------------------------------------------------
# Training epoch
# ---------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
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

        reconstruction = output[
            "reconstruction"
        ]

        loss = F.mse_loss(
            reconstruction,
            target,
        )

        loss.backward()

        optimizer.step()

        batch_size = target.shape[0]

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

@torch.no_grad()
def validate(
    model,
    loader,
    device,
):
    model.eval()

    total_loss = 0.0
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

        reconstruction = output[
            "reconstruction"
        ]

        loss = F.mse_loss(
            reconstruction,
            target,
        )

        batch_size = target.shape[0]

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


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

    model = Autoencoder(
        input_dim=60,
        latent_dim=LATENT_DIM,
        output_dim=40,
    ).to(device)

    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        ),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -----------------------------------------------------
    # Checkpoint directory
    # -----------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Resume configuration
    # -----------------------------------------------------

    start_epoch = 1

    best_val_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    # -----------------------------------------------------
    # Resume from the BEST existing checkpoint.
    # -----------------------------------------------------

    if CHECKPOINT_PATH.exists():
        print()
        print(
            "Resuming from:",
            CHECKPOINT_PATH,
        )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        start_epoch = (
            int(
                checkpoint["epoch"]
            )
            + 1
        )

        best_val_loss = float(
            checkpoint["val_loss"]
        )

        print(
            "Resuming after epoch:",
            checkpoint["epoch"],
        )

        print(
            "Current best validation MSE:",
            best_val_loss,
        )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print()
    print(
        "Starting AE training..."
    )
    print()

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_loss = validate(
            model=model,
            loader=val_loader,
            device=device,
        )

        improved = (
            val_loss
            < best_val_loss
        )

        marker = ""

        if improved:
            best_val_loss = val_loss

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
                "train_loss": train_loss,
                "val_loss": val_loss,
                "latent_dim": LATENT_DIM,
                "seed": SEED,
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
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train MSE: "
            f"{train_loss:.6f} | "
            f"val MSE: "
            f"{val_loss:.6f}"
            f"{marker}"
        )

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        if (
            epochs_without_improvement
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
        "Training finished."
    )

    print(
        f"Best validation MSE: "
        f"{best_val_loss:.6f}"
    )

    print(
        "Best checkpoint:",
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()