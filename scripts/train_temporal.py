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
from src.data.normalization import (
    compute_normalization_stats,
    compute_state_normalization_stats,
    normalize_modalities,
    normalize_states,
)
from src.data.observation_operators import (
    generate_clean_modalities,
)
from src.models.temporal_mvae import (
    TemporalMVAE,
)
from src.training.losses import (
    temporal_mvae_loss,
)


# =========================================================
# Configuration
# =========================================================

SEED = 42

DATA_PATH = Path(
    "data/raw/lorenz96_clean.npz"
)

CHECKPOINT_DIR = Path(
    "experiments/checkpoints"
)

LOG_DIR = Path(
    "experiments/logs"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "temporal_mvae_clean_best.pt"
)

LOG_PATH = (
    LOG_DIR
    / "temporal_mvae_clean_training.json"
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

CURRENT_WEIGHT = 1.0

FORECAST_WEIGHT = 1.0

NUM_WORKERS = 0


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

def get_beta(
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
# Batch preparation
# =========================================================

def prepare_batch(
    batch,
    device,
):

    modalities = {
        name:
        batch[
            "modalities"
        ][name].to(
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
        ][name].to(
            device,
            non_blocking=True,
        )
        for name in (
            "A",
            "B",
            "C",
        )
    }

    current_target = (
        batch[
            "current_truth"
        ].to(
            device,
            non_blocking=True,
        )
    )

    future_target = (
        batch[
            "future_truth"
        ].to(
            device,
            non_blocking=True,
        )
    )

    return (
        modalities,
        masks,
        current_target,
        future_target,
    )


# =========================================================
# Training epoch
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

    total_current = 0.0

    total_forecast = 0.0

    total_kl = 0.0

    total_samples = 0

    for batch in loader:

        (
            modalities,
            masks,
            current_target,
            future_target,
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

        losses = temporal_mvae_loss(
            reconstruction=(
                output[
                    "reconstruction"
                ]
            ),
            current_target=(
                current_target
            ),
            forecast=(
                output[
                    "forecast"
                ]
            ),
            future_target=(
                future_target
            ),
            mu=output["mu"],
            logvar=output[
                "logvar"
            ],
            beta=beta,
            current_weight=(
                CURRENT_WEIGHT
            ),
            forecast_weight=(
                FORECAST_WEIGHT
            ),
        )

        losses[
            "loss"
        ].backward()

        # Recurrent forecasting can occasionally
        # produce large gradients early in training.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        batch_size = (
            current_target.shape[0]
        )

        total_samples += (
            batch_size
        )

        total_loss += (
            losses["loss"].item()
            * batch_size
        )

        total_current += (
            losses["current"].item()
            * batch_size
        )

        total_forecast += (
            losses["forecast"].item()
            * batch_size
        )

        total_kl += (
            losses["kl"].item()
            * batch_size
        )

    return {
        "total": (
            total_loss
            / total_samples
        ),
        "current": (
            total_current
            / total_samples
        ),
        "forecast": (
            total_forecast
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
def validate(
    model,
    loader,
    device,
    beta,
):

    model.eval()

    total_loss = 0.0

    total_current = 0.0

    total_forecast = 0.0

    total_kl = 0.0

    total_samples = 0

    # Individual forecast-horizon errors.
    horizon_sse = {
        horizon: 0.0
        for horizon
        in FORECAST_HORIZONS
    }

    horizon_elements = {
        horizon: 0
        for horizon
        in FORECAST_HORIZONS
    }

    for batch in loader:

        (
            modalities,
            masks,
            current_target,
            future_target,
        ) = prepare_batch(
            batch,
            device,
        )

        # Deterministic point estimate:
        #
        # z_t = posterior mean
        #
        # transition itself is deterministic.
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

        losses = temporal_mvae_loss(
            reconstruction=(
                output[
                    "reconstruction"
                ]
            ),
            current_target=(
                current_target
            ),
            forecast=(
                output[
                    "forecast"
                ]
            ),
            future_target=(
                future_target
            ),
            mu=output["mu"],
            logvar=output[
                "logvar"
            ],
            beta=beta,
            current_weight=(
                CURRENT_WEIGHT
            ),
            forecast_weight=(
                FORECAST_WEIGHT
            ),
        )

        batch_size = (
            current_target.shape[0]
        )

        total_samples += (
            batch_size
        )

        total_loss += (
            losses["loss"].item()
            * batch_size
        )

        total_current += (
            losses["current"].item()
            * batch_size
        )

        total_forecast += (
            losses["forecast"].item()
            * batch_size
        )

        total_kl += (
            losses["kl"].item()
            * batch_size
        )

        forecast = output[
            "forecast"
        ]

        for horizon_index, horizon in enumerate(
            FORECAST_HORIZONS
        ):

            error = (
                forecast[
                    :,
                    horizon_index,
                    :
                ]
                -
                future_target[
                    :,
                    horizon_index,
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

            horizon_elements[
                horizon
            ] += (
                error.numel()
            )

    horizon_mse = {
        horizon:
        (
            horizon_sse[
                horizon
            ]
            /
            horizon_elements[
                horizon
            ]
        )
        for horizon
        in FORECAST_HORIZONS
    }

    return {
        "total": (
            total_loss
            / total_samples
        ),
        "current": (
            total_current
            / total_samples
        ),
        "forecast": (
            total_forecast
            / total_samples
        ),
        "kl": (
            total_kl
            / total_samples
        ),
        "horizon_mse": (
            horizon_mse
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

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Clean multimodal observations
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
    # Observation normalization: fit TRAIN only
    # -----------------------------------------------------

    observation_stats = (
        compute_normalization_stats(
            train_clean
        )
    )

    train_masks = {
        name:
        np.ones_like(
            values,
            dtype=np.float32,
        )
        for name, values
        in train_clean.items()
    }

    val_masks = {
        name:
        np.ones_like(
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
    # State normalization: fit TRAIN only
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

    # -----------------------------------------------------
    # Lazy temporal datasets
    # -----------------------------------------------------

    train_dataset = (
        LorenzMultimodalDataset(
            modalities=(
                train_observations
            ),
            masks=train_masks,
            truth=train_truth,
            history_length=(
                HISTORY_LENGTH
            ),
            forecast_horizons=(
                FORECAST_HORIZONS
            ),
        )
    )

    val_dataset = (
        LorenzMultimodalDataset(
            modalities=(
                val_observations
            ),
            masks=val_masks,
            truth=val_truth,
            history_length=(
                HISTORY_LENGTH
            ),
            forecast_horizons=(
                FORECAST_HORIZONS
            ),
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = TemporalMVAE(
        modality_dim=20,
        latent_dim=LATENT_DIM,
        state_dim=40,
        encoder_hidden_dim=64,
        transition_hidden_dim=64,
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
    # Training
    # -----------------------------------------------------

    print()
    print(
        "Starting Temporal MVAE training..."
    )

    print(
        "History length:",
        HISTORY_LENGTH,
    )

    print(
        "Forecast horizons:",
        FORECAST_HORIZONS,
    )

    print(
        "Noise: 0%"
    )

    print(
        "Missingness: 0%"
    )

    print(
        "Available modalities: ABC"
    )

    print(
        "Current loss weight:",
        CURRENT_WEIGHT,
    )

    print(
        "Forecast loss weight:",
        FORECAST_WEIGHT,
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
        "Checkpoint criterion: "
        "validation total objective"
    )

    print(
        "Validation point estimate: "
        "decoder(posterior mean)"
    )

    best_score = float(
        "inf"
    )

    best_epoch = 0

    patience_counter = 0

    history = []

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

        # After warm-up beta is fixed, so this becomes
        # a stable objective for model selection.
        val_score = (
            val_metrics[
                "total"
            ]
        )

        improved = (
            val_score
            < best_score
        )

        # Do not early-stop or freeze a winner while
        # KL weighting itself is still changing.
        eligible_for_selection = (
            epoch
            >= KL_WARMUP_EPOCHS
        )

        if (
            eligible_for_selection
            and improved
        ):

            best_score = (
                val_score
            )

            best_epoch = (
                epoch
            )

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "val_total": float(
                        val_metrics[
                            "total"
                        ]
                    ),
                    "val_current": float(
                        val_metrics[
                            "current"
                        ]
                    ),
                    "val_forecast": float(
                        val_metrics[
                            "forecast"
                        ]
                    ),
                    "val_kl": float(
                        val_metrics[
                            "kl"
                        ]
                    ),
                    "val_horizon_mse": {
                        str(k): float(v)
                        for k, v
                        in val_metrics[
                            "horizon_mse"
                        ].items()
                    },
                    "beta": float(
                        beta
                    ),
                    "latent_dim": (
                        LATENT_DIM
                    ),
                    "history_length": (
                        HISTORY_LENGTH
                    ),
                    "forecast_horizons": (
                        FORECAST_HORIZONS
                    ),
                    "current_weight": (
                        CURRENT_WEIGHT
                    ),
                    "forecast_weight": (
                        FORECAST_WEIGHT
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
                    "seed": SEED,
                },
                CHECKPOINT_PATH,
            )

        elif eligible_for_selection:

            patience_counter += 1

        horizon_text = " | ".join(
            [
                (
                    f"h{h}: "
                    f"{val_metrics['horizon_mse'][h]:.6f}"
                )
                for h
                in FORECAST_HORIZONS
            ]
        )

        marker = (
            "  <-- best"
            if (
                eligible_for_selection
                and improved
            )
            else ""
        )

        print(
            f"Epoch {epoch:03d}"
            f" | beta: {beta:.5f}"
            f" | train total: "
            f"{train_metrics['total']:.6f}"
            f" | current: "
            f"{train_metrics['current']:.6f}"
            f" | forecast: "
            f"{train_metrics['forecast']:.6f}"
            f" | KL: "
            f"{train_metrics['kl']:.6f}"
            f" || val total: "
            f"{val_metrics['total']:.6f}"
            f" | current: "
            f"{val_metrics['current']:.6f}"
            f" | forecast: "
            f"{val_metrics['forecast']:.6f}"
            f" | KL: "
            f"{val_metrics['kl']:.6f}"
            f" | {horizon_text}"
            f"{marker}"
        )

        history.append(
            {
                "epoch": epoch,
                "beta": float(
                    beta
                ),
                "train": (
                    train_metrics
                ),
                "validation": (
                    val_metrics
                ),
            }
        )

        if (
            eligible_for_selection
            and patience_counter
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # -----------------------------------------------------
    # Save training history
    # -----------------------------------------------------

    with open(
        LOG_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )

    print()
    print(
        "Temporal MVAE training finished."
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation total objective:",
        best_score,
    )

    print(
        "Best checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Training history:",
        LOG_PATH,
    )


if __name__ == "__main__":
    main()