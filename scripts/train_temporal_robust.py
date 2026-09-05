from __future__ import annotations

import json
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
    normalize_modalities,
    normalize_states,
)

from src.data.noise import (
    compute_training_feature_stds,
)

from src.data.corruption import (
    CorruptionCondition,
    TrainingCorruptionSampler,
    corrupt_modalities,
    clean_condition,
)

from src.models.temporal_mvae import TemporalMVAE

from src.training.losses import temporal_mvae_loss


SEED = 42

DATA_PATH = Path(
    "data/raw/lorenz96_clean.npz"
)

CHECKPOINT_PATH = Path(
    "experiments/checkpoints/"
    "temporal_mvae_robust_best.pt"
)

LOG_PATH = Path(
    "experiments/logs/"
    "temporal_mvae_robust_training.json"
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


def set_seed(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def make_dataset(
    modalities,
    masks,
    truth,
):

    return LorenzMultimodalDataset(
        modalities=modalities,
        masks=masks,
        truth=truth,
    )


def evaluate(
    model,
    loader,
    device,
    beta,
):

    model.eval()

    total_sum = 0.0
    current_sum = 0.0
    forecast_sum = 0.0
    kl_sum = 0.0

    h_sums = {
        1: 0.0,
        5: 0.0,
        10: 0.0,
    }

    sample_count = 0

    with torch.no_grad():

        for batch in loader:

            modalities = {
                name:
                batch["modalities"][name].to(
                    device
                )
                for name in (
                    "A",
                    "B",
                    "C",
                )
            }

            masks = {
                name:
                batch["masks"][name].to(
                    device
                )
                for name in (
                    "A",
                    "B",
                    "C",
                )
            }

            current_truth = (
                batch[
                    "current_truth"
                ].to(
                    device
                )
            )

            future_truth = (
                batch[
                    "future_truth"
                ].to(
                    device
                )
            )

            output = model(
                modalities,
                masks,
                deterministic=True,
            )

            loss_output = (
                temporal_mvae_loss(
                    reconstruction=(
                        output[
                            "reconstruction"
                        ]
                    ),
                    current_target=(
                        current_truth
                    ),
                    forecast=(
                        output[
                            "forecast"
                        ]
                    ),
                    future_target=(
                        future_truth
                    ),
                    mu=output["mu"],
                    logvar=output["logvar"],
                    beta=beta,
                    current_weight=(
                        CURRENT_WEIGHT
                    ),
                    forecast_weight=(
                        FORECAST_WEIGHT
                    ),
                )
            )

            batch_size = (
                current_truth.shape[0]
            )

            sample_count += (
                batch_size
            )

            total_sum += (
                loss_output[
                    "loss"
                ].item()
                * batch_size
            )

            current_sum += (
                loss_output[
                    "current"
                ].item()
                * batch_size
            )

            forecast_sum += (
                loss_output[
                    "forecast"
                ].item()
                * batch_size
            )

            kl_sum += (
                loss_output[
                    "kl"
                ].item()
                * batch_size
            )

            for index, horizon in enumerate(
                FORECAST_HORIZONS
            ):

                horizon_mse = (
                    torch.nn.functional.mse_loss(
                        output[
                            "forecast"
                        ][
                            :,
                            index,
                            :
                        ],
                        future_truth[
                            :,
                            index,
                            :
                        ],
                        reduction="mean",
                    )
                )

                h_sums[
                    horizon
                ] += (
                    horizon_mse.item()
                    * batch_size
                )

    return {
        "total": (
            total_sum
            / sample_count
        ),
        "current": (
            current_sum
            / sample_count
        ),
        "forecast": (
            forecast_sum
            / sample_count
        ),
        "kl": (
            kl_sum
            / sample_count
        ),
        "h1": (
            h_sums[1]
            / sample_count
        ),
        "h5": (
            h_sums[5]
            / sample_count
        ),
        "h10": (
            h_sums[10]
            / sample_count
        ),
    }


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

    print()
    print(
        "Loading Lorenz-96 data..."
    )

    data = np.load(
        DATA_PATH
    )

    # Change these two keys only if your NPZ
    # uses different names.
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

    # =====================================================
    # Fixed clean validation set
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
            condition=clean_condition(),
            seed=(
                SEED
                + 999999
            ),
        )
    )

    val_dataset = make_dataset(
        val_modalities,
        val_masks,
        val_truth,
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

    model = TemporalMVAE(
        latent_dim=LATENT_DIM
    ).to(
        device
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

    print()
    print(
        "Starting ROBUST Temporal MVAE training..."
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

    logs = []

    best_validation_task = float(
        "inf"
    )

    epochs_without_improvement = 0

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        beta = beta_for_epoch(
            epoch
        )

        # =================================================
        # Sample ONE corruption condition for this epoch
        # =================================================

        condition = (
            sampler.sample()
        )

        train_modalities, train_masks = (
            corrupt_modalities(
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
        )

        train_dataset = (
            make_dataset(
                train_modalities,
                train_masks,
                train_truth,
            )
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
        # Training
        # =================================================

        model.train()

        train_total = 0.0
        train_current = 0.0
        train_forecast = 0.0
        train_kl = 0.0

        sample_count = 0

        for batch in train_loader:

            modalities = {
                name:
                batch[
                    "modalities"
                ][name].to(
                    device
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
                    device
                )
                for name in (
                    "A",
                    "B",
                    "C",
                )
            }

            current_truth = (
                batch[
                    "current_truth"
                ].to(
                    device
                )
            )

            future_truth = (
                batch[
                    "future_truth"
                ].to(
                    device
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                modalities,
                masks,
                deterministic=False,
            )

            loss_output = (
                temporal_mvae_loss(
                    reconstruction=(
                        output[
                            "reconstruction"
                        ]
                    ),
                    current_target=(
                        current_truth
                    ),
                    forecast=(
                        output[
                            "forecast"
                        ]
                    ),
                    future_target=(
                        future_truth
                    ),
                    mu=output["mu"],
                    logvar=output["logvar"],
                    beta=beta,
                    current_weight=(
                        CURRENT_WEIGHT
                    ),
                    forecast_weight=(
                        FORECAST_WEIGHT
                    ),
                )
            )

            loss = (
                loss_output[
                    "loss"
                ]
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            batch_size = (
                current_truth.shape[0]
            )

            sample_count += (
                batch_size
            )

            train_total += (
                loss.item()
                * batch_size
            )

            train_current += (
                loss_output[
                    "current"
                ].item()
                * batch_size
            )

            train_forecast += (
                loss_output[
                    "forecast"
                ].item()
                * batch_size
            )

            train_kl += (
                loss_output[
                    "kl"
                ].item()
                * batch_size
            )

        train_metrics = {
            "total": (
                train_total
                / sample_count
            ),
            "current": (
                train_current
                / sample_count
            ),
            "forecast": (
                train_forecast
                / sample_count
            ),
            "kl": (
                train_kl
                / sample_count
            ),
        }

        # =================================================
        # Validation
        # =================================================

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            beta,
        )

        # Predictive task only.
        #
        # Do NOT use KL for model selection.

        validation_task = (
            val_metrics["current"]
            + val_metrics["forecast"]
        )

        improved = False

        if (
            epoch
            >= KL_WARMUP_EPOCHS
            and validation_task
            < best_validation_task
        ):

            best_validation_task = (
                validation_task
            )

            epochs_without_improvement = 0

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
                    "best_validation_task": (
                        best_validation_task
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
                    "history_length": (
                        HISTORY_LENGTH
                    ),
                    "forecast_horizons": (
                        FORECAST_HORIZONS
                    ),
                    "beta_max": (
                        BETA_MAX
                    ),
                },
                CHECKPOINT_PATH,
            )

        elif epoch >= KL_WARMUP_EPOCHS:

            epochs_without_improvement += 1

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
            "validation_task": (
                validation_task
            ),
            "best_validation_task": (
                best_validation_task
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

        condition_text = (
            f"noise={int(condition.noise_level * 100)}% "
            f"missing={int(condition.missing_rate * 100)}% "
            f"modalities={''.join(condition.available_modalities)}"
        )

        print(
            f"Epoch {epoch:03d} | "
            f"{condition_text} | "
            f"beta {beta:.5f} | "
            f"train total {train_metrics['total']:.6f} | "
            f"val current {val_metrics['current']:.6f} | "
            f"val forecast {val_metrics['forecast']:.6f} | "
            f"val task {validation_task:.6f} | "
            f"h1 {val_metrics['h1']:.6f} | "
            f"h5 {val_metrics['h5']:.6f} | "
            f"h10 {val_metrics['h10']:.6f}"
            + (
                "  <-- best"
                if improved
                else ""
            )
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
        "Robust Temporal MVAE training finished."
    )

    print(
        "Best checkpoint:",
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()