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

from src.models.robust_mvae import (
    RobustMVAE,
)

from src.models.temporal_mvae import (
    TemporalMVAE,
)


SEED = 42

DATA_PATH = Path(
    "data/raw/lorenz96_clean.npz"
)

STATIC_CHECKPOINT = Path(
    "experiments/checkpoints/"
    "mvae_robust_best.pt"
)

TEMPORAL_CHECKPOINT = Path(
    "experiments/checkpoints/"
    "temporal_mvae_robust_best.pt"
)

RESULT_DIR = Path(
    "experiments/results"
)

JSON_PATH = (
    RESULT_DIR
    / "static_vs_temporal_results.json"
)

CSV_PATH = (
    RESULT_DIR
    / "static_vs_temporal_results.csv"
)

BATCH_SIZE = 256

LATENT_DIM = 16

HISTORY_LENGTH = 20

FORECAST_HORIZONS = (
    1,
    5,
    10,
)


def set_seed(
    seed: int,
) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_conditions():

    conditions = []

    # Noise-only
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

    # Missingness-only
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

    # Modality dropout
    combinations = (
        ("A", "B", "C"),
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("A",),
        ("B",),
        ("C",),
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


def make_dataset(
    modalities,
    masks,
    truth,
):

    return LorenzMultimodalDataset(
        modalities=modalities,
        masks=masks,
        truth=truth,
        history_length=HISTORY_LENGTH,
        forecast_horizons=(
            FORECAST_HORIZONS
        ),
    )


def make_loader(
    dataset,
):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )


def load_static_model(
    device,
):

    checkpoint = torch.load(
        STATIC_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model = RobustMVAE(
        modality_dim=20,
        latent_dim=LATENT_DIM,
        output_dim=40,
    ).to(device)

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


def load_temporal_model(
    device,
):

    checkpoint = torch.load(
        TEMPORAL_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model = TemporalMVAE(
        modality_dim=20,
        latent_dim=LATENT_DIM,
        state_dim=40,
        encoder_hidden_dim=64,
        transition_hidden_dim=64,
    ).to(device)

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


@torch.no_grad()
def evaluate_static(
    model,
    loader,
    device,
):

    model.eval()

    sse = 0.0
    sae = 0.0
    elements = 0

    for batch in loader:

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

        output = model(
            modalities=modalities,
            masks=masks,
            deterministic=True,
        )

        prediction = (
            output[
                "reconstruction"
            ]
        )

        error = (
            prediction
            - target
        )

        sse += torch.sum(
            error ** 2
        ).item()

        sae += torch.sum(
            torch.abs(
                error
            )
        ).item()

        elements += (
            error.numel()
        )

    mse = (
        sse
        / elements
    )

    mae = (
        sae
        / elements
    )

    return {
        "mse": mse,
        "rmse": math.sqrt(
            mse
        ),
        "mae": mae,
    }


@torch.no_grad()
def evaluate_temporal(
    model,
    loader,
    device,
):

    model.eval()

    sse = 0.0
    sae = 0.0
    elements = 0

    for batch in loader:

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

        target = (
            batch[
                "current_truth"
            ].to(
                device,
                non_blocking=True,
            )
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
            deterministic=True,
        )

        prediction = (
            output[
                "reconstruction"
            ]
        )

        error = (
            prediction
            - target
        )

        sse += torch.sum(
            error ** 2
        ).item()

        sae += torch.sum(
            torch.abs(
                error
            )
        ).item()

        elements += (
            error.numel()
        )

    mse = (
        sse
        / elements
    )

    mae = (
        sae
        / elements
    )

    return {
        "mse": mse,
        "rmse": math.sqrt(
            mse
        ),
        "mae": mae,
    }


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
        "family",
        "condition",
        "noise_level",
        "missing_rate",
        "available_modalities",
        "static_mse",
        "static_rmse",
        "static_mae",
        "temporal_mse",
        "temporal_rmse",
        "temporal_mae",
        "winner",
        "temporal_improvement_percent",
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
            writer.writerow(row)


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

    print()
    print(
        "Generating clean modalities..."
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

    test_truth = normalize_states(
        test_truth_raw,
        state_stats[
            "mean"
        ],
        state_stats[
            "std"
        ],
    )

    print()
    print(
        "Loading robust static checkpoint..."
    )

    (
        static_model,
        static_checkpoint,
    ) = load_static_model(
        device
    )

    print(
        "Static checkpoint epoch:",
        static_checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "Loading robust temporal checkpoint..."
    )

    (
        temporal_model,
        temporal_checkpoint,
    ) = load_temporal_model(
        device
    )

    print(
        "Temporal checkpoint epoch:",
        temporal_checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    conditions = (
        build_conditions()
    )

    results = []

    print()
    print(
        "=" * 130
    )

    print(
        "ROBUST STATIC MVAE VS ROBUST TEMPORAL MVAE"
    )

    print(
        "=" * 130
    )

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

        (
            test_modalities,
            test_masks,
        ) = corrupt_modalities(
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

        dataset = make_dataset(
            modalities=(
                test_modalities
            ),
            masks=test_masks,
            truth=test_truth,
        )

        loader = make_loader(
            dataset
        )

        static_metrics = (
            evaluate_static(
                model=static_model,
                loader=loader,
                device=device,
            )
        )

        temporal_metrics = (
            evaluate_temporal(
                model=temporal_model,
                loader=loader,
                device=device,
            )
        )

        static_mse = (
            static_metrics[
                "mse"
            ]
        )

        temporal_mse = (
            temporal_metrics[
                "mse"
            ]
        )

        if temporal_mse < static_mse:

            winner = (
                "TEMPORAL"
            )

            temporal_improvement = (
                (
                    static_mse
                    - temporal_mse
                )
                /
                static_mse
                * 100.0
            )

        elif static_mse < temporal_mse:

            winner = (
                "STATIC"
            )

            temporal_improvement = (
                -(
                    (
                        temporal_mse
                        - static_mse
                    )
                    /
                    temporal_mse
                    * 100.0
                )
            )

        else:

            winner = (
                "TIE"
            )

            temporal_improvement = (
                0.0
            )

        row = {
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
            "static_mse": float(
                static_metrics[
                    "mse"
                ]
            ),
            "static_rmse": float(
                static_metrics[
                    "rmse"
                ]
            ),
            "static_mae": float(
                static_metrics[
                    "mae"
                ]
            ),
            "temporal_mse": float(
                temporal_metrics[
                    "mse"
                ]
            ),
            "temporal_rmse": float(
                temporal_metrics[
                    "rmse"
                ]
            ),
            "temporal_mae": float(
                temporal_metrics[
                    "mae"
                ]
            ),
            "winner": (
                winner
            ),
            "temporal_improvement_percent": float(
                temporal_improvement
            ),
        }

        results.append(
            row
        )

        print(
            f"{family:11s}"
            f" | {condition_name:18s}"
            f" | static "
            f"{static_mse:.6f}"
            f" | temporal "
            f"{temporal_mse:.6f}"
            f" | winner "
            f"{winner:8s}"
            f" | temporal improvement "
            f"{temporal_improvement:+.2f}%"
        )

    save_results(
        results
    )

    print()
    print(
        "=" * 130
    )

    print(
        "KEY TEMPORAL CONDITIONS"
    )

    print(
        "=" * 130
    )

    key_names = {
        "missing_30",
        "missing_50",
        "modalities_AB",
        "modalities_AC",
        "modalities_BC",
        "modalities_A",
        "modalities_B",
        "modalities_C",
    }

    key_rows = [
        row
        for row in results
        if row[
            "condition"
        ] in key_names
    ]

    temporal_wins = sum(
        1
        for row in key_rows
        if row[
            "winner"
        ] == "TEMPORAL"
    )

    print(
        f"Temporal wins on "
        f"{temporal_wins}"
        f"/{len(key_rows)} "
        f"key incomplete-observation conditions."
    )

    for row in key_rows:

        print(
            f"{row['condition']:18s}"
            f" | static "
            f"{row['static_mse']:.6f}"
            f" | temporal "
            f"{row['temporal_mse']:.6f}"
            f" | improvement "
            f"{row['temporal_improvement_percent']:+.2f}%"
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