from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------
# Allow imports from the project root when this script
# is executed directly.
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.data.lorenz96 import generate_dataset


# =========================================================
# FROZEN DATASET SPECIFICATION
# =========================================================

DIMENSION = 40

FORCING = 8.0

DT = 0.01

SAMPLE_EVERY = 5

EFFECTIVE_DT = (
    DT * SAMPLE_EVERY
)

BURN_IN_STEPS = 1000

NUM_TRAJECTORIES = 100

NUM_SAVED_STEPS = 2000

TRAIN_TRAJECTORIES = 70

VAL_TRAJECTORIES = 15

TEST_TRAJECTORIES = 15

PERTURBATION_STD = 0.1

BASE_SEED = 42


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "lorenz96_clean.npz"
)


def validate_configuration() -> None:
    """
    Validate the frozen dataset specification before
    generating anything.
    """
    split_total = (
        TRAIN_TRAJECTORIES
        + VAL_TRAJECTORIES
        + TEST_TRAJECTORIES
    )

    if split_total != NUM_TRAJECTORIES:
        raise ValueError(
            "Train/validation/test trajectory counts "
            "must sum to NUM_TRAJECTORIES."
        )


def print_configuration() -> None:
    """
    Display the exact experimental specification.
    """
    print()
    print("=" * 60)
    print("LORENZ-96 DATASET GENERATION")
    print("=" * 60)

    print(
        f"Dimension N:             "
        f"{DIMENSION}"
    )

    print(
        f"Forcing F:               "
        f"{FORCING}"
    )

    print(
        f"RK4 integration dt:      "
        f"{DT}"
    )

    print(
        f"Sample every:            "
        f"{SAMPLE_EVERY} integration steps"
    )

    print(
        f"Effective saved dt:      "
        f"{EFFECTIVE_DT}"
    )

    print(
        f"Burn-in steps:           "
        f"{BURN_IN_STEPS}"
    )

    print(
        f"Number trajectories:     "
        f"{NUM_TRAJECTORIES}"
    )

    print(
        f"Saved states/trajectory: "
        f"{NUM_SAVED_STEPS}"
    )

    print(
        f"Train trajectories:      "
        f"{TRAIN_TRAJECTORIES}"
    )

    print(
        f"Validation trajectories: "
        f"{VAL_TRAJECTORIES}"
    )

    print(
        f"Test trajectories:       "
        f"{TEST_TRAJECTORIES}"
    )

    print(
        f"Initial perturbation SD: "
        f"{PERTURBATION_STD}"
    )

    print(
        f"Base random seed:        "
        f"{BASE_SEED}"
    )

    print("=" * 60)
    print()


def main() -> None:

    validate_configuration()

    print_configuration()

    print(
        "Generating clean Lorenz-96 trajectories..."
    )

    states = generate_dataset(
        num_trajectories=NUM_TRAJECTORIES,
        dimension=DIMENSION,
        forcing=FORCING,
        dt=DT,
        burn_in_steps=BURN_IN_STEPS,
        num_saved_steps=NUM_SAVED_STEPS,
        sample_every=SAMPLE_EVERY,
        perturbation_std=PERTURBATION_STD,
        base_seed=BASE_SEED,
    )

    # -----------------------------------------------------
    # Split by complete trajectories.
    # -----------------------------------------------------

    train_end = TRAIN_TRAJECTORIES

    val_end = (
        TRAIN_TRAJECTORIES
        + VAL_TRAJECTORIES
    )

    train_states = states[
        :train_end
    ]

    val_states = states[
        train_end:val_end
    ]

    test_states = states[
        val_end:
    ]

    # -----------------------------------------------------
    # Sanity checks.
    # -----------------------------------------------------

    expected_train_shape = (
        TRAIN_TRAJECTORIES,
        NUM_SAVED_STEPS,
        DIMENSION,
    )

    expected_val_shape = (
        VAL_TRAJECTORIES,
        NUM_SAVED_STEPS,
        DIMENSION,
    )

    expected_test_shape = (
        TEST_TRAJECTORIES,
        NUM_SAVED_STEPS,
        DIMENSION,
    )

    assert (
        train_states.shape
        == expected_train_shape
    )

    assert (
        val_states.shape
        == expected_val_shape
    )

    assert (
        test_states.shape
        == expected_test_shape
    )

    if not np.isfinite(states).all():
        raise RuntimeError(
            "Dataset contains NaN or infinite values."
        )

    # -----------------------------------------------------
    # Create output directory if necessary.
    # -----------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Save dataset and metadata.
    # -----------------------------------------------------

    np.savez_compressed(
        OUTPUT_PATH,

        train_states=train_states,
        val_states=val_states,
        test_states=test_states,

        dimension=np.int32(
            DIMENSION
        ),

        forcing=np.float32(
            FORCING
        ),

        dt=np.float32(
            DT
        ),

        sample_every=np.int32(
            SAMPLE_EVERY
        ),

        effective_dt=np.float32(
            EFFECTIVE_DT
        ),

        burn_in_steps=np.int32(
            BURN_IN_STEPS
        ),

        num_trajectories=np.int32(
            NUM_TRAJECTORIES
        ),

        num_saved_steps=np.int32(
            NUM_SAVED_STEPS
        ),

        train_trajectories=np.int32(
            TRAIN_TRAJECTORIES
        ),

        val_trajectories=np.int32(
            VAL_TRAJECTORIES
        ),

        test_trajectories=np.int32(
            TEST_TRAJECTORIES
        ),

        perturbation_std=np.float32(
            PERTURBATION_STD
        ),

        base_seed=np.int32(
            BASE_SEED
        ),
    )

    # -----------------------------------------------------
    # Final report.
    # -----------------------------------------------------

    print()
    print("Dataset generated successfully.")
    print()

    print(
        "Train shape:",
        train_states.shape,
    )

    print(
        "Validation shape:",
        val_states.shape,
    )

    print(
        "Test shape:",
        test_states.shape,
    )

    print()

    print(
        "Overall minimum:",
        float(states.min()),
    )

    print(
        "Overall maximum:",
        float(states.max()),
    )

    print(
        "Overall mean:",
        float(states.mean()),
    )

    print(
        "Overall standard deviation:",
        float(states.std()),
    )

    print()

    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()