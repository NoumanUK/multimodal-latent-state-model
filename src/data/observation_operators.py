from __future__ import annotations

import numpy as np


STATE_DIM = 40
MODALITY_DIM = 20


def _validate_states(states: np.ndarray) -> np.ndarray:
    """
    Validate Lorenz-96 state data.

    Expected final dimension:
        (..., 40)

    Examples:
        (40,)
        (2000, 40)
        (70, 2000, 40)
    """
    states = np.asarray(states)

    if states.ndim < 1:
        raise ValueError("states must have at least one dimension.")

    if states.shape[-1] != STATE_DIM:
        raise ValueError(
            f"Expected final dimension {STATE_DIM}, "
            f"but received shape {states.shape}."
        )

    if not np.all(np.isfinite(states)):
        raise ValueError(
            "states contains NaN or infinite values."
        )

    return states


def modality_a(states: np.ndarray) -> np.ndarray:
    """
    Modality A: sparse direct observations.

    Observe alternating Lorenz-96 state components:

        x1, x3, x5, ..., x39

    In zero-based Python indexing:

        0, 2, 4, ..., 38

    Input:
        (..., 40)

    Output:
        (..., 20)
    """
    states = _validate_states(states)

    observations = states[..., 0::2]

    return observations.astype(np.float32, copy=False)


def modality_b(states: np.ndarray) -> np.ndarray:
    """
    Modality B: shifted local linear mixtures.

    Measurements are:

        (x2  + x3)  / sqrt(2)
        (x4  + x5)  / sqrt(2)
        ...
        (x38 + x39) / sqrt(2)
        (x40 + x1)  / sqrt(2)

    This produces a non-orthogonal linear observation
    of neighbouring Lorenz-96 state variables.

    Input:
        (..., 40)

    Output:
        (..., 20)
    """
    states = _validate_states(states)

    left_indices = np.arange(
        1,
        STATE_DIM,
        2,
        dtype=np.int64,
    )

    right_indices = (
        left_indices + 1
    ) % STATE_DIM

    left = states[..., left_indices]
    right = states[..., right_indices]

    observations = (
        left + right
    ) / np.sqrt(2.0)

    return observations.astype(np.float32, copy=False)


def nonlinear_observation(
    x: np.ndarray,
    gamma: int = 3,
) -> np.ndarray:
    """
    Nonlinear scalar observation transformation.

    H_gamma(x) =
        x / 2 *
        [ (|x| / 2)^(gamma - 1) + 1 ]

    gamma = 1 gives a linear transformation.
    gamma = 3 is used for Modality C.
    """
    if gamma < 1:
        raise ValueError(
            "gamma must be greater than or equal to 1."
        )

    return (
        (x / 2.0)
        * (
            np.power(
                np.abs(x) / 2.0,
                gamma - 1,
            )
            + 1.0
        )
    )


def modality_c(
    states: np.ndarray,
    gamma: int = 3,
) -> np.ndarray:
    """
    Modality C: nonlinear observations.

    Apply H_gamma to the complementary alternating
    state components:

        x2, x4, x6, ..., x40

    In zero-based Python indexing:

        1, 3, 5, ..., 39

    Default:
        gamma = 3

    Input:
        (..., 40)

    Output:
        (..., 20)
    """
    states = _validate_states(states)

    selected = states[..., 1::2]

    observations = nonlinear_observation(
        selected,
        gamma=gamma,
    )

    return observations.astype(np.float32, copy=False)


def generate_clean_modalities(
    states: np.ndarray,
    gamma: int = 3,
) -> dict[str, np.ndarray]:
    """
    Generate all three clean synthetic observation
    modalities from Lorenz-96 ground truth.

    Returns:
        {
            "A": (..., 20),
            "B": (..., 20),
            "C": (..., 20),
        }
    """
    states = _validate_states(states)

    return {
        "A": modality_a(states),
        "B": modality_b(states),
        "C": modality_c(
            states,
            gamma=gamma,
        ),
    }