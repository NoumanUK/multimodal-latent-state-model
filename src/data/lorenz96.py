from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lorenz96_rhs(
    state: NDArray[np.floating],
    forcing: float = 8.0,
) -> NDArray[np.floating]:
    """
    Compute the time derivative for the Lorenz-96 system.

    Equation:
        dx_i/dt =
            (x_{i+1} - x_{i-2}) * x_{i-1}
            - x_i
            + F

    Cyclic boundary conditions are used.

    Parameters
    ----------
    state:
        Current Lorenz-96 state with shape (N,).

    forcing:
        Constant forcing parameter F.

    Returns
    -------
    derivative:
        Time derivative with shape (N,).
    """
    x = np.asarray(state, dtype=np.float64)

    derivative = (
        (np.roll(x, -1) - np.roll(x, 2))
        * np.roll(x, 1)
        - x
        + forcing
    )

    return derivative


def rk4_step(
    state: NDArray[np.floating],
    dt: float,
    forcing: float = 8.0,
) -> NDArray[np.floating]:
    """
    Advance the Lorenz-96 system by one RK4 integration step.

    Parameters
    ----------
    state:
        Current state with shape (N,).

    dt:
        Integration time step.

    forcing:
        Lorenz-96 forcing parameter F.

    Returns
    -------
    next_state:
        State after one RK4 step.
    """
    k1 = lorenz96_rhs(state, forcing)

    k2 = lorenz96_rhs(
        state + 0.5 * dt * k1,
        forcing,
    )

    k3 = lorenz96_rhs(
        state + 0.5 * dt * k2,
        forcing,
    )

    k4 = lorenz96_rhs(
        state + dt * k3,
        forcing,
    )

    next_state = state + (
        dt / 6.0
    ) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )

    return next_state


def make_initial_state(
    dimension: int = 40,
    forcing: float = 8.0,
    perturbation_std: float = 0.1,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Create an initial Lorenz-96 state near the forcing equilibrium.

    x_i = F + epsilon_i
    epsilon_i ~ N(0, perturbation_std^2)
    """
    if dimension < 4:
        raise ValueError(
            "Lorenz-96 requires dimension >= 4."
        )

    if perturbation_std < 0:
        raise ValueError(
            "perturbation_std must be non-negative."
        )

    rng = np.random.default_rng(seed)

    state = (
        forcing
        + rng.normal(
            loc=0.0,
            scale=perturbation_std,
            size=dimension,
        )
    )

    return state.astype(np.float64)


def generate_trajectory(
    dimension: int = 40,
    forcing: float = 8.0,
    dt: float = 0.01,
    burn_in_steps: int = 1000,
    num_saved_steps: int = 2000,
    sample_every: int = 5,
    perturbation_std: float = 0.1,
    seed: int | None = None,
) -> NDArray[np.float32]:
    """
    Generate one Lorenz-96 trajectory.

    The system is first integrated for burn_in_steps.
    Those states are discarded.

    Afterwards, one state is saved every sample_every
    RK4 integration steps.

    Returns
    -------
    trajectory:
        Array with shape:
            (num_saved_steps, dimension)
    """
    if dt <= 0:
        raise ValueError(
            "dt must be greater than zero."
        )

    if burn_in_steps < 0:
        raise ValueError(
            "burn_in_steps cannot be negative."
        )

    if num_saved_steps <= 0:
        raise ValueError(
            "num_saved_steps must be positive."
        )

    if sample_every <= 0:
        raise ValueError(
            "sample_every must be positive."
        )

    state = make_initial_state(
        dimension=dimension,
        forcing=forcing,
        perturbation_std=perturbation_std,
        seed=seed,
    )

    # Burn-in period.
    for _ in range(burn_in_steps):
        state = rk4_step(
            state=state,
            dt=dt,
            forcing=forcing,
        )

    trajectory = np.empty(
        (num_saved_steps, dimension),
        dtype=np.float32,
    )

    for saved_index in range(num_saved_steps):

        for _ in range(sample_every):
            state = rk4_step(
                state=state,
                dt=dt,
                forcing=forcing,
            )

        trajectory[saved_index] = state.astype(
            np.float32
        )

    return trajectory


def generate_dataset(
    num_trajectories: int = 100,
    dimension: int = 40,
    forcing: float = 8.0,
    dt: float = 0.01,
    burn_in_steps: int = 1000,
    num_saved_steps: int = 2000,
    sample_every: int = 5,
    perturbation_std: float = 0.1,
    base_seed: int = 42,
) -> NDArray[np.float32]:
    """
    Generate multiple independent Lorenz-96 trajectories.

    Each trajectory receives a deterministic but different
    random seed derived from base_seed.

    Returns
    -------
    dataset:
        Array with shape:
            (
                num_trajectories,
                num_saved_steps,
                dimension
            )
    """
    if num_trajectories <= 0:
        raise ValueError(
            "num_trajectories must be positive."
        )

    dataset = np.empty(
        (
            num_trajectories,
            num_saved_steps,
            dimension,
        ),
        dtype=np.float32,
    )

    for trajectory_index in range(num_trajectories):

        trajectory_seed = (
            base_seed
            + trajectory_index
        )

        dataset[trajectory_index] = (
            generate_trajectory(
                dimension=dimension,
                forcing=forcing,
                dt=dt,
                burn_in_steps=burn_in_steps,
                num_saved_steps=num_saved_steps,
                sample_every=sample_every,
                perturbation_std=perturbation_std,
                seed=trajectory_seed,
            )
        )

    return dataset