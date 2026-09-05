from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Mean reconstruction/state-estimation MSE.
    """

    return F.mse_loss(
        prediction,
        target,
        reduction="mean",
    )


def kl_standard_normal(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence:

        KL[
            q(z|x)
            ||
            N(0, I)
        ]

    Average over both batch and latent dimensions.

    This normalization makes beta easier to interpret
    and keeps the KL magnitude relatively stable across
    latent dimensionalities.
    """

    kl_per_element = -0.5 * (
        1.0
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    return kl_per_element.mean()


def vae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
):
    reconstruction = reconstruction_mse(
        prediction,
        target,
    )

    kl = kl_standard_normal(
        mu,
        logvar,
    )

    total = (
        reconstruction
        + beta * kl
    )

    return {
        "loss": total,
        "reconstruction": reconstruction,
        "kl": kl,
    }
def temporal_mvae_loss(
    reconstruction,
    current_target,
    forecast,
    future_target,
    mu,
    logvar,
    beta,
    current_weight=1.0,
    forecast_weight=1.0,
):
    """
    Temporal MVAE objective.

    reconstruction:
        [B, 40]

    current_target:
        [B, 40]

    forecast:
        [B, H, 40]

    future_target:
        [B, H, 40]

    Objective:

        current reconstruction MSE
        +
        forecast MSE
        +
        beta * KL
    """

    current_reconstruction = (
        F.mse_loss(
            reconstruction,
            current_target,
            reduction="mean",
        )
    )

    forecast_reconstruction = (
        F.mse_loss(
            forecast,
            future_target,
            reduction="mean",
        )
    )

    kl = kl_standard_normal(
        mu,
        logvar,
    )

    total = (
        current_weight
        * current_reconstruction
        +
        forecast_weight
        * forecast_reconstruction
        +
        beta
        * kl
    )

    return {
        "loss": total,
        "current": (
            current_reconstruction
        ),
        "forecast": (
            forecast_reconstruction
        ),
        "kl": kl,
    }