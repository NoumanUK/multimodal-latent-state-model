from __future__ import annotations

import torch
from torch import nn


class LatentTransition(nn.Module):
    """
    Deterministic learned latent transition:

        z_{t+1} = T(z_t)

    A residual connection is used so that the network learns
    the latent-state increment rather than having to reconstruct
    the entire next latent state from scratch.

    Input:
        z_t: [B, latent_dim]

    Output:
        z_next: [B, latent_dim]
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.network = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                latent_dim,
            ),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        if z.ndim != 2:
            raise ValueError(
                "Expected latent tensor [B,D], "
                f"received {tuple(z.shape)}"
            )

        delta = self.network(
            z
        )

        return z + delta

    def step(
        self,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        next_z = self(
            z
        )

        return {
            "z": next_z,
        }