from __future__ import annotations

import torch
from torch import nn


class Autoencoder(nn.Module):
    """
    Deterministic multimodal autoencoder baseline.

    Input:
        concatenated current observations

            [A_t, B_t, C_t]

        shape:
            (batch, 60)

    Output:
        reconstructed Lorenz-96 hidden state

        shape:
            (batch, 40)
    """

    def __init__(
        self,
        input_dim: int = 60,
        latent_dim: int = 16,
        output_dim: int = 40,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        if latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive."
            )

        if output_dim <= 0:
            raise ValueError(
                "output_dim must be positive."
            )

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                128,
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                64,
            ),
            nn.ReLU(),

            nn.Linear(
                64,
                latent_dim,
            ),
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                latent_dim,
                64,
            ),
            nn.ReLU(),

            nn.Linear(
                64,
                128,
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                output_dim,
            ),
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode observations into latent vector.
        """
        if x.ndim != 2:
            raise ValueError(
                "Expected x with shape "
                "(batch, features)."
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input "
                f"features, received {x.shape[-1]}."
            )

        return self.encoder(x)

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent vector into Lorenz-96 state.
        """
        if z.ndim != 2:
            raise ValueError(
                "Expected z with shape "
                "(batch, latent_dim)."
            )

        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension "
                f"{self.latent_dim}, received "
                f"{z.shape[-1]}."
            )

        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            {
                "reconstruction": (batch, 40),
                "latent":         (batch, latent_dim),
            }
        """
        z = self.encode(x)

        reconstruction = self.decode(z)

        return {
            "reconstruction": reconstruction,
            "latent": z,
        }