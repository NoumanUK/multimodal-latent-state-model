from __future__ import annotations

import torch
from torch import nn


class VAE(nn.Module):
    """
    Static variational latent-state estimator.

    Input:
        concatenated current observations
        A_t + B_t + C_t
        shape: (batch, 60)

    Output:
        reconstructed hidden state X_t
        shape: (batch, 40)

    Latent posterior:
        q(z | y) = N(mu, diag(sigma^2))
    """

    def __init__(
        self,
        input_dim: int = 60,
        latent_dim: int = 16,
        output_dim: int = 40,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # -------------------------------------------------
        # Shared encoder
        # -------------------------------------------------

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
        )

        # -------------------------------------------------
        # Variational posterior parameters
        # -------------------------------------------------

        self.mu_layer = nn.Linear(
            64,
            latent_dim,
        )

        self.logvar_layer = nn.Linear(
            64,
            latent_dim,
        )

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------

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
        observations: torch.Tensor,
    ):
        hidden = self.encoder(
            observations
        )

        mu = self.mu_layer(
            hidden
        )

        logvar = self.logvar_layer(
            hidden
        )

        return mu, logvar

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        z = mu + sigma * epsilon

        epsilon ~ N(0, I)
        """

        std = torch.exp(
            0.5 * logvar
        )

        epsilon = torch.randn_like(
            std
        )

        z = (
            mu
            + std * epsilon
        )

        return z

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(
            z
        )

    def forward(
        self,
        observations: torch.Tensor,
    ):
        mu, logvar = self.encode(
            observations
        )

        z = self.reparameterize(
            mu,
            logvar,
        )

        reconstruction = self.decode(
            z
        )

        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }