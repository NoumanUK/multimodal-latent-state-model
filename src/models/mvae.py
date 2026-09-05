from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn

from src.models.poe import (
    gaussian_product_of_experts,
)


class ModalityEncoder(nn.Module):
    """
    Encoder for one observation modality.

    Input:
        [batch, 20]

    Output:
        mu:
            [batch, latent_dim]

        logvar:
            [batch, latent_dim]
    """

    def __init__(
        self,
        input_dim: int = 20,
        latent_dim: int = 16,
    ):
        super().__init__()

        self.network = nn.Sequential(
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

        self.mu_layer = nn.Linear(
            64,
            latent_dim,
        )

        self.logvar_layer = nn.Linear(
            64,
            latent_dim,
        )

    def forward(
        self,
        observations: torch.Tensor,
    ):

        hidden = self.network(
            observations
        )

        mu = self.mu_layer(
            hidden
        )

        logvar = self.logvar_layer(
            hidden
        )

        return (
            mu,
            logvar,
        )


class MVAE(nn.Module):
    """
    Static multimodal variational latent-state estimator.

    Modalities:
        A, B, C

    Each modality has an independent encoder.

    The resulting Gaussian experts are fused using a
    Product of Experts together with a standard-normal
    prior.

    The fused latent representation is decoded into
    the normalized Lorenz-96 state X_t.
    """

    VALID_MODALITIES = (
        "A",
        "B",
        "C",
    )

    def __init__(
        self,
        modality_dim: int = 20,
        latent_dim: int = 16,
        output_dim: int = 40,
    ):
        super().__init__()

        self.modality_dim = modality_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # -------------------------------------------------
        # Separate modality encoders.
        # -------------------------------------------------

        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(
                    input_dim=modality_dim,
                    latent_dim=latent_dim,
                )
                for name in self.VALID_MODALITIES
            }
        )

        # -------------------------------------------------
        # Shared state decoder.
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

    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:

        std = torch.exp(
            0.5 * logvar
        )

        epsilon = torch.randn_like(
            std
        )

        return (
            mu
            + std * epsilon
        )

    def encode_modality(
        self,
        name: str,
        observations: torch.Tensor,
    ):

        if name not in self.encoders:

            raise ValueError(
                f"Unknown modality: {name}"
            )

        return self.encoders[
            name
        ](
            observations
        )

    def fuse(
        self,
        modality_posteriors: Dict[
            str,
            tuple[
                torch.Tensor,
                torch.Tensor,
            ]
        ],
    ):

        if not modality_posteriors:

            raise ValueError(
                "At least one modality must "
                "be available."
            )

        mus = []
        logvars = []

        for name in self.VALID_MODALITIES:

            if name not in (
                modality_posteriors
            ):
                continue

            mu, logvar = (
                modality_posteriors[
                    name
                ]
            )

            mus.append(
                mu
            )

            logvars.append(
                logvar
            )

        stacked_mu = torch.stack(
            mus,
            dim=0,
        )

        stacked_logvar = torch.stack(
            logvars,
            dim=0,
        )

        fused_mu, fused_logvar = (
            gaussian_product_of_experts(
                mus=stacked_mu,
                logvars=stacked_logvar,
                include_standard_normal_prior=True,
            )
        )

        return (
            fused_mu,
            fused_logvar,
        )

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        return self.decoder(
            z
        )

    def forward(
        self,
        modalities: Dict[
            str,
            torch.Tensor,
        ],
        available_modalities: Iterable[
            str
        ] | None = None,
        deterministic: bool = False,
    ):

        if available_modalities is None:

            available_modalities = (
                self.VALID_MODALITIES
            )

        available_modalities = tuple(
            available_modalities
        )

        if len(
            available_modalities
        ) == 0:

            raise ValueError(
                "At least one modality "
                "must be available."
            )

        modality_posteriors = {}

        for name in (
            available_modalities
        ):

            if name not in (
                self.VALID_MODALITIES
            ):

                raise ValueError(
                    f"Invalid modality: "
                    f"{name}"
                )

            if name not in modalities:

                raise KeyError(
                    f"Missing tensor for "
                    f"modality {name}."
                )

            mu, logvar = (
                self.encode_modality(
                    name,
                    modalities[
                        name
                    ],
                )
            )

            modality_posteriors[
                name
            ] = (
                mu,
                logvar,
            )

        fused_mu, fused_logvar = (
            self.fuse(
                modality_posteriors
            )
        )

        if deterministic:

            z = fused_mu

        else:

            z = (
                self.reparameterize(
                    fused_mu,
                    fused_logvar,
                )
            )

        reconstruction = (
            self.decode(
                z
            )
        )

        return {
            "reconstruction": (
                reconstruction
            ),
            "mu": (
                fused_mu
            ),
            "logvar": (
                fused_logvar
            ),
            "z": z,
            "modality_posteriors": (
                modality_posteriors
            ),
        }