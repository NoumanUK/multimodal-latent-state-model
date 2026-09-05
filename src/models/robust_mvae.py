from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import nn


MODALITY_NAMES = (
    "A",
    "B",
    "C",
)


class MaskAwareModalityEncoder(nn.Module):
    """
    Static encoder for one modality.

    Uses ONLY the current timestamp.

    Input:
        values:
            [B, 20]

        mask:
            [B, 20]

    Concatenated input:
        [B, 40]

    Output:
        mu:
            [B, latent_dim]

        logvar:
            [B, latent_dim]
    """

    def __init__(
        self,
        modality_dim: int = 20,
        latent_dim: int = 16,
    ) -> None:

        super().__init__()

        self.modality_dim = (
            modality_dim
        )

        self.latent_dim = (
            latent_dim
        )

        input_dim = (
            modality_dim * 2
        )

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
        values: torch.Tensor,
        mask: torch.Tensor,
    ):

        if values.ndim != 2:

            raise ValueError(
                "values must have shape "
                "[batch, modality_dim]."
            )

        if mask.ndim != 2:

            raise ValueError(
                "mask must have shape "
                "[batch, modality_dim]."
            )

        if values.shape != mask.shape:

            raise ValueError(
                "values and mask must have "
                "identical shapes."
            )

        x = torch.cat(
            (
                values,
                mask,
            ),
            dim=-1,
        )

        hidden = self.network(
            x
        )

        mu = self.mu_layer(
            hidden
        )

        logvar = self.logvar_layer(
            hidden
        )

        # Same numerical protection used in
        # the temporal model.
        logvar = torch.clamp(
            logvar,
            min=-10.0,
            max=10.0,
        )

        return (
            mu,
            logvar,
        )


def masked_product_of_experts(
    mus: torch.Tensor,
    logvars: torch.Tensor,
    availability: torch.Tensor,
):
    """
    Per-sample Product of Experts.

    mus:
        [B, M, D]

    logvars:
        [B, M, D]

    availability:
        [B, M]

        1 = modality available
        0 = modality unavailable

    Includes a standard-normal prior expert.

    If all modalities are unavailable for a sample,
    the result becomes the N(0, I) prior.
    """

    if mus.ndim != 3:

        raise ValueError(
            "mus must have shape "
            "[batch, modalities, latent_dim]."
        )

    if logvars.shape != mus.shape:

        raise ValueError(
            "mus and logvars must have "
            "the same shape."
        )

    if availability.ndim != 2:

        raise ValueError(
            "availability must have shape "
            "[batch, modalities]."
        )

    if (
        availability.shape[0]
        != mus.shape[0]
        or availability.shape[1]
        != mus.shape[1]
    ):

        raise ValueError(
            "availability shape is incompatible "
            "with expert tensors."
        )

    availability = (
        availability.to(
            dtype=mus.dtype
        )
    )

    availability = (
        availability.unsqueeze(
            -1
        )
    )

    precision = torch.exp(
        -logvars
    )

    # Ignore unavailable experts.
    weighted_precision = (
        precision
        * availability
    )

    # Standard N(0, I) prior:
    #
    # prior precision = 1
    # prior mean contribution = 0
    total_precision = (
        1.0
        + torch.sum(
            weighted_precision,
            dim=1,
        )
    )

    weighted_mean = torch.sum(
        mus
        * weighted_precision,
        dim=1,
    )

    fused_mu = (
        weighted_mean
        / total_precision
    )

    fused_var = (
        1.0
        / total_precision
    )

    fused_logvar = torch.log(
        fused_var
    )

    return (
        fused_mu,
        fused_logvar,
    )


class RobustMVAE(nn.Module):
    """
    Mask-aware STATIC multimodal VAE.

    Important:
        This model does NOT use temporal history.

    It uses only:
        A_t
        B_t
        C_t

    plus corresponding masks.

    This is the robust static control for comparison
    with the Temporal MVAE.
    """

    def __init__(
        self,
        modality_dim: int = 20,
        latent_dim: int = 16,
        output_dim: int = 40,
    ) -> None:

        super().__init__()

        self.modality_dim = (
            modality_dim
        )

        self.latent_dim = (
            latent_dim
        )

        self.output_dim = (
            output_dim
        )

        self.encoders = nn.ModuleDict(
            {
                name:
                MaskAwareModalityEncoder(
                    modality_dim=(
                        modality_dim
                    ),
                    latent_dim=(
                        latent_dim
                    ),
                )
                for name
                in MODALITY_NAMES
            }
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
            + epsilon * std
        )

    def encode(
        self,
        modalities: Mapping[
            str,
            torch.Tensor,
        ],
        masks: Mapping[
            str,
            torch.Tensor,
        ],
    ):

        mus = []

        logvars = []

        availability_list = []

        modality_posteriors = {}

        for name in MODALITY_NAMES:

            if name not in modalities:

                raise KeyError(
                    f"Missing modality {name}."
                )

            if name not in masks:

                raise KeyError(
                    f"Missing mask {name}."
                )

            values = modalities[
                name
            ]

            mask = masks[
                name
            ]

            mu, logvar = (
                self.encoders[
                    name
                ](
                    values,
                    mask,
                )
            )

            modality_posteriors[
                name
            ] = {
                "mu": mu,
                "logvar": logvar,
            }

            mus.append(
                mu
            )

            logvars.append(
                logvar
            )

            # A modality is considered available for
            # this sample if at least one current feature
            # is observed.
            availability = (
                torch.sum(
                    mask,
                    dim=-1,
                )
                > 0
            ).to(
                dtype=values.dtype
            )

            availability_list.append(
                availability
            )

        stacked_mu = torch.stack(
            mus,
            dim=1,
        )

        stacked_logvar = torch.stack(
            logvars,
            dim=1,
        )

        availability = torch.stack(
            availability_list,
            dim=1,
        )

        fused_mu, fused_logvar = (
            masked_product_of_experts(
                mus=stacked_mu,
                logvars=stacked_logvar,
                availability=availability,
            )
        )

        return {
            "mu": fused_mu,
            "logvar": fused_logvar,
            "modality_posteriors": (
                modality_posteriors
            ),
            "availability": (
                availability
            ),
        }

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
        masks: Dict[
            str,
            torch.Tensor,
        ],
        deterministic: bool = False,
    ):

        encoded = self.encode(
            modalities=modalities,
            masks=masks,
        )

        mu = encoded[
            "mu"
        ]

        logvar = encoded[
            "logvar"
        ]

        if deterministic:

            z = mu

        else:

            z = self.reparameterize(
                mu,
                logvar,
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
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "modality_posteriors": (
                encoded[
                    "modality_posteriors"
                ]
            ),
            "availability": (
                encoded[
                    "availability"
                ]
            ),
        }