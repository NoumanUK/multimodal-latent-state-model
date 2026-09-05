from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from src.models.transition import (
    LatentTransition,
)


MODALITY_NAMES = (
    "A",
    "B",
    "C",
)


# =========================================================
# Temporal modality encoder
# =========================================================

class TemporalModalityEncoder(
    nn.Module
):
    """
    Encodes one modality's complete temporal history.

    Input values:
        [B, L, modality_dim]

    Input masks:
        [B, L, modality_dim]

    Values and masks are concatenated:

        [B, L, 2 * modality_dim]

    The sequence is processed using a GRU.

    The final GRU hidden state defines
    a Gaussian latent expert:

        q_m(z_t | y^m_{t-L+1:t})
    """

    def __init__(
        self,
        modality_dim: int = 20,
        hidden_dim: int = 64,
        latent_dim: int = 16,
    ):
        super().__init__()

        self.modality_dim = (
            modality_dim
        )

        self.hidden_dim = (
            hidden_dim
        )

        self.latent_dim = (
            latent_dim
        )

        input_dim = (
            modality_dim * 2
        )

        self.input_projection = (
            nn.Sequential(
                nn.Linear(
                    input_dim,
                    hidden_dim,
                ),
                nn.ReLU(),
            )
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.mu_layer = nn.Linear(
            hidden_dim,
            latent_dim,
        )

        self.logvar_layer = nn.Linear(
            hidden_dim,
            latent_dim,
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        if values.ndim != 3:

            raise ValueError(
                "Expected values [B,L,D], "
                f"received {tuple(values.shape)}"
            )

        if mask.shape != values.shape:

            raise ValueError(
                "Mask must have same shape "
                "as modality values."
            )

        if (
            values.shape[-1]
            != self.modality_dim
        ):

            raise ValueError(
                "Incorrect modality dimension."
            )

        mask = mask.to(
            dtype=values.dtype,
            device=values.device,
        )

        encoder_input = torch.cat(
            (
                values,
                mask,
            ),
            dim=-1,
        )

        projected = (
            self.input_projection(
                encoder_input
            )
        )

        _, hidden = self.gru(
            projected
        )

        # [1, B, H] -> [B, H]
        hidden = hidden[-1]

        mu = self.mu_layer(
            hidden
        )

        logvar = self.logvar_layer(
            hidden
        )

        logvar = torch.clamp(
            logvar,
            min=-10.0,
            max=10.0,
        )

        return (
            mu,
            logvar,
        )


# =========================================================
# Product of Experts
# =========================================================

def temporal_product_of_experts(
    mus: torch.Tensor,
    logvars: torch.Tensor,
    availability: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Gaussian Product of Experts with standard-normal prior.

    mus:
        [B, M, D]

    logvars:
        [B, M, D]

    availability:
        [B, M]

        True / 1 = modality expert available
        False / 0 = modality unavailable

    Includes:

        p(z) = N(0, I)

    automatically.

    If every modality is unavailable for one sample,
    the result becomes exactly the standard-normal prior.
    """

    if mus.shape != logvars.shape:

        raise ValueError(
            "mus and logvars must have "
            "identical shapes."
        )

    if mus.ndim != 3:

        raise ValueError(
            "Expected mus/logvars [B,M,D]."
        )

    if availability.ndim != 2:

        raise ValueError(
            "Expected availability [B,M]."
        )

    if (
        availability.shape[0]
        != mus.shape[0]
        or availability.shape[1]
        != mus.shape[1]
    ):

        raise ValueError(
            "Availability shape does not "
            "match expert dimensions."
        )

    logvars = torch.clamp(
        logvars,
        min=-10.0,
        max=10.0,
    )

    precision = torch.exp(
        -logvars
    )

    weights = (
        availability
        .to(
            dtype=mus.dtype,
            device=mus.device,
        )
        .unsqueeze(-1)
    )

    precision = (
        precision
        * weights
    )

    weighted_mean_sum = (
        mus
        * precision
    ).sum(
        dim=1
    )

    precision_sum = (
        precision.sum(
            dim=1
        )
    )

    # -----------------------------------------------------
    # Standard-normal prior:
    #
    # mean      = 0
    # variance  = 1
    # precision = 1
    #
    # So:
    #
    # numerator   += 0
    # denominator += 1
    # -----------------------------------------------------

    precision_sum = (
        precision_sum
        + 1.0
    )

    variance = (
        1.0
        / precision_sum
    )

    joint_mu = (
        variance
        * weighted_mean_sum
    )

    joint_logvar = torch.log(
        variance
    )

    return (
        joint_mu,
        joint_logvar,
    )


# =========================================================
# Temporal Multimodal VAE
# =========================================================

class TemporalMVAE(
    nn.Module
):

    def __init__(
        self,
        modality_dim: int = 20,
        latent_dim: int = 16,
        state_dim: int = 40,
        encoder_hidden_dim: int = 64,
        transition_hidden_dim: int = 64,
    ):
        super().__init__()

        self.modality_dim = (
            modality_dim
        )

        self.latent_dim = (
            latent_dim
        )

        self.state_dim = (
            state_dim
        )

        # -------------------------------------------------
        # One temporal encoder for each modality
        # -------------------------------------------------

        self.encoders = nn.ModuleDict(
            {
                name:
                TemporalModalityEncoder(
                    modality_dim=(
                        modality_dim
                    ),
                    hidden_dim=(
                        encoder_hidden_dim
                    ),
                    latent_dim=(
                        latent_dim
                    ),
                )
                for name
                in MODALITY_NAMES
            }
        )

        # -------------------------------------------------
        # Shared state decoder
        #
        # z_t -> X_t
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
                state_dim,
            ),
        )

        # -------------------------------------------------
        # Deterministic residual latent transition
        #
        # z_{t+1} = T(z_t)
        # -------------------------------------------------

        self.transition = (
            LatentTransition(
                latent_dim=latent_dim,
                hidden_dim=(
                    transition_hidden_dim
                ),
            )
        )

    # =====================================================
    # VAE reparameterization
    # =====================================================

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

    # =====================================================
    # Encode multimodal temporal history
    # =====================================================

    def encode(
        self,
        modalities: dict[
            str,
            torch.Tensor,
        ],
        masks: dict[
            str,
            torch.Tensor,
        ],
        available_modalities:
            Iterable[str]
            | None = None,
    ) -> dict:

        if available_modalities is None:

            available_modalities = (
                MODALITY_NAMES
            )

        requested = set(
            available_modalities
        )

        invalid = (
            requested
            - set(
                MODALITY_NAMES
            )
        )

        if invalid:

            raise ValueError(
                "Unknown modalities: "
                f"{sorted(invalid)}"
            )

        modality_mus = []

        modality_logvars = []

        modality_availability = []

        posterior_dictionary = {}

        batch_size = None

        for name in MODALITY_NAMES:

            if (
                name not in modalities
                or name not in masks
            ):

                raise KeyError(
                    f"Missing modality "
                    f"or mask: {name}"
                )

            values = (
                modalities[
                    name
                ]
            )

            mask = (
                masks[
                    name
                ]
            )

            if values.ndim != 3:

                raise ValueError(
                    f"{name} must have shape "
                    "[B,L,D]."
                )

            if mask.shape != values.shape:

                raise ValueError(
                    f"{name} mask must match "
                    "the modality values."
                )

            if (
                values.shape[-1]
                != self.modality_dim
            ):

                raise ValueError(
                    f"{name} has incorrect "
                    "modality dimension."
                )

            if batch_size is None:

                batch_size = (
                    values.shape[0]
                )

            elif (
                values.shape[0]
                != batch_size
            ):

                raise ValueError(
                    "All modalities must have "
                    "the same batch size."
                )

            # ---------------------------------------------
            # Sensor included in this experiment
            # ---------------------------------------------

            if name in requested:

                mu, logvar = (
                    self.encoders[
                        name
                    ](
                        values,
                        mask,
                    )
                )

                # A modality is considered available
                # for this particular sample if at
                # least one observation exists anywhere
                # in its complete history window.
                sample_available = (
                    mask.sum(
                        dim=(
                            1,
                            2,
                        )
                    )
                    > 0
                )

                posterior_dictionary[
                    name
                ] = {
                    "mu": mu,
                    "logvar": logvar,
                    "available": (
                        sample_available
                    ),
                }

            # ---------------------------------------------
            # Sensor globally unavailable
            # ---------------------------------------------

            else:

                mu = torch.zeros(
                    batch_size,
                    self.latent_dim,
                    dtype=values.dtype,
                    device=values.device,
                )

                logvar = torch.zeros_like(
                    mu
                )

                sample_available = (
                    torch.zeros(
                        batch_size,
                        dtype=torch.bool,
                        device=values.device,
                    )
                )

            modality_mus.append(
                mu
            )

            modality_logvars.append(
                logvar
            )

            modality_availability.append(
                sample_available
            )

        # -------------------------------------------------
        # [B,D] per modality
        #
        # ->
        #
        # [B,M,D]
        # -------------------------------------------------

        mus = torch.stack(
            modality_mus,
            dim=1,
        )

        logvars = torch.stack(
            modality_logvars,
            dim=1,
        )

        availability = torch.stack(
            modality_availability,
            dim=1,
        )

        joint_mu, joint_logvar = (
            temporal_product_of_experts(
                mus=mus,
                logvars=logvars,
                availability=availability,
            )
        )

        return {
            "mu": (
                joint_mu
            ),
            "logvar": (
                joint_logvar
            ),
            "modality_posteriors": (
                posterior_dictionary
            ),
            "availability": (
                availability
            ),
        }

    # =====================================================
    # State decoder
    # =====================================================

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        if z.ndim != 2:

            raise ValueError(
                "Expected latent tensor [B,D]."
            )

        return self.decoder(
            z
        )

    # =====================================================
    # Latent forecasting
    # =====================================================

    def forecast(
        self,
        z_current: torch.Tensor,
        horizons: tuple[int, ...] = (
            1,
            5,
            10,
        ),
    ) -> dict:
        """
        Recursively advances the latent state.

        Example:

            z_t
             |
             T
             v
            z_t+1
             |
             T
             v
            z_t+2
             ...
             |
             v
            z_t+10

        States are decoded only at the requested
        forecast horizons.
        """

        if z_current.ndim != 2:

            raise ValueError(
                "Expected z_current [B,D]."
            )

        if len(horizons) == 0:

            raise ValueError(
                "At least one forecast "
                "horizon is required."
            )

        if any(
            horizon <= 0
            for horizon
            in horizons
        ):

            raise ValueError(
                "Forecast horizons must "
                "be positive integers."
            )

        sorted_horizons = tuple(
            sorted(
                set(
                    horizons
                )
            )
        )

        maximum_horizon = max(
            sorted_horizons
        )

        z = z_current

        state_forecasts = {}

        latent_forecasts = {}

        for step in range(
            1,
            maximum_horizon + 1,
        ):

            transition_output = (
                self.transition.step(
                    z
                )
            )

            z = (
                transition_output[
                    "z"
                ]
            )

            if step in sorted_horizons:

                latent_forecasts[
                    step
                ] = z

                state_forecasts[
                    step
                ] = self.decode(
                    z
                )

        # -------------------------------------------------
        # Preserve original requested horizon order.
        #
        # Example:
        #
        # horizons = (1,5,10)
        #
        # states:
        # [B,3,40]
        # -------------------------------------------------

        states = torch.stack(
            [
                state_forecasts[
                    horizon
                ]
                for horizon
                in horizons
            ],
            dim=1,
        )

        latents = torch.stack(
            [
                latent_forecasts[
                    horizon
                ]
                for horizon
                in horizons
            ],
            dim=1,
        )

        return {
            "states": (
                states
            ),
            "latents": (
                latents
            ),
            "state_by_horizon": (
                state_forecasts
            ),
            "latent_by_horizon": (
                latent_forecasts
            ),
        }

    # =====================================================
    # Complete forward pass
    # =====================================================

    def forward(
        self,
        modalities: dict[
            str,
            torch.Tensor,
        ],
        masks: dict[
            str,
            torch.Tensor,
        ],
        available_modalities:
            Iterable[str]
            | None = None,
        horizons: tuple[int, ...] = (
            1,
            5,
            10,
        ),
        deterministic: bool = False,
    ) -> dict:
        """
        Complete model.

        Temporal observations
            ->
        modality-specific posterior experts
            ->
        Product of Experts
            ->
        q(z_t)
            ->
        current state reconstruction
            +
        recursive latent forecasting

        deterministic=False:
            sample z_t using the VAE
            reparameterization trick.

        deterministic=True:
            use posterior mean z_t = mu.
            Used for deterministic validation/test metrics.
        """

        encoded = self.encode(
            modalities=modalities,
            masks=masks,
            available_modalities=(
                available_modalities
            ),
        )

        mu = (
            encoded[
                "mu"
            ]
        )

        logvar = (
            encoded[
                "logvar"
            ]
        )

        # -------------------------------------------------
        # Current latent state
        # -------------------------------------------------

        if deterministic:

            z = mu

        else:

            z = (
                self.reparameterize(
                    mu,
                    logvar,
                )
            )

        # -------------------------------------------------
        # Current hidden-state estimate
        # -------------------------------------------------

        reconstruction = (
            self.decode(
                z
            )
        )

        # -------------------------------------------------
        # Future hidden-state predictions
        # -------------------------------------------------

        forecasts = (
            self.forecast(
                z_current=z,
                horizons=horizons,
            )
        )

        return {
            "reconstruction": (
                reconstruction
            ),
            "forecast": (
                forecasts[
                    "states"
                ]
            ),
            "forecast_latents": (
                forecasts[
                    "latents"
                ]
            ),
            "forecast_by_horizon": (
                forecasts[
                    "state_by_horizon"
                ]
            ),
            "mu": (
                mu
            ),
            "logvar": (
                logvar
            ),
            "z": (
                z
            ),
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