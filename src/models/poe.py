from __future__ import annotations

import torch


def gaussian_product_of_experts(
    mus: torch.Tensor,
    logvars: torch.Tensor,
    include_standard_normal_prior: bool = True,
):
    """
    Product of diagonal Gaussian experts.

    Parameters
    ----------
    mus:
        Tensor of shape:
            [num_experts, batch, latent_dim]

    logvars:
        Tensor of shape:
            [num_experts, batch, latent_dim]

    include_standard_normal_prior:
        If True, includes an additional N(0, I)
        prior expert.

    Returns
    -------
    fused_mu:
        [batch, latent_dim]

    fused_logvar:
        [batch, latent_dim]

    Mathematical rule
    -----------------
    For Gaussian experts:

        precision_i = 1 / variance_i

    Product precision:

        precision = sum_i precision_i

    Product mean:

        mu = (
            sum_i precision_i * mu_i
        ) / precision

    Product variance:

        variance = 1 / precision
    """

    if mus.ndim != 3:
        raise ValueError(
            "mus must have shape "
            "[num_experts, batch, latent_dim]."
        )

    if logvars.ndim != 3:
        raise ValueError(
            "logvars must have shape "
            "[num_experts, batch, latent_dim]."
        )

    if mus.shape != logvars.shape:
        raise ValueError(
            "mus and logvars must have "
            "identical shapes."
        )

    if mus.shape[0] < 1:
        raise ValueError(
            "At least one expert is required."
        )

    # -------------------------------------------------
    # Convert log-variance to precision.
    # -------------------------------------------------

    variances = torch.exp(
        logvars
    )

    precisions = (
        1.0
        / variances
    )

    weighted_means = (
        mus
        * precisions
    )

    # -------------------------------------------------
    # Sum modality experts.
    # -------------------------------------------------

    total_precision = (
        precisions.sum(
            dim=0
        )
    )

    total_weighted_mean = (
        weighted_means.sum(
            dim=0
        )
    )

    # -------------------------------------------------
    # Standard normal prior:
    #
    # p(z) = N(0, I)
    #
    # variance = 1
    # precision = 1
    # weighted mean contribution = 0
    # -------------------------------------------------

    if include_standard_normal_prior:

        total_precision = (
            total_precision
            + 1.0
        )

    fused_variance = (
        1.0
        / total_precision
    )

    fused_mu = (
        fused_variance
        * total_weighted_mean
    )

    fused_logvar = torch.log(
        fused_variance
    )

    return (
        fused_mu,
        fused_logvar,
    )