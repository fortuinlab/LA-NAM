import typing
from typing import Final, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

_Structure: typing.TypeAlias = Literal["scalar", "layerwise", "diagonal"]
"""Prior precision structure."""

_Likelihood: typing.TypeAlias = Literal["regression", "classification"]
"""Gaussian or categorical likelihood."""


class HyperModule(nn.Module):
    """The hyperparameters optimized via the log marginal likelihood."""

    _log_sigma_noise: nn.Parameter
    _log_prior_prec: nn.Parameter

    sigma_noise_init: Final[float]
    prior_prec_init: Final[float]
    prior_prec_type: Final[_Structure]
    temperature: Final[float]

    def __init__(
        self,
        *,
        sigma_noise_init: float = 1.0,
        sigma_noise_grad: bool = True,
        prior_prec_init: float = 1.0,
        prior_prec_grad: bool = True,
        prior_prec_type: _Structure = "layerwise",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        self.sigma_noise_init = sigma_noise_init
        self.prior_prec_init = prior_prec_init
        self.prior_prec_type = prior_prec_type
        self.temperature = temperature

        self._log_sigma_noise = nn.Parameter()
        self._log_sigma_noise.requires_grad = sigma_noise_grad

        self._log_prior_prec = nn.Parameter()
        self._log_prior_prec.requires_grad = prior_prec_grad

    def initialize_prior_prec(self, module: nn.Module) -> None:
        if self.prior_prec_type == "scalar":
            prior_prec_size = 1
        elif self.prior_prec_type == "layerwise":
            prior_prec_size = len(set(module.parameters()))
        elif self.prior_prec_type == "diagonal":
            prior_prec_size = len(parameters_to_vector(module.parameters()))

        log_prior_prec_init = np.log(self.temperature * self.prior_prec_init)
        self._log_prior_prec.data = log_prior_prec_init * torch.ones(prior_prec_size)

    def initialize_sigma_noise(self, criterion: nn.Module) -> None:
        if isinstance(criterion, nn.MSELoss):
            log_sigma_noise_init = np.log(self.sigma_noise_init)
            self._log_sigma_noise.data = log_sigma_noise_init * torch.ones(1)
        elif isinstance(criterion, nn.CrossEntropyLoss):
            self._log_sigma_noise.data = torch.empty(0)

    @property
    def prior_precision(self) -> torch.Tensor:
        assert self._log_prior_prec.nelement()
        return torch.exp(self._log_prior_prec)

    @property
    def likelihood(self) -> _Likelihood:
        if self._log_sigma_noise.nelement():
            return "regression"
        return "classification"

    @property
    def sigma_noise(self) -> torch.Tensor:
        if self._log_sigma_noise.nelement():
            return torch.exp(self._log_sigma_noise)
        return torch.ones(1, device=self._log_sigma_noise.device)

    @property
    def crit_factor(self) -> torch.Tensor:
        value = torch.tensor(self.temperature)
        if self._log_sigma_noise.nelement():
            value = value / (2 * self.sigma_noise**2)
        return value
