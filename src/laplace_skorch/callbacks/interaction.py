import typing
from collections.abc import Iterator, Sequence
from typing import Any, Final, Literal

import numpy.typing as npt
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from laplace.baselaplace import FullLaplace
from skorch.callbacks import Callback, LRScheduler

from laplace_skorch.hyper import HyperModule
from laplace_skorch.modules import Categorical, Mixed, Numerical, Router
from laplace_skorch.utils import topk_unravel

if typing.TYPE_CHECKING:
    from laplace_skorch.net import LaplaceAdditiveNetMixin

_Method: typing.TypeAlias = Literal["mutinf", "marglik"]
"""Whether to use mutual information or marginal likelihood for detection."""


class _ModuleKwargs(typing.TypedDict):
    """Keyword arguments for `Numerical` and `Mixed` interaction networks."""

    activation_cls: typing.NotRequired[type[nn.Module]]
    hidden_dims: typing.NotRequired[Sequence[int]]


class LaplaceAdditive2ndStage(Callback):
    """Detects feature interaction and handles second-stage of LA-NAM fit."""

    interactions: Final[int]
    method: Final[_Method]
    max_epochs: Final[int | None]
    kwargs: _ModuleKwargs

    values_: torch.Tensor
    scores_: tuple[torch.Tensor, torch.Tensor]

    def __init__(
        self,
        *,
        interactions: int = 0,
        method: _Method = "mutinf",
        max_epochs: int | None = None,
        **kwargs: typing.Unpack[_ModuleKwargs],
    ) -> None:
        if method not in ("mutinf", "marglik"):
            raise ValueError(f"Unknown detection method: `{method}`.")

        self.interactions = interactions
        self.method = method
        self.max_epochs = max_epochs
        self.kwargs = kwargs

    def on_train_end(
        self,
        net: "LaplaceAdditiveNetMixin",
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        **kwargs: Any,
    ) -> None:
        from laplace_skorch.net import LaplaceNet

        # Return if stage two isn't requested or has already been performed.
        if self.interactions == 0 or hasattr(self, "scores_"):
            return

        if self.method == "mutinf":
            method_fn = _mutual_information
        elif self.method == "marglik":
            method_fn = _marginal_likelihood_slack

        # Compute and find top-k elements in the upper triangular.
        M = torch.triu(method_fn(_curvature_of_last_layer(net, X, y)), diagonal=1)
        topk_scores, topk_indices = topk_unravel(M, self.interactions)

        self.values_ = M.cpu()
        self.scores_ = (topk_scores.cpu(), topk_indices.cpu())

        # Append interaction networks (in sorted tuple order).
        for i, j in sorted(tuple(index.tolist() for index in topk_indices)):
            self._append_module_for_stage_two(net, i, j, **self.kwargs)

        # Re-initialize components for stage two.
        self._initialize_hyper_module_for_stage_two(net)
        self._initialize_callbacks_for_stage_two(net)

        net.module_.to(net.device)
        net.hyper_module_.to(net.device)
        net.initialize_optimizer()

        # Training data has already been pre-processed: Bypass data transformations
        # in `LaplaceAdditiveNetMixin.partial_fit(...)` by casting to `LaplaceNet`.
        LaplaceNet.partial_fit(net, X, y, epochs=self.max_epochs)

    @staticmethod
    def _append_module_for_stage_two(
        net: "LaplaceAdditiveNetMixin",
        i: int,
        j: int,
        **kwargs: typing.Unpack[_ModuleKwargs],
    ) -> None:
        mi, mj = net.module_[i], net.module_[j]
        out_features = net.module_.out_features

        if isinstance(mi, Numerical) and isinstance(mj, Numerical):
            net.module_[i, j] = Numerical(2, out_features, **kwargs)
        elif isinstance(mi, Numerical) and isinstance(mj, Categorical):
            net.module_[i, j] = Mixed(
                [1, mj.in_features_encoded], out_features, **kwargs
            )
        elif isinstance(mi, Categorical) and isinstance(mj, Numerical):
            net.module_[i, j] = Mixed(
                [mi.in_features_encoded, 1], out_features, **kwargs
            )
        elif isinstance(mi, Categorical) and isinstance(mj, Categorical):
            net.module_[i, j] = Categorical(
                [mi.in_features_encoded, mj.in_features_encoded], out_features
            )

    @staticmethod
    def _initialize_hyper_module_for_stage_two(net: "LaplaceAdditiveNetMixin") -> None:
        log_sigma_noise = net.hyper_module_._log_sigma_noise.detach()
        log_prior_prec = net.hyper_module_._log_prior_prec.detach()

        net.hyper_module_ = HyperModule()
        net.hyper_module_.initialize_prior_prec(net.module_)
        net.hyper_module_.initialize_sigma_noise(net.criterion_)

        # Copy the prior precisions of stage one to stage two.
        net.hyper_module_._log_prior_prec.data[: len(log_prior_prec)] = log_prior_prec
        if net.hyper_module_.likelihood == "regression":
            net.hyper_module_._log_sigma_noise.data = log_sigma_noise

    @staticmethod
    def _initialize_callbacks_for_stage_two(net: "LaplaceAdditiveNetMixin") -> None:
        for idx, (name, callback) in enumerate(net.callbacks_):
            # If there is an `LRScheduler`, reset the schedule for stage two.
            if isinstance(callback, LRScheduler):
                lr_scheduler = LRScheduler(**callback.get_params(), last_epoch=-1)
                lr_scheduler.initialize()
                net.callbacks_[idx] = (name, lr_scheduler)


def _mutual_information(curvature: FullLaplace) -> torch.Tensor:
    """Mutual information of parameters in the posterior covariance matrix."""

    stdev = curvature.posterior_covariance.diag().sqrt()

    return -0.5 * torch.log(
        1.0 - curvature.posterior_covariance / torch.outer(stdev, stdev)
    )


def _marginal_likelihood_slack(curvature: FullLaplace) -> torch.Tensor:
    """Marg. lik. slack of parameters in the posterior precision matrix."""

    diagp = curvature.posterior_precision.diag()

    return -torch.log(
        1.0 - torch.square(curvature.posterior_precision) / torch.outer(diagp, diagp)
    )


def _module_for_last_layer(net: "LaplaceAdditiveNetMixin") -> nn.Module:
    """Module of the last-layer approximation for detecting interaction."""

    in_features = net.module_.in_features
    out_features = net.module_.out_features

    if out_features != 1:
        raise NotImplementedError(f"Unsupported output size: `{out_features}`")

    class RouterOutputsLinearMap(nn.Linear):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if net.hyper_module_.likelihood == "classification":
                return F.pad(super().forward(input), pad=(1, 0))
            else:
                return super().forward(input)

    module = RouterOutputsLinearMap(
        in_features, out_features, bias=False, device=net.device
    )
    module.weight.data.fill_(1.0)
    return module


def _iterator_for_last_layer(
    net: "LaplaceAdditiveNetMixin", X: npt.ArrayLike, y: npt.ArrayLike
) -> torch.utils.data.DataLoader:
    """Inputs of the last-layer approximation for detecting interaction."""

    dataset = net.dataset(X, y)

    # Use `iterator_valid` batch size, or fall back to `iterator_train`.
    batch_size: int = net.get_params("iterator_valid").get("batch_size")
    batch_size = batch_size or net.batch_size

    # If neither is specified, just use the dataset length.
    if batch_size == -1:
        batch_size = len(dataset)

    class RouterOutputsLoader(torch.utils.data.DataLoader):
        def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
            for X_batch, y_batch in super().__iter__():
                f_batch = Router.forward(net.module_, X_batch.to(net.device)).detach()
                yield f_batch, y_batch

    return RouterOutputsLoader(dataset, batch_size)


def _curvature_of_last_layer(
    net: "LaplaceAdditiveNetMixin", X: npt.ArrayLike, y: npt.ArrayLike
) -> FullLaplace:
    """Laplace approximation of the last-layer for detecting interaction."""

    curvature = FullLaplace(
        _module_for_last_layer(net),
        net.hyper_module_.likelihood,
        net.hyper_module_.sigma_noise.item(),
    )
    curvature.fit(_iterator_for_last_layer(net, X, y))

    return curvature
