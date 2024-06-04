import copy
import typing
from typing import Any

import numpy as np
import numpy.typing as npt
from skorch.callbacks import Callback

if typing.TYPE_CHECKING:
    from laplace_skorch.net import LaplaceNet


class LogMarginalLikelihood(Callback):
    """Computes the log marginal likelihood after hyper-optimization."""

    prev_likelihood_: float
    best_likelihood_: float

    def on_train_begin(self, net: "LaplaceNet", **kwargs: Any) -> None:
        self.prev_likelihood_ = -np.inf
        self.best_likelihood_ = -np.inf

    def on_epoch_end(self, net: "LaplaceNet", **kwargs: Any) -> None:
        epoch: int = net.history[-1, "epoch"]

        if (
            epoch > net.hyper_epoch_burnin
            and ((epoch - net.hyper_epoch_burnin - 1) % net.hyper_epoch_freq) == 0
        ):
            likelihood = net.log_marginal_likelihood_.item()
            self.prev_likelihood_ = likelihood

            is_best = likelihood > self.best_likelihood_
            if is_best:
                self.best_likelihood_ = likelihood
        else:
            likelihood, is_best = self.prev_likelihood_, False

        net.history.record("log_marginal_likelihood", likelihood)
        net.history.record("log_marginal_likelihood_best", is_best)

        prior_precision = net.hyper_module_.prior_precision
        sigma_noise = net.hyper_module_.sigma_noise

        if prior_precision.requires_grad:
            net.history.record("prior_precision", prior_precision.detach().cpu())

        if sigma_noise.requires_grad:
            net.history.record("sigma_noise", sigma_noise.item())


class LogMarginalLikelihoodCheckpoint(Callback):
    """Restore the model with highest likelihood after training."""

    best_module_dict_: dict[str, Any] | None
    best_hyper_module_dict_: dict[str, Any] | None

    def on_train_begin(self, net: "LaplaceNet", **kwargs: Any) -> None:
        self.best_module_dict_ = None
        self.best_hyper_module_dict_ = None

    def on_epoch_end(self, net: "LaplaceNet", **kwargs: Any) -> None:
        is_best: bool = net.history[-1, "log_marginal_likelihood_best"]

        if is_best:
            self.best_module_dict_ = copy.deepcopy(net.module_.state_dict())
            self.best_hyper_module_dict_ = copy.deepcopy(net.hyper_module_.state_dict())

    def on_train_end(
        self, net: "LaplaceNet", X: npt.ArrayLike, y: npt.ArrayLike, **kwargs: Any
    ) -> None:
        if self.best_module_dict_ is not None:
            net.module_.load_state_dict(self.best_module_dict_)

        if self.best_hyper_module_dict_ is not None:
            net.hyper_module_.load_state_dict(self.best_hyper_module_dict_)

        # Use `iterator_valid` with `dataset_train` to get a full pass.
        net.initialize_curvature()
        net.curvature_.fit(net.get_iterator(net.dataset(X, y), training=False))
