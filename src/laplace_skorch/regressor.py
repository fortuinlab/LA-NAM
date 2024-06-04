import typing
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal, Self, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import skorch
import torch
import torch.nn as nn
import torch.utils.data
from laplace.baselaplace import KronLaplace, ParametricLaplace
from sklearn.preprocessing import StandardScaler
from skorch.callbacks import Callback
from skorch.dataset import Dataset

from laplace_skorch.hyper import HyperModule
from laplace_skorch.modules import Router
from laplace_skorch.net import LaplaceAdditiveNetMixin, LaplaceNetMixin
from laplace_skorch.utils import log_prob_density


class LaplaceNetRegressor(skorch.NeuralNetRegressor, LaplaceNetMixin):
    def __init__(
        self,
        module: nn.Module | type[nn.Module],
        criterion: nn.Module | type[nn.Module] = nn.MSELoss,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 0.01,
        max_epochs: int = 10,
        batch_size: int = 128,
        iterator_train: type[torch.utils.data.DataLoader] = torch.utils.data.DataLoader,
        iterator_valid: type[torch.utils.data.DataLoader] = torch.utils.data.DataLoader,
        dataset: type[Dataset] = Dataset,
        train_split: Callable | None = None,
        callbacks: Sequence[Callback] | Literal["disable"] | None = None,
        predict_nonlinearity: Callable | Literal["auto"] | None = None,
        warm_start: bool = False,
        verbose: int = 1,
        device: str | torch.device | None = "cpu",
        curvature: type[ParametricLaplace] = KronLaplace,
        hyper_module: type[HyperModule] = HyperModule,
        hyper_optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        hyper_steps: int = 10,
        hyper_epoch_freq: int = 1,
        hyper_epoch_burnin: int = 0,
        **kwargs: Any,
    ) -> None:
        if predict_nonlinearity is not None:
            class_name = self.__class__.__name__

            warnings.warn(
                f"`predict_nonlinearity` is not needed in `{class_name}`.", stacklevel=1
            )

        super().__init__(
            module=module,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            iterator_train=iterator_train,
            iterator_valid=iterator_valid,
            dataset=dataset,
            train_split=train_split,
            callbacks=callbacks,
            predict_nonlinearity=predict_nonlinearity,
            warm_start=warm_start,
            verbose=verbose,
            device=device,
            curvature=curvature,
            hyper_module=hyper_module,
            hyper_optimizer=hyper_optimizer,
            hyper_steps=hyper_steps,
            hyper_epoch_freq=hyper_epoch_freq,
            hyper_epoch_burnin=hyper_epoch_burnin,
            **kwargs,
        )

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.floating]:
        raise NotImplementedError

    def _with_obs_noise(
        self, f_mean: npt.NDArray[np.floating], f_cov: npt.NDArray[np.floating]
    ) -> None:
        N, M = f_mean.shape
        obs_noise: float = self.hyper_module_.sigma_noise.item()
        f_cov[:, np.diag_indices(M)] += obs_noise**2

    @overload
    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.floating]: ...

    @overload
    def predict(
        self, X: npt.ArrayLike, *, return_std: Literal[True]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...

    @overload
    def predict(
        self, X: npt.ArrayLike, *, return_cov: Literal[True]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...

    def predict(
        self, X: npt.ArrayLike, *, return_std: bool = False, return_cov: bool = False
    ) -> (
        npt.NDArray[np.floating]
        | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    ):
        from skorch.utils import to_numpy

        if return_std and return_cov:
            raise ValueError(
                "At most one of `return_std` and `return_cov` can be requested."
            )

        self.check_is_fitted()

        f_mean, f_cov = self.forward(X, training=False)
        f_mean, f_cov = to_numpy(f_mean), to_numpy(f_cov)
        self._with_obs_noise(f_mean, f_cov)

        if return_std:
            f_std = np.sqrt(np.diagonal(f_cov, axis1=1, axis2=2))
            return f_mean, f_std
        elif return_cov:
            return f_mean, f_cov
        else:
            return f_mean

    def score(
        self, X: npt.ArrayLike, y: npt.ArrayLike, sample_weight: Any | None = None
    ) -> float:
        if sample_weight is not None:
            raise NotImplementedError("`sample_weight` is not supported.")

        return log_prob_density(y, *self.predict(X, return_std=True))


class LaplaceAdditiveNetRegressor(LaplaceAdditiveNetMixin, LaplaceNetRegressor):
    """Laplace-Approximated NAM (LA-NAM) for regression."""

    class _AdditiveRegressionModel(Router):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            output = super().forward(input)
            output = output.view(input.shape[0], -1, self.out_features).sum(1)

            return output

    module: type[_AdditiveRegressionModel]
    module_: _AdditiveRegressionModel

    target_transformer_: StandardScaler

    def __init__(
        self,
        module: nn.Module | type[nn.Module] = _AdditiveRegressionModel,
        criterion: nn.Module | type[nn.Module] = nn.MSELoss,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 0.01,
        max_epochs: int = 10,
        batch_size: int = 128,
        iterator_train: type[torch.utils.data.DataLoader] = torch.utils.data.DataLoader,
        iterator_valid: type[torch.utils.data.DataLoader] = torch.utils.data.DataLoader,
        dataset: type[Dataset] = Dataset,
        train_split: Callable | None = None,
        callbacks: Sequence[Callback] | Literal["disable"] | None = None,
        predict_nonlinearity: Callable | Literal["auto"] | None = None,
        warm_start: bool = False,
        verbose: int = 1,
        device: str | torch.device | None = "cpu",
        curvature: type[ParametricLaplace] = KronLaplace,
        hyper_module: type[HyperModule] = HyperModule,
        hyper_optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        hyper_steps: int = 10,
        hyper_epoch_freq: int = 1,
        hyper_epoch_burnin: int = 0,
        **kwargs: Any,
    ) -> None:
        if module is not self._AdditiveRegressionModel:
            class_name = self.__class__.__name__
            raise ValueError(f"The `module` cannot be changed in `{class_name}`.")

        super().__init__(
            module=module,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            iterator_train=iterator_train,
            iterator_valid=iterator_valid,
            dataset=dataset,
            train_split=train_split,
            callbacks=callbacks,
            predict_nonlinearity=predict_nonlinearity,
            warm_start=warm_start,
            verbose=verbose,
            device=device,
            curvature=curvature,
            hyper_module=hyper_module,
            hyper_optimizer=hyper_optimizer,
            hyper_steps=hyper_steps,
            hyper_epoch_freq=hyper_epoch_freq,
            hyper_epoch_burnin=hyper_epoch_burnin,
            **kwargs,
        )

    def _initialize_preprocessors(self) -> Self:
        super()._initialize_preprocessors()
        self.target_transformer_ = StandardScaler()
        return self

    def predict_proba(self, X: npt.ArrayLike) -> typing.Never:
        raise NotImplementedError

    @overload
    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.floating]: ...

    @overload
    def predict(
        self, X: npt.ArrayLike, *, return_std: Literal[True]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...

    @overload
    def predict(
        self, X: npt.ArrayLike, *, return_cov: Literal[True]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...

    def predict(
        self,
        X: npt.ArrayLike,
        *,
        return_std: bool = False,
        return_cov: bool = False,
    ) -> (
        npt.NDArray[np.floating]
        | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
    ):
        if return_std and return_cov:
            raise ValueError(
                "At most one of `return_std` and `return_cov` can be requested."
            )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, copy=False)  # type: ignore[arg-type]
        X = self.column_transformer_.fit_transform(X).astype(np.float32)

        if return_std:
            y_pred, y_std = super().predict(X, return_std=True)
            y_pred = self.target_transformer_.inverse_transform(y_pred)
            y_std *= self.target_transformer_.scale_
            return y_pred, y_std
        elif return_cov:
            y_pred, y_cov = super().predict(X, return_cov=True)
            y_pred = self.target_transformer_.inverse_transform(y_pred)
            y_cov *= self.target_transformer_.var_
            return y_pred, y_cov
        else:
            return self.target_transformer_.inverse_transform(super().predict(X))
