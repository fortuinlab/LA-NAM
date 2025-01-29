import typing
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from laplace.baselaplace import KronLaplace, ParametricLaplace
from laplace.utils import normal_samples
from sklearn.preprocessing import FunctionTransformer
from skorch.callbacks import Callback
from skorch.dataset import Dataset
from skorch.utils import to_numpy

from laplace_skorch.hyper import HyperModule
from laplace_skorch.modules import Router
from laplace_skorch.net import LaplaceAdditiveNetMixin, LaplaceNetMixin
from laplace_skorch.utils import log_prob_mass


class _PredictProbaKwargs(typing.TypedDict):
    use_probit: typing.NotRequired[bool]
    num_samples: typing.NotRequired[int]


class LaplaceNetClassifier(skorch.NeuralNetClassifier, LaplaceNetMixin):
    def __init__(
        self,
        module: nn.Module | type[nn.Module],
        criterion: nn.Module | type[nn.Module] = torch.nn.CrossEntropyLoss,
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

    def predict_proba(
        self, X: npt.ArrayLike, *, use_probit: bool = False, num_samples: int = 100
    ) -> npt.NDArray[np.floating]:
        self.check_is_fitted()
        f_mean, f_cov = self.forward(X, training=False)

        if use_probit:
            kappa_inv = torch.sqrt(
                1.0 + (torch.pi / 8) * torch.diagonal(f_cov, dim1=1, dim2=2)
            )
            f_probas = torch.softmax(f_mean / kappa_inv, dim=-1)
        else:
            f_samples = normal_samples(
                f_mean, var=torch.diagonal(f_cov, dim1=1, dim2=2), n_samples=num_samples
            )
            f_probas = torch.softmax(f_samples, dim=-1).mean(0)

        return to_numpy(f_probas)

    def predict(
        self, X: npt.ArrayLike, **kwargs: typing.Unpack[_PredictProbaKwargs]
    ) -> npt.NDArray[np.integer]:
        return self.predict_proba(X, **kwargs).argmax(axis=1)

    def score(
        self, X: npt.ArrayLike, y: npt.ArrayLike, sample_weight: Any | None = None
    ) -> float:
        if sample_weight is not None:
            raise NotImplementedError("`sample_weight` is not supported.")

        return log_prob_mass(y, self.predict_proba(X))


class LaplaceAdditiveNetBinaryClassifier(LaplaceAdditiveNetMixin, LaplaceNetClassifier):
    """Laplace-Approximated NAM (LA-NAM) for binary classification."""

    class _AdditiveBinaryClassificationModel(Router):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            output = super().forward(input)
            output = output.view(input.shape[0], -1, self.out_features).sum(1)

            return F.pad(output, pad=(1, 0))

    module: type[_AdditiveBinaryClassificationModel]
    module_: _AdditiveBinaryClassificationModel

    target_transformer_: FunctionTransformer

    def __init__(
        self,
        module: nn.Module | type[nn.Module] = _AdditiveBinaryClassificationModel,
        criterion: nn.Module | type[nn.Module] = torch.nn.CrossEntropyLoss,
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
        if module is not self._AdditiveBinaryClassificationModel:
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
        self.target_transformer_ = FunctionTransformer()
        return self

    def predict_proba(
        self, X: npt.ArrayLike, *, use_probit: bool = False, num_samples: int = 100
    ) -> npt.NDArray[np.floating]:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, copy=False)  # type: ignore[arg-type]
        X = self.column_transformer_.transform(X).astype(np.float32)

        return super().predict_proba(X, use_probit=use_probit, num_samples=num_samples)

    def predict(
        self, X: npt.ArrayLike, **kwargs: typing.Unpack[_PredictProbaKwargs]
    ) -> npt.NDArray[np.integer]:
        return self.predict_proba(X, **kwargs).argmax(axis=1)
