import abc
import functools
import typing
import warnings
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Final, Literal, Self, cast

import asdl
import numpy as np
import numpy.typing as npt
import pandas as pd
import skorch
import torch
import torch.nn as nn
import torch.utils.data
from laplace.baselaplace import KronLaplace, ParametricLaplace
from laplace.curvature import AsdlInterface, CurvlinopsInterface
from laplace.utils import expand_prior_precision
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from skorch.callbacks import Callback, EpochTimer, PassthroughScoring
from skorch.dataset import Dataset, unpack_data
from skorch.utils import TeeGenerator, to_tensor
from torch.nn.utils import parameters_to_vector

from laplace_skorch.callbacks import (
    ElapsedPrintLog,
    ElapsedTimer,
    LogMarginalLikelihood,
    LogMarginalLikelihoodCheckpoint,
)
from laplace_skorch.hyper import HyperModule
from laplace_skorch.modules import Categorical, Numerical, Router

_T = typing.TypeVar("_T")


class LaplaceNet(skorch.NeuralNet):
    """`NeuralNet` with marginal-likelihood optimization of hyperparameters."""

    curvature: Final[type[ParametricLaplace]]
    curvature_: ParametricLaplace

    hyper_module: Final[type[HyperModule]]
    hyper_module_: HyperModule

    hyper_optimizer: Final[type[torch.optim.Optimizer]]
    hyper_optimizer_: torch.optim.Optimizer

    hyper_steps: int
    hyper_epoch_freq: int
    hyper_epoch_burnin: int

    # Keep track of the total number of samples in the training dataset.
    # Must end with underscore to avoid ending up in `get_params(...)` !
    _num_samples_: int

    def __init__(
        self,
        module: nn.Module | type[nn.Module],
        criterion: nn.Module | type[nn.Module],
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 0.01,
        max_epochs: int = 10,
        batch_size: int = 128,
        iterator_train: type[torch.utils.data.DataLoader] = torch.utils.data.DataLoader,
        iterator_valid: type[torch.utils.data.DataLoader] = torch.utils.data.DataLoader,
        dataset: type[Dataset] = Dataset,
        train_split: Callable | None = None,
        callbacks: Sequence[Callback] | Literal["disable"] | None = None,
        predict_nonlinearity: Callable | Literal["auto"] | None = "auto",
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
        class_name = self.__class__.__name__

        if train_split is not None:
            warnings.warn(
                f"`{class_name}` optimizes hyperparameters using an estimate of the "
                f"marginal likelihood and does not require additional validation data. "
                f"Keep `train_split` set to `None` to avoid creating an unnecessary "
                f"split of the training data.",
                stacklevel=1,
            )

        if callbacks == "disable":
            warnings.warn(
                f"`{class_name}` relies on callbacks to keep track of the marginal "
                f"likelihood and to restore the best model after training. You can "
                f"disable a specific callback by setting the corresponding keyword "
                f"argument `callbacks__{{name}}` to `None`.",
                stacklevel=1,
            )

        self._register_attribute(
            "curvature", curvature, cuda_dependent_attributes=False
        )
        self.curvature = curvature

        self.hyper_module = hyper_module
        self.hyper_optimizer = hyper_optimizer
        self.hyper_steps = hyper_steps
        self.hyper_epoch_freq = hyper_epoch_freq
        self.hyper_epoch_burnin = hyper_epoch_burnin

        super().__init__(
            module,
            criterion,
            optimizer,
            lr,
            max_epochs,
            batch_size,
            iterator_train,
            iterator_valid,
            dataset,
            train_split,
            callbacks,
            predict_nonlinearity,
            warm_start,
            verbose,
            device,
            **kwargs,
        )

    def get_default_callbacks(self) -> list[tuple[str, Callback]]:
        return [
            ("epoch_timer", EpochTimer()),
            ("train_timer", ElapsedTimer()),
            (
                "train_loss",
                PassthroughScoring(name="train_loss", on_train=True),
            ),
            (
                "valid_loss",
                PassthroughScoring(name="valid_loss"),
            ),
            (
                "log_marginal_likelihood",
                LogMarginalLikelihood(),
            ),
            (
                "log_marginal_likelihood_cp",
                LogMarginalLikelihoodCheckpoint(),
            ),
            ("print_log", ElapsedPrintLog()),
        ]

    def get_iterator(
        self, dataset: Dataset, training: bool = False
    ) -> torch.utils.data.DataLoader:
        if training:
            kwargs = self.get_params_for("iterator_train")
            iterator = self.iterator_train
        else:
            kwargs = self.get_params_for("iterator_valid")
            iterator = self.iterator_valid

        if "batch_size" not in kwargs:
            kwargs["batch_size"] = self.batch_size

        if kwargs["batch_size"] == -1:
            kwargs["batch_size"] = len(dataset)

        # Pop any keyword arguments pertaining to the sampler.
        sampler_params = _pop_params_for("sampler", kwargs)

        # Initialize the sampler before passing it to the iterator.
        if (sampler_class := kwargs.get("sampler")) is not None:
            kwargs["sampler"] = sampler_class(dataset, **sampler_params)

        return iterator(dataset, **kwargs)

    def initialize_module(self) -> Self:
        super().initialize_module()

        params = self.get_params_for("hyper_module")
        self.hyper_module_ = self.initialized_instance(self.hyper_module, params)
        self.hyper_module_.initialize_prior_prec(self.module_)
        return self

    def initialize_criterion(self) -> Self:
        super().initialize_criterion()

        if (
            not (isinstance(self.criterion_, nn.MSELoss | nn.CrossEntropyLoss))
            or self.criterion_.reduction != "mean"
        ):
            raise ValueError(
                "Criterion must be `torch.nn.MSELoss` or a `torch.nn.CrossEntropyLoss` "
                "and must have its `reduction` set to `mean`."
            )

        self.hyper_module_.initialize_sigma_noise(self.criterion_)
        self.hyper_module_.to(self.device)
        return self

    def initialize_optimizer(self) -> Self:
        args, kwargs = self.get_params_for_optimizer(
            "optimizer", self.module_.named_parameters()
        )
        self.optimizer_ = self.optimizer(*args, **kwargs)

        if "weight_decay" in kwargs:
            class_name = self.__class__.__name__
            warnings.warn(
                f"Weight decay is handled internally by `{class_name}` and should not "
                f"be passed explicitly as a keyword argument (e.g. as `weight_decay`).",
                stacklevel=1,
            )

        args, kwargs = self.get_params_for_optimizer(
            "hyper_optimizer", self.hyper_module_.named_parameters()
        )
        self.hyper_optimizer_ = self.hyper_optimizer(*args, **kwargs)

        return self

    def get_params_for_curvature(self) -> tuple[Sequence[Any], Mapping[str, Any]]:
        args = (
            self.module_,
            self.hyper_module_.likelihood,
            self.hyper_module_.sigma_noise,
            self.hyper_module_.prior_precision,
        )

        kwargs = self.get_params_for("curvature")
        kwargs["temperature"] = self.hyper_module_.temperature
        return args, kwargs

    def initialize_curvature(self) -> Self:
        args, kwargs = self.get_params_for_curvature()
        self.curvature_ = self.curvature(*args, **kwargs)

        return self

    def get_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        X: Any = None,
        training: bool = False,
    ) -> torch.Tensor:
        prior_prec = self.hyper_module_.prior_precision.detach()
        crit_factor = self.hyper_module_.crit_factor.detach()
        num_samples = self._num_samples_

        theta = parameters_to_vector(self.module_.parameters())
        delta = expand_prior_precision(prior_prec, self.module_)

        return (
            super().get_loss(y_pred, y_true, X, training)
            + (0.5 * (delta * theta) @ theta) / num_samples / crit_factor
        )

    def train_step(
        self, batch: tuple[Any, Any], **fit_params: Any
    ) -> Mapping[str, Any]:
        step_accumulator = self.get_train_step_accumulator()

        def step_fn() -> torch.Tensor:
            self.optimizer_.zero_grad()
            step = self.train_step_single(batch, **fit_params)
            step_accumulator.store_step(step)

            self.notify(
                "on_grad_computed",
                named_parameters=TeeGenerator(self.get_all_learnable_params()),
                batch=batch,
            )
            return step["loss"]

        self.optimizer_.step(step_fn)
        return step_accumulator.get_step()

    @property
    def log_marginal_likelihood_(self) -> torch.Tensor:
        prior_prec = self.hyper_module_.prior_precision
        sigma_noise = self.hyper_module_.sigma_noise

        if self.hyper_module_.likelihood == "classification":
            return self.curvature_.log_marginal_likelihood(prior_prec)
        else:
            return self.curvature_.log_marginal_likelihood(prior_prec, sigma_noise)

    def on_epoch_begin(
        self,
        net: skorch.NeuralNet,
        dataset_train: Dataset,
        dataset_valid: Dataset | None,
        **kwargs: Any,
    ) -> None:
        self._num_samples_ = len(dataset_train)

        super().on_epoch_begin(net, dataset_train, dataset_valid, **kwargs)

    def on_epoch_end(
        self,
        net: skorch.NeuralNet,
        dataset_train: Dataset,
        dataset_valid: Dataset | None,
        **kwargs: Any,
    ) -> None:
        epoch: int = self.history[-1, "epoch"]

        if (
            epoch > self.hyper_epoch_burnin
            and ((epoch - self.hyper_epoch_burnin - 1) % self.hyper_epoch_freq) == 0
        ):
            self.initialize_curvature()

            try:
                # `iterator_train` does not necessarily perform an entire pass on the
                # training dataset (e.g. when `iterator_train__drop_last` is `True`.)
                # Use `iterator_valid` with `dataset_train` to get a full pass.
                self.curvature_.fit(self.get_iterator(dataset_train, training=False))
            except KeyboardInterrupt as keyboard_interrupt:
                # `skorch` uses `KeyboardInterrupt` to trigger an early stop.
                # Clean up state and hooks left over by the curvature interface.
                if isinstance(self.curvature_.backend, AsdlInterface):
                    for param in self.module_.parameters():
                        asdl.utils.restore_original_requires_grad(param)
                if isinstance(self.curvature_.backend, CurvlinopsInterface):
                    _remove_leftover_curvlinops_hooks(self.module_)
                raise keyboard_interrupt

            def step_fn() -> float:
                self.hyper_optimizer_.zero_grad()
                loss = -self.log_marginal_likelihood_
                loss.backward()
                return loss.item()

            hyper_requires_grad = (
                self.hyper_module_._log_prior_prec.requires_grad
                or self.hyper_module_._log_sigma_noise.requires_grad
            )

            if hyper_requires_grad:
                for _ in range(self.hyper_steps):
                    self.hyper_optimizer_.step(step_fn)

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.floating]:
        warnings.warn(
            "Using MAP estimate to make predictions. Use `LaplaceNetClassifier` "
            "to obtain predictions from the posterior predictive distribution.",
            stacklevel=1,
        )

        return super().predict_proba(X)


class LaplaceNetMixin(LaplaceNet, abc.ABC):
    """Use the GLM predictive for predictions."""

    @torch.enable_grad()
    def infer_with_predictive(
        self, X: npt.ArrayLike
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X = to_tensor(X, device=self.device)
        Js, f_mean = self.curvature_.backend.jacobians(X)
        f_cov = self.curvature_.functional_variance(Js)
        return f_mean.detach(), f_cov.detach()

    def evaluation_step(
        self, batch: tuple[npt.ArrayLike, npt.ArrayLike], training: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.check_is_fitted()
        self.module_.train(training)
        X_batch, _ = unpack_data(batch)
        return self.infer_with_predictive(X_batch)

    @abc.abstractmethod
    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.floating]:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.floating | np.integer]:
        raise NotImplementedError


def _pop_params_for(prefix: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pop keyword arguments starting with `prefix` from `kwargs`."""

    if not prefix.endswith("__"):
        prefix += "__"

    def hasprefix(key: str) -> bool:
        return key.startswith(prefix)

    def removeprefix(key: str) -> str:
        return key[len(prefix) :]

    return {
        removeprefix(key): kwargs.pop(key) for key in list(kwargs) if hasprefix(key)
    }


def _remove_leftover_curvlinops_hooks(module: nn.Module) -> None:
    """Remove forward pass hooks left over by `curvlinops`."""

    for submodule in module.modules():
        for hook_id, hook in list(submodule._forward_pre_hooks.items()):
            if isinstance(hook, functools.partial):
                hook = hook.func
            if hook.__module__.startswith("curvlinops"):
                del submodule._forward_pre_hooks[hook_id]

        for hook_id, hook in list(submodule._forward_hooks.items()):
            if isinstance(hook, functools.partial):
                hook = hook.func
            if hook.__module__.startswith("curvlinops"):
                del submodule._forward_hooks[hook_id]


class LaplaceAdditiveNetMixin(LaplaceNetMixin):
    """Add-ons for Laplace-Approximated Neural Additive Models (LA-NAMs)."""

    _X_ref_: weakref.ReferenceType[pd.DataFrame]
    _y_ref_: weakref.ReferenceType[pd.Series]

    module: type[Router]
    module_: Router

    column_transformer_: ColumnTransformer
    target_transformer_: StandardScaler | FunctionTransformer

    feature_bias_: npt.NDArray[np.floating]
    feature_vars_: npt.NDArray[np.floating]

    def initialize(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Self:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, copy=False)  # type: ignore[arg-type]
        if not isinstance(y, pd.Series):
            y = pd.Series(y, copy=False)  # type: ignore[arg-type]

        self._X_ref_ = weakref.ref(X)
        self._y_ref_ = weakref.ref(y)
        self._initialize_preprocessors()
        return super().initialize()

    def _initialize_preprocessors(self) -> Self:
        if (X := self._X_ref_()) is None or self._y_ref_() is None:
            raise RuntimeError("Broken weak reference to `X` or `y`.")

        self.column_transformer_ = ColumnTransformer(_transformers_from_dataframe(X))
        return self

    def initialized_instance(
        self, instance_or_cls: _T | type[_T], kwargs: Mapping[str, Any]
    ) -> _T | Router:
        if isinstance(instance_or_cls, type) and issubclass(instance_or_cls, Router):
            if (X := self._X_ref_()) is None:
                raise AssertionError("Broken weak reference to `X`.")
            return instance_or_cls(_modules_from_dataframe(X, **kwargs))
        return super().initialized_instance(instance_or_cls, kwargs)

    def partial_fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike, classes: Any = None, **fit_params: Any
    ) -> Self:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, copy=False)  # type: ignore[arg-type]
        if not isinstance(y, pd.Series):
            y = pd.Series(y, copy=False)  # type: ignore[arg-type]

        if not self.initialized_:
            self.initialize(X, y)

        dtype_X = torch.float32
        if isinstance(self.target_transformer_, StandardScaler):
            y = y.to_frame()
            dtype_y = torch.float32
        else:
            dtype_y = torch.int64

        X = torch.tensor(self.column_transformer_.fit_transform(X), dtype=dtype_X)
        y = torch.tensor(self.target_transformer_.fit_transform(y), dtype=dtype_y)
        return super().partial_fit(X, y, classes, **fit_params)

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike, **fit_params: Any) -> Self:
        if not self.warm_start or not self.initialized_:
            self.initialize(X, y)
        return self.partial_fit(X, y, **fit_params)

    @torch.no_grad()
    def on_train_end(
        self,
        net: Self,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        feature_moments = StandardScaler()

        for X_batch, _ in net.get_iterator(net.dataset(X, y), training=False):
            f_batch = Router.forward(self.module_, X_batch.to(net.device))
            feature_moments.partial_fit(f_batch.cpu())

        self.feature_bias_ = feature_moments.mean_
        self.feature_vars_ = feature_moments.var_


def _transformers_from_dataframe(
    X: pd.DataFrame,
) -> list[tuple[str, TransformerMixin, list[str | int]]]:
    transformers: list[tuple[str, TransformerMixin, list[str | int]]]
    transformers = []

    for column_name, column in cast(Iterable[tuple[str | int, pd.Series]], X.items()):
        if column.dtype in [np.int32, np.int64, np.float32, np.float64]:
            transformers.append((str(column_name), StandardScaler(), [column_name]))
        elif column.dtype in [bool, object, "category"]:
            transformers.append((str(column_name), OrdinalEncoder(), [column_name]))
        else:
            raise ValueError(
                f"The dtype of `{column_name}` (`{column.dtype}`) is not supported."
            )

    return transformers


class _ModuleKwargs(typing.TypedDict):
    hidden_dims: typing.NotRequired[Sequence[int]]
    activation_cls: typing.NotRequired[type[nn.Module]]


def _modules_from_dataframe(
    X: pd.DataFrame, **kwargs: typing.Unpack[_ModuleKwargs]
) -> dict[int, Categorical | Numerical]:
    modules: dict[int, Categorical | Numerical]
    modules = {}

    for column_pos, (column_name, column) in enumerate(X.items()):
        if column.dtype in [np.int32, np.int64, np.float32, np.float64]:
            modules[column_pos] = Numerical(in_features=1, out_features=1, **kwargs)
        elif column.dtype in [bool, object, "category"]:
            num_classes = len(column.unique())

            modules[column_pos] = Categorical(num_classes, out_features=1)
        else:
            raise ValueError(
                f"The dtype of `{column_name}` (`{column.dtype}`) is not supported."
            )

    return modules
