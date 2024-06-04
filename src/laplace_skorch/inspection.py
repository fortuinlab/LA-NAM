import itertools
from collections.abc import Iterable, Sequence
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.inspection._partial_dependence import _grid_from_X
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _get_column_indices

from laplace_skorch.classifier import LaplaceAdditiveNetBinaryClassifier
from laplace_skorch.net import LaplaceNet
from laplace_skorch.regressor import LaplaceAdditiveNetRegressor
from laplace_skorch.utils import powerset


def path_to_module(parent: nn.Module, child: nn.Module) -> str:
    return next(name for name, module in parent.named_modules() if module is child)


def parameter_slices(module: nn.Module) -> Iterable[tuple[str, slice]]:
    names = (name for name, _ in module.named_parameters())
    pairs = itertools.pairwise(
        itertools.accumulate(
            itertools.chain(
                [0], (parameter.nelement() for parameter in module.parameters())
            ),
        ),
    )

    return zip(names, map(lambda tup: slice(*tup), pairs), strict=True)


def grid_from_X(
    X: pd.DataFrame,
    features: str | Sequence[str] | int | Sequence[int],
    percentiles: tuple[float, float],
    grid_resolution: int,
) -> tuple[pd.DataFrame, list[int], list[npt.NDArray[Any]]]:
    indices = _get_column_indices(X, features)
    is_categorical = [
        dtype in [bool, object, "category"] for dtype in X.iloc[:, indices].dtypes
    ]

    grid, values = _grid_from_X(
        X.iloc[:, indices], percentiles, is_categorical, grid_resolution
    )

    X_grid = [X.iloc[[0]]] * len(grid)
    X_grid = pd.concat(X_grid, ignore_index=True)

    for axis, index in enumerate(indices):
        X_grid.iloc[:, index] = grid[:, axis]

    return X_grid, indices, values


def indep_subnet_predictions(
    net: LaplaceNet, module: nn.Module, X: torch.Tensor
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []

    def forward_hook_fn(
        _module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor
    ) -> None:
        outputs.append(output.detach().cpu())

    with module.register_forward_hook(forward_hook_fn):
        for batch, _ in net.get_iterator(net.dataset(X), training=False):
            net.module_(batch.to(net.device))

    return torch.concat(outputs)


def indep_subnet_covariances(
    net: LaplaceNet, module: nn.Module, X: torch.Tensor
) -> torch.Tensor:
    prefix = path_to_module(net.module_, module)
    outputs: list[torch.Tensor] = []

    for batch, _ in net.get_iterator(net.dataset(X), training=True):
        Js, _ = net.curvature_.backend.jacobians(batch.to(net.device))
        Js_mod = torch.zeros_like(Js)

        for name, indices in parameter_slices(net.module_):
            if name.startswith(prefix):
                Js_mod[:, :, indices] = Js[:, :, indices]

        output = net.curvature_.functional_variance(Js_mod)
        outputs.append(output.detach().cpu())

    return torch.concat(outputs)


@overload
def partial_dependence(
    estimator: LaplaceAdditiveNetRegressor | LaplaceAdditiveNetBinaryClassifier,
    X: npt.ArrayLike,
    features: str | Sequence[str] | int | Sequence[int],
    *,
    percentiles: tuple[float, float] = (0.05, 0.95),
    grid_resolution: int = 100,
) -> tuple[list[npt.NDArray], npt.NDArray[np.floating]]: ...


@overload
def partial_dependence(
    estimator: LaplaceAdditiveNetRegressor | LaplaceAdditiveNetBinaryClassifier,
    X: npt.ArrayLike,
    features: str | Sequence[str] | int | Sequence[int],
    *,
    percentiles: tuple[float, float] = (0.05, 0.95),
    grid_resolution: int = 100,
    return_std: Literal[True],
) -> tuple[
    list[npt.NDArray], tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
]: ...


@overload
def partial_dependence(
    estimator: LaplaceAdditiveNetRegressor | LaplaceAdditiveNetBinaryClassifier,
    X: npt.ArrayLike,
    features: str | Sequence[str] | int | Sequence[int],
    *,
    percentiles: tuple[float, float] = (0.05, 0.95),
    grid_resolution: int = 100,
    return_cov: Literal[True],
) -> tuple[
    list[npt.NDArray], tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
]: ...


def partial_dependence(
    estimator: LaplaceAdditiveNetRegressor | LaplaceAdditiveNetBinaryClassifier,
    X: npt.ArrayLike,
    features: str | Sequence[str] | int | Sequence[int],
    *,
    percentiles: tuple[float, float] = (0.05, 0.95),
    grid_resolution: int = 100,
    return_std: bool = False,
    return_cov: bool = False,
) -> tuple[
    list[npt.NDArray],
    npt.NDArray[np.floating]
    | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
]:
    from skorch.utils import to_numpy

    if return_std and return_cov:
        raise ValueError(
            "At most one of `return_std` and `return_cov` can be requested."
        )

    valid_estimator_types: tuple[type, ...] = (
        LaplaceAdditiveNetRegressor,
        LaplaceAdditiveNetBinaryClassifier,
    )

    if isinstance(estimator, valid_estimator_types):
        net = estimator
    else:
        raise ValueError(
            "`estimator` must be a `LaplaceAdditiveNetRegressor` "
            "or a `LaplaceAdditiveNetBinaryClassifier`."
        )

    net.check_is_fitted()

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)  # type: ignore[arg-type]

    X_grid, indices, values = grid_from_X(X, features, percentiles, grid_resolution)
    X_grid = torch.tensor(
        net.column_transformer_.transform(X_grid), dtype=torch.float32
    )
    output_shape = [len(value) for value in values]

    modules = [
        net.module_[index] for index in powerset(indices) if index in net.module_
    ]

    f_mean = [indep_subnet_predictions(net, module, X_grid) for module in modules]
    f_mean = to_numpy(torch.stack(f_mean).sum(0)).reshape(output_shape)

    total_bias = sum(net.feature_bias_[index] for index in indices)
    f_mean = f_mean - total_bias

    if return_std or return_cov:
        f_cov = [indep_subnet_covariances(net, module, X_grid) for module in modules]
        f_cov = to_numpy(torch.stack(f_cov).sum(0))

        if net.hyper_module_.likelihood == "regression":
            f_cov = f_cov[:, :, 0].reshape(output_shape)
        else:
            f_cov = f_cov[:, :, 1].reshape(output_shape)
    if return_std:
        f_std = np.sqrt(f_cov)

    if isinstance(target_scaler := net.target_transformer_, StandardScaler):
        f_mean *= target_scaler.scale_
        if return_cov:
            f_cov *= target_scaler.var_
        if return_std:
            f_std *= target_scaler.scale_

    if return_std:
        return values, (f_mean, f_std)
    elif return_cov:
        return values, (f_mean, f_cov)
    else:
        return values, f_mean
