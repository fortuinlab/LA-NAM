import dataclasses
import itertools
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple, Self, cast, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.inspection._partial_dependence import _grid_from_X as _grid_from_X_sk
from sklearn.preprocessing import StandardScaler
from sklearn.utils._indexing import _get_column_indices, _safe_indexing
from sklearn.utils._optional_dependencies import check_matplotlib_support

from laplace_skorch.classifier import LaplaceAdditiveNetBinaryClassifier
from laplace_skorch.net import LaplaceNet
from laplace_skorch.regressor import LaplaceAdditiveNetRegressor
from laplace_skorch.utils import powerset

if TYPE_CHECKING:
    from matplotlib.axes import Axes as MplAxes
    from matplotlib.figure import Figure as MplFigure


def _path_to_module(parent: nn.Module, child: nn.Module) -> str:
    return next(name for name, module in parent.named_modules() if module is child)


def _parameter_slices(module: nn.Module) -> Iterable[tuple[str, slice]]:
    names = (name for name, _ in module.named_parameters())
    pairs = itertools.pairwise(
        itertools.accumulate(
            itertools.chain(
                [0], (parameter.nelement() for parameter in module.parameters())
            ),
        ),
    )

    return zip(names, map(lambda tup: slice(*tup), pairs), strict=True)


def _grid_from_X(
    X: pd.DataFrame,
    features: str | Sequence[str] | int | Sequence[int],
    percentiles: tuple[float, float],
    grid_resolution: int,
) -> tuple[pd.DataFrame, list[int], list[npt.NDArray[Any]]]:
    indices = _get_column_indices(X, features)
    is_categorical = [
        dtype in [bool, object, "category"] for dtype in X.iloc[:, indices].dtypes
    ]

    grid, values = _grid_from_X_sk(
        X.iloc[:, indices], percentiles, is_categorical, grid_resolution
    )

    X_grid = [X.iloc[[0]]] * len(grid)
    X_grid = pd.concat(X_grid, ignore_index=True)

    for axis, index in enumerate(indices):
        X_grid.iloc[:, index] = pd.Series(
            grid[:, axis], dtype=X_grid.iloc[:, index].dtype
        )

    return X_grid, indices, values


def subnetwork_outputs_for_X(
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


def subnetwork_covar_for_X(
    net: LaplaceNet, module: nn.Module, X: torch.Tensor
) -> torch.Tensor:
    prefix = _path_to_module(net.module_, module)
    outputs: list[torch.Tensor] = []

    for batch, _ in net.get_iterator(net.dataset(X), training=True):
        Js, _ = net.curvature_.backend.jacobians(batch.to(net.device))
        Js_mod = torch.zeros_like(Js)

        for name, indices in _parameter_slices(net.module_):
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

    X_grid, indices, values = _grid_from_X(X, features, percentiles, grid_resolution)
    X_grid = torch.tensor(
        net.column_transformer_.transform(X_grid), dtype=torch.float32
    )
    output_shape = [len(value) for value in values]

    modules = [
        net.module_[index] for index in powerset(indices) if index in net.module_
    ]

    f_mean = [subnetwork_outputs_for_X(net, module, X_grid) for module in modules]
    f_mean = to_numpy(torch.stack(f_mean).sum(0)).reshape(output_shape)

    total_bias = sum(net.feature_bias_[index] for index in indices)
    f_mean = f_mean - total_bias

    if return_std or return_cov:
        f_cov = [subnetwork_covar_for_X(net, module, X_grid) for module in modules]
        f_cov = to_numpy(torch.stack(f_cov).sum(0))

        if net.hyper_module_.likelihood == "regression":
            f_cov = f_cov[:, 0, 0].reshape(output_shape)
        else:
            f_cov = f_cov[:, 1, 1].reshape(output_shape)
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


class LaplaceAdditivePartialDependenceDisplay:
    @dataclasses.dataclass(frozen=True)
    class _OneWayData:
        class XData(NamedTuple):
            values: npt.NDArray
            raw_values: npt.NDArray

        class YData(NamedTuple):
            mean: npt.NDArray[np.floating]
            stddev: npt.NDArray[np.floating]

        label: str
        is_categorical: bool
        x_data: XData
        y_data: YData

    @dataclasses.dataclass(frozen=True)
    class _TwoWayData:
        class XData(NamedTuple):
            x0: npt.NDArray
            x1: npt.NDArray

        class YData(NamedTuple):
            mean: npt.NDArray[np.floating]
            stddev: npt.NDArray[np.floating]

        labels: tuple[str, str]
        is_categorical: tuple[bool, bool]
        x_data: XData
        y_data: YData

    data: Final[_OneWayData | _TwoWayData]

    def __init__(self, data: _OneWayData | _TwoWayData) -> None:
        self.data = data

    @classmethod
    def from_estimator(
        cls,
        estimator: LaplaceAdditiveNetRegressor | LaplaceAdditiveNetBinaryClassifier,
        X: npt.ArrayLike,
        features: str | Sequence[str] | int | Sequence[int],
        *,
        percentiles: tuple[float, float] = (0.05, 0.95),
        grid_resolution: int = 100,
        ax: "MplAxes | None" = None,
        std_factor: float = 2.0,
        histograms: bool = True,
        dashed_zero: bool = True,
        plot_kwargs: dict[str, Any] | None = None,
        fill_kwargs: dict[str, Any] | None = None,
        hist_kwargs: dict[str, Any] | None = None,
        display_type: Literal["mean", "stddev"] = "mean",
        contour_levels: int = 10,
        symmetric_norm: bool = False,
        contour_kwargs: dict[str, Any] | None = None,
        heatmap_kwargs: dict[str, Any] | None = None,
        colorbar_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        labels = cls._label_from_estimator(estimator, X, features)
        is_categorical = cls._is_categorical_from_estimator(estimator, X, features)

        values, (f_mean, f_std) = partial_dependence(
            estimator,
            X,
            features,
            percentiles=percentiles,
            grid_resolution=grid_resolution,
            return_std=True,
        )

        if len(values) == 1:
            raw_values = cls._raw_values_from_X(
                X, features, values[0], is_categorical[0]
            )

            data = cls._OneWayData(
                labels[0],
                is_categorical[0],
                cls._OneWayData.XData(values[0], raw_values),
                cls._OneWayData.YData(f_mean, f_std),
            )
        elif len(values) == 2:
            data = cls._TwoWayData(
                cast(tuple[str, str], tuple(labels)),
                cast(tuple[bool, bool], tuple(is_categorical)),
                cls._TwoWayData.XData(*values),
                cls._TwoWayData.YData(f_mean, f_std),
            )  # type: ignore[assignment]
        else:
            raise ValueError(f"`{cls.__name__}` can only display up to two features.")

        display = cls(data)
        return display.plot(
            ax=ax,
            std_factor=std_factor,
            histograms=histograms,
            dashed_zero=dashed_zero,
            plot_kwargs=plot_kwargs,
            fill_kwargs=fill_kwargs,
            hist_kwargs=hist_kwargs,
            display_type=display_type,
            contour_levels=contour_levels,
            symmetric_norm=symmetric_norm,
            contour_kwargs=contour_kwargs,
            heatmap_kwargs=heatmap_kwargs,
            colorbar_kwargs=colorbar_kwargs,
        )

    @staticmethod
    def _raw_values_from_X(
        X: npt.ArrayLike,
        features: str | Sequence[str] | int | Sequence[int],
        values: npt.NDArray,
        is_categorical: bool,
    ) -> npt.NDArray:
        assert isinstance(features, int | str) or len(features) == 1

        raw_values = _safe_indexing(X, features, axis=1)
        if not is_categorical:
            raw_values = raw_values[
                (values[0] <= raw_values) & (raw_values <= values[-1])
            ]
        return raw_values

    @staticmethod
    def _label_from_estimator(
        estimator: LaplaceAdditiveNetRegressor | LaplaceAdditiveNetBinaryClassifier,
        X: npt.ArrayLike,
        features: str | Sequence[str] | int | Sequence[int],
    ) -> list[str]:
        feature_names_in = estimator.column_transformer_.feature_names_in_
        return [feature_names_in[idx] for idx in _get_column_indices(X, features)]

    @staticmethod
    def _is_categorical_from_estimator(
        estimator: LaplaceAdditiveNetRegressor | LaplaceAdditiveNetBinaryClassifier,
        X: npt.ArrayLike,
        features: str | Sequence[str] | int | Sequence[int],
    ) -> list[bool]:
        def is_categorical(index: int) -> bool:
            from laplace_skorch.modules import Categorical

            return isinstance(estimator.module_[index], Categorical)

        return [is_categorical(idx) for idx in _get_column_indices(X, features)]

    def _plot_one_way_partial_dependence(
        self,
        ax: "MplAxes",
        std_factor: float,
        histograms: bool,
        dashed_zero: bool,
        plot_kwargs: dict[str, Any],
        fill_kwargs: dict[str, Any],
        hist_kwargs: dict[str, Any],
    ) -> None:
        assert isinstance(self.data, self._OneWayData)

        if "alpha" not in fill_kwargs:
            fill_kwargs["alpha"] = 0.2

        ax.autoscale(enable=True, axis="x", tight=True)
        ax.set_xlabel(self.data.label)

        if histograms:
            self._plot_one_way_histogram(ax, hist_kwargs)

        ax.set_ylabel("Partial dependence")

        x_values, _ = self.data.x_data
        f_mean, f_std = self.data.y_data

        f_below = f_mean - std_factor * f_std
        f_above = f_mean + std_factor * f_std

        if self.data.is_categorical:
            x_ticks = np.arange(len(x_values)) + 0.5
            ax.set_xticks(x_ticks, x_values)

            ax.step(
                np.r_[0, x_ticks, len(x_values)],
                np.r_[f_mean[0], f_mean, f_mean[-1]],
                where="mid",
                **plot_kwargs,
            )
            ax.fill_between(
                np.r_[0, x_ticks, len(x_values)],
                np.r_[f_below[0], f_below, f_below[-1]],
                np.r_[f_above[0], f_above, f_above[-1]],
                step="mid",
                **fill_kwargs,
            )
        else:
            ax.plot(x_values, f_mean, **plot_kwargs)
            ax.fill_between(x_values, f_below, f_above, **fill_kwargs)

        if dashed_zero:
            ax.axhline(0, ls="--", color="silver", lw=0.8, zorder=-1)

    def _plot_one_way_histogram(
        self, ax: "MplAxes", hist_kwargs: dict[str, Any]
    ) -> None:
        assert isinstance(self.data, self._OneWayData)

        if "color" not in hist_kwargs:
            hist_kwargs["color"] = "#eeeeee"

        ax_twin: "MplAxes" = ax.twinx()  # type: ignore[assignment]
        ax_twin.patch.set_visible(True)

        ax.patch.set_visible(False)
        ax.set_zorder(ax_twin.get_zorder() + 1)

        values, raw_values = self.data.x_data

        if self.data.is_categorical:
            _, counts = np.unique(raw_values, return_counts=True)
            ax_twin.bar(np.arange(len(values)) + 0.5, counts, width=0.5, **hist_kwargs)
        else:
            if "bins" not in hist_kwargs:
                hist_kwargs["bins"] = 40
            ax_twin.hist(raw_values, **hist_kwargs)

        ax_twin.set_yticks([])
        ax_twin.autoscale(enable=True, axis="x", tight=True)

    def _plot_two_way_partial_dependence(
        self,
        ax: "MplAxes",
        display_type: Literal["mean", "stddev"],
        contour_levels: int,
        symmetric_norm: bool,
        contour_kwargs: dict[str, Any],
        heatmap_kwargs: dict[str, Any],
        colorbar_kwargs: dict[str, Any],
    ) -> None:
        import matplotlib.ticker

        assert isinstance(self.data, self._TwoWayData)

        if display_type == "mean":
            Z = self.data.y_data.mean
            label, cmap = "Partial dependence (mean)", "viridis"
        elif display_type == "stddev":
            Z = self.data.y_data.stddev
            label, cmap = "Uncertainty (std. dev.)", "binary"
        else:
            raise ValueError("`display_type` must be 'mean' or 'stddev'.")

        if Z.shape[1] >= Z.shape[0]:
            colorbar_kwargs = {"label": label, "location": "top", **colorbar_kwargs}
        else:
            colorbar_kwargs = {"label": label, "location": "right", **colorbar_kwargs}

        levels_ticker = matplotlib.ticker.MaxNLocator(
            nbins=contour_levels, symmetric=symmetric_norm
        )
        levels = levels_ticker.tick_values(Z.min(), Z.max())

        if all(self.data.is_categorical):
            self._plot_two_way_heatmap(ax, Z, cmap, heatmap_kwargs, colorbar_kwargs)
        elif any(self.data.is_categorical):
            self._plot_two_way_heatmap_with_levels(
                ax, Z, cmap, levels, heatmap_kwargs, colorbar_kwargs
            )
        else:
            self._plot_two_way_contourf(
                ax, Z, cmap, levels, contour_kwargs, colorbar_kwargs
            )

    def _plot_two_way_heatmap(
        self,
        ax: "MplAxes",
        Z: npt.NDArray[np.floating],
        cmap: str,
        heatmap_kwargs: dict[str, Any],
        colorbar_kwargs: dict[str, Any],
    ) -> None:
        import matplotlib.pyplot as plt

        assert isinstance(self.data, self._TwoWayData)

        default_kwargs = dict(interpolation="nearest", cmap=cmap)
        heatmap_kwargs = {**default_kwargs, **heatmap_kwargs}

        image = ax.imshow(Z, **heatmap_kwargs)
        fig: "MplFigure" = ax.figure  # type: ignore[assignment]
        fig.colorbar(image, ax=ax, **colorbar_kwargs)

        ax.set(
            xticks=np.arange(len(self.data.x_data[1])),
            yticks=np.arange(len(self.data.x_data[0])),
            xticklabels=self.data.x_data[1],
            yticklabels=self.data.x_data[0],
            xlabel=self.data.labels[1],
            ylabel=self.data.labels[0],
        )
        plt.setp(ax.get_xticklabels(), rotation=90)

    def _plot_two_way_heatmap_with_levels(
        self,
        ax: "MplAxes",
        Z: npt.NDArray[np.floating],
        cmap: str,
        levels: Sequence[float],
        heatmap_kwargs: dict[str, Any],
        colorbar_kwargs: dict[str, Any],
    ) -> None:
        from matplotlib import colors
        from matplotlib import pyplot as plt

        assert isinstance(self.data, self._TwoWayData)

        default_kwargs = dict(cmap=cmap, interpolation="nearest", aspect="auto")
        heatmap_kwargs = {**default_kwargs, **heatmap_kwargs}

        if self.data.is_categorical[0]:
            ax.set_yticks(
                ticks=np.arange(len(self.data.x_data[0])), labels=self.data.x_data[0]
            )
            xmin, xmax = self.data.x_data[1][0], self.data.x_data[1][-1]
            ymin, ymax = len(self.data.x_data[0]) - 0.5, -0.5
        else:
            ax.set_xticks(
                ticks=np.arange(len(self.data.x_data[1])), labels=self.data.x_data[1]
            )
            xmin, xmax = -0.5, len(self.data.x_data[1]) - 0.5
            ymin, ymax = self.data.x_data[0][0], self.data.x_data[0][-1]

        cmap = plt.colormaps[heatmap_kwargs["cmap"]]
        norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        image = ax.imshow(
            Z, norm=norm, extent=(xmin, xmax, ymin, ymax), **heatmap_kwargs
        )

        fig: "MplFigure" = ax.figure  # type: ignore[assignment]
        fig.colorbar(image, ax=ax, **colorbar_kwargs)
        ax.set(xlabel=self.data.labels[1], ylabel=self.data.labels[0])

    def _plot_two_way_contourf(
        self,
        ax: "MplAxes",
        Z: npt.NDArray[np.floating],
        cmap: str,
        levels: Sequence[float],
        contour_kwargs: dict[str, Any],
        colorbar_kwargs: dict[str, Any],
    ) -> None:
        assert isinstance(self.data, self._TwoWayData)

        default_kwargs = dict(cmap=cmap, antialiased=True)
        contour_kwargs = {**default_kwargs, **contour_kwargs}

        X, Y = np.meshgrid(self.data.x_data[0], self.data.x_data[1])
        contour = ax.contourf(X, Y, Z.T, levels, **contour_kwargs)

        fig: "MplFigure" = ax.figure  # type: ignore[assignment]
        fig.colorbar(contour, ax=ax, **colorbar_kwargs)
        ax.set(xlabel=self.data.labels[0], ylabel=self.data.labels[1])

    def plot(
        self,
        *,
        ax: "MplAxes | None" = None,
        std_factor: float = 2.0,
        histograms: bool = True,
        dashed_zero: bool = True,
        plot_kwargs: dict[str, Any] | None = None,
        fill_kwargs: dict[str, Any] | None = None,
        hist_kwargs: dict[str, Any] | None = None,
        display_type: Literal["mean", "stddev"] = "mean",
        contour_levels: int = 10,
        symmetric_norm: bool = False,
        contour_kwargs: dict[str, Any] | None = None,
        heatmap_kwargs: dict[str, Any] | None = None,
        colorbar_kwargs: dict[str, Any] | None = None,
    ) -> Self:
        check_matplotlib_support(caller_name=f"{self.__class__.__name__}.plot")

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        fill_kwargs = {} if fill_kwargs is None else fill_kwargs
        hist_kwargs = {} if hist_kwargs is None else hist_kwargs
        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        heatmap_kwargs = {} if heatmap_kwargs is None else heatmap_kwargs
        colorbar_kwargs = {} if colorbar_kwargs is None else colorbar_kwargs

        if isinstance(self.data, self._OneWayData):
            self._plot_one_way_partial_dependence(
                ax,
                std_factor,
                histograms,
                dashed_zero,
                plot_kwargs,
                fill_kwargs,
                hist_kwargs,
            )
        elif isinstance(self.data, self._TwoWayData):
            self._plot_two_way_partial_dependence(
                ax,
                display_type,
                contour_levels,
                symmetric_norm,
                contour_kwargs,
                heatmap_kwargs,
                colorbar_kwargs,
            )
        else:
            raise AssertionError

        return self
