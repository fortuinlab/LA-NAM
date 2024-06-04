import itertools
from collections.abc import Iterable, Sequence
from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import scipy.stats as stats
import torch


@overload
def log_prob_density(
    y_true: npt.ArrayLike, y_mean: npt.ArrayLike, y_std: float | npt.ArrayLike
) -> float: ...


@overload
def log_prob_density(
    y_true: npt.ArrayLike,
    y_mean: npt.ArrayLike,
    y_std: float | npt.ArrayLike,
    *,
    reduction: Literal["mean", "sum"],
) -> float: ...


@overload
def log_prob_density(
    y_true: npt.ArrayLike,
    y_mean: npt.ArrayLike,
    y_std: float | npt.ArrayLike,
    *,
    reduction: Literal["none"],
) -> npt.NDArray[np.floating]: ...


def log_prob_density(
    y_true: npt.ArrayLike,
    y_mean: npt.ArrayLike,
    y_std: float | npt.ArrayLike,
    *,
    scale: float = 1.0,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> float | npt.NDArray[np.floating]:
    """Log-likelihood of Gaussian distributions (for classification)."""

    log_pdf = stats.norm.logpdf(
        np.ravel(y_true) * scale, np.ravel(y_mean) * scale, np.ravel(y_std) * scale
    )

    if reduction == "mean":
        return np.mean(log_pdf)
    elif reduction == "sum":
        return np.sum(log_pdf)
    return np.reshape(log_pdf, np.shape(y_true))


@overload
def log_prob_mass(y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> float: ...


@overload
def log_prob_mass(
    y_true: npt.ArrayLike, y_prob: npt.ArrayLike, *, reduction: Literal["mean", "sum"]
) -> float: ...


@overload
def log_prob_mass(
    y_true: npt.ArrayLike, y_prob: npt.ArrayLike, *, reduction: Literal["none"]
) -> npt.NDArray[np.floating]: ...


def log_prob_mass(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    *,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> float | npt.NDArray[np.floating]:
    """Log-likelihood of categorical distributions (for regression)."""

    num_samples, num_classes = np.shape(y_prob)

    y_onehot = np.zeros((num_samples, num_classes), dtype=int)
    y_onehot[np.arange(num_samples), np.asarray(y_true, int)] = 1

    log_pmf = stats.multinomial.logpmf(y_onehot, n=1, p=y_prob)

    if reduction == "mean":
        return np.mean(log_pmf)
    elif reduction == "sum":
        return np.sum(log_pmf)
    return np.reshape(log_pmf, np.shape(y_true))


def topk_unravel(
    input: torch.Tensor, k: int, *, largest: bool = True, sorted: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the `k` largest elements of `input` with unraveled indices."""

    values, raveled = torch.topk(input.ravel(), k, -1, largest, sorted)
    unraveled = torch.stack(torch.unravel_index(raveled, input.shape), dim=1)
    return values, unraveled


def powerset(s: Sequence[int]) -> Iterable[tuple[int, ...]]:
    """Iterable powerset of a `Sequence[int]` (skipping the empty set)."""

    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )
