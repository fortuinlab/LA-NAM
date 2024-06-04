import functools
import itertools
import operator
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Final, TypeAlias, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class Numerical(nn.Sequential):
    """A feature network attending to one or more numerical features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_dims: Sequence[int] = [64],
        activation_cls: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.append(nn.Linear(in_features, hidden_dims[0]))
        self.append(activation_cls())

        for in_hidden, out_hidden in itertools.pairwise(hidden_dims):
            self.append(nn.Linear(in_hidden, out_hidden))
            self.append(activation_cls())

        self.append(nn.Linear(hidden_dims[-1], out_features))

    @property
    def in_features(self) -> int:
        return cast(nn.Linear, self[0]).in_features

    @property
    def out_features(self) -> int:
        return cast(nn.Linear, self[-1]).out_features


class Categorical(nn.Sequential):
    """A feature network attending to one or more categorical features.s"""

    _cumprod: torch.LongTensor

    in_features: Final[int]
    in_features_encoded: Final[int]
    out_features: Final[int]

    def __init__(self, num_classes: int | Sequence[int], out_features: int) -> None:
        super().__init__()

        if isinstance(num_classes, int):
            num_classes = [num_classes]
        self._register_cumprod(num_classes)

        self.in_features = len(num_classes)
        self.in_features_encoded = functools.reduce(operator.mul, num_classes)
        self.out_features = out_features

        self.append(
            nn.Linear(
                in_features=self.in_features_encoded, out_features=self.out_features
            )
        )

    def _register_cumprod(self, num_classes: Sequence[int]) -> None:
        cumprod = torch.cat(
            [
                torch.tensor([1], dtype=torch.int64),
                torch.cumprod(torch.tensor(num_classes[:-1], dtype=torch.int64), dim=0),
            ]
        )

        self.register_buffer("_cumprod", cumprod.long())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        encoded = torch.sum(input.long() * self._cumprod, dim=1)
        encoded = F.one_hot(encoded, num_classes=self.in_features_encoded).float()

        return super().forward(encoded)


class Mixed(Numerical):
    """A feature network attending to mixed features types."""

    _num_classes: Sequence[int]

    def __init__(
        self,
        num_classes: Sequence[int],
        out_features: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_features=sum(num_classes), out_features=out_features, **kwargs
        )

        self._num_classes = num_classes

    @property
    def in_features(self) -> int:
        return len(self._num_classes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        encoded = torch.hstack(
            [
                F.one_hot(input[:, pos].long(), num) if num > 1 else input[:, [pos]]
                for pos, num in enumerate(self._num_classes)
            ]
        )

        return super().forward(encoded)


class Router(nn.Module):
    """Routes input features to feature networks and stacks the outputs."""

    _RouterModule: TypeAlias = Numerical | Categorical | Mixed
    _RouterMap: TypeAlias = (
        Mapping[int, _RouterModule] | Mapping[tuple[int, ...], _RouterModule]
    )

    _keys: list[tuple[int, ...]]

    def __init__(self, modules: _RouterMap | None = None) -> None:
        super().__init__()

        self._keys = list()

        if modules:
            self.update(modules)

    @staticmethod
    def _key_to_name(key: tuple[int, ...]) -> str:
        return "_" + "_".join(map(str, key))

    def __getitem__(self, key: int | tuple[int, ...]) -> _RouterModule:
        if isinstance(key, int):
            key = (key,)

        return cast(Router._RouterModule, self._modules[self._key_to_name(key)])

    def __setitem__(self, key: int | tuple[int, ...], module: _RouterModule) -> None:
        if isinstance(key, int):
            key = (key,)

        if key in self:
            del self[key]

        self._keys.append(key)
        self.register_module(self._key_to_name(key), module)

    def __delitem__(self, key: int | tuple[int, ...]) -> None:
        if isinstance(key, int):
            key = (key,)

        self._keys.remove(key)
        del self._modules[self._key_to_name(key)]

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return iter(self._keys)

    def __contains__(self, key: tuple[int, ...]) -> bool:
        return key in self._keys

    def keys(self) -> Iterable[tuple[int, ...]]:
        return self._keys.copy()

    def values(self) -> Iterable[_RouterModule]:
        return cast(Iterable[Router._RouterModule], self._modules.values())

    def items(self) -> Iterable[tuple[tuple[int, ...], _RouterModule]]:
        return zip(self.keys(), self.values(), strict=True)

    def update(self, modules: _RouterMap) -> None:
        for key, module in modules.items():
            self[key] = module

    @property
    def in_features(self) -> int:
        return max(max(key) for key in self.keys()) + 1

    @functools.cached_property
    def out_features(self) -> int:
        return next(mod.out_features for mod in self.values())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.hstack([self[key](input[..., key]) for key in self])
