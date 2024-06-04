import datetime
import math
import time
from collections.abc import Callable, Sequence
from typing import Any, Final, Self

from skorch import NeuralNet
from skorch.callbacks import Callback, PrintLog, WandbLogger
from skorch.callbacks.logging import filter_log_keys


class ElapsedTimer(Callback):
    """Keeps track of elapsed time in seconds since the start of training."""

    start_time_: float | None = None

    def on_train_begin(self, net: NeuralNet, **kwargs: Any) -> None:
        self.start_time_ = time.time()

    def on_epoch_end(self, net: NeuralNet, **kwargs: Any) -> None:
        assert self.start_time_ is not None

        elapsed = round(time.time() - self.start_time_)
        elapsed = datetime.timedelta(seconds=elapsed)
        net.history.record("elapsed", elapsed)


class ElapsedPrintLog(PrintLog):
    """Prints the training progress every `interval` seconds."""

    interval: Final[float]
    previous_time_: float
    previous_epoch_: int

    def __init__(
        self,
        keys_ignored: str | list[str] | None = None,
        sink: Callable[[str], None] = print,
        tablefmt: str = "simple",
        floatfmt: str = ".4f",
        stralign: str = "right",
        interval: float = 10.0,  # sec.
    ) -> None:
        super().__init__(keys_ignored, sink, tablefmt, floatfmt, stralign)

        self.interval = interval

    def initialize(self) -> Self:
        super().initialize()

        # Don't print the epoch duration.
        self.keys_ignored_.add("dur")

        # Don't print the hyperparameter values.
        self.keys_ignored_.add("prior_precision")
        self.keys_ignored_.add("sigma_noise")
        return self

    def _sorted_keys(self, keys: Sequence[str]) -> list[str]:
        sorted_keys = list()

        # Make sure `epoch` comes first.
        if ("epoch" in keys) and ("epoch" not in self.keys_ignored_):
            sorted_keys.append("epoch")

        # Ignore keys like `*_best` or `event_*`.
        for key in filter_log_keys(sorted(keys), keys_ignored=self.keys_ignored_):
            if key != "elapsed":
                sorted_keys.append(key)

        # Add `event_*` keys.
        for key in sorted(keys):
            if key.startswith("event_") and (key not in self.keys_ignored_):
                sorted_keys.append(key)

        # Make sure `elapsed` comes last.
        if ("elapsed" in keys) and ("elapsed" not in self.keys_ignored_):
            sorted_keys.append("elapsed")

        return sorted_keys

    def on_train_begin(self, net: NeuralNet, **kwargs: Any) -> None:
        self.previous_time_ = -math.inf
        self.previous_epoch_ = 0

    def on_epoch_end(self, net: NeuralNet, **kwargs: Any) -> None:
        current_time = time.time()

        # Wait `interval` before displaying a new epoch.
        if (current_time - self.previous_time_) > self.interval:
            super().on_epoch_end(net)
            self.previous_time_ = current_time
            self.previous_epoch_ = net.history[-1, "epoch"]

    def on_train_end(self, net: NeuralNet, **kwargs: Any) -> None:
        try:
            final_epoch = net.history[-1, "epoch"]
        except IndexError:
            return

        # Display the final epoch if it was not already displayed.
        if final_epoch != self.previous_epoch_:
            super().on_epoch_end(net)


class NoGradientsWandbLogger(WandbLogger):
    """Logs metrics to Weights & Biases without the gradient histograms."""

    def on_train_begin(self, net: NeuralNet, **kwargs: Any) -> None:
        pass  # @override `WandbLogger.on_train_begin`
