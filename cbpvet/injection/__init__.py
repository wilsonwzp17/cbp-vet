"""Injection-side extensions: pair mechanics and the dual-inversion harness."""

from . import pair_model
from .dual_injector import DualInjector, default_epoch_sampler

__all__ = ["pair_model", "DualInjector", "default_epoch_sampler"]

from .rebalance import rebalance_negatives  # noqa: E402
__all__.append("rebalance_negatives")
