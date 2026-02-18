"""Abstract base for reward components and the composite reward function."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from clash_royale_gymnasium.types.reward_context import RewardContext


class RewardComponent(ABC):
    """A single, weighted reward signal.

    Subclass this to create custom reward components.

    Parameters
    ----------
    weight : float
        Multiplier applied to the raw value returned by :meth:`compute`.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    @abstractmethod
    def compute(self, ctx: RewardContext) -> float:
        """Return **unweighted** reward for this component.

        The environment multiplies by ``self.weight`` automatically.
        """

    def weighted(self, ctx: RewardContext) -> float:
        """Return weighted reward (``weight × compute(ctx)``)."""
        return self.weight * self.compute(ctx)

    def reset(self) -> None:
        """Called at the start of each episode (optional override)."""


class RewardFunction:
    """Composite reward: sum of weighted :class:`RewardComponent` instances.

    Parameters
    ----------
    components : list[RewardComponent]
        Ordered list of reward signals.

    Example
    -------
    >>> rf = RewardFunction([
    ...     DamageComponent(weight=1.0),
    ...     ElixirComponent(weight=0.5),
    ...     TerminalComponent(weight=1.0),
    ... ])
    """

    def __init__(self, components: List[RewardComponent]) -> None:
        self.components = components

    def __call__(self, ctx: RewardContext) -> float:
        return sum(c.weighted(ctx) for c in self.components)

    def reset(self) -> None:
        """Reset all components at episode boundary."""
        for c in self.components:
            c.reset()

    def breakdown(self, ctx: RewardContext) -> dict[str, float]:
        """Return per-component weighted values (for logging)."""
        return {type(c).__name__: c.weighted(ctx) for c in self.components}
