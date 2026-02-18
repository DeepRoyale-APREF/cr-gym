"""Action types — hierarchical actions and masking for AlphaStar-like models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from clash_royale_engine.utils.constants import N_HEIGHT_TILES, N_WIDE_TILES


class Strategy(IntEnum):
    """High-level strategic intent (first head of the hierarchical policy)."""

    AGGRESSIVE = 0
    DEFENSIVE = 1
    FARMING = 2


N_STRATEGIES: int = len(Strategy)
N_CARDS: int = 4
N_TILE_X: int = N_WIDE_TILES   # 18
N_TILE_Y: int = N_HEIGHT_TILES  # 32
NOOP_IDX: int = N_CARDS  # card_idx=4 means "do nothing"


@dataclass(slots=True)
class HierarchicalAction:
    """Output of the hierarchical policy heads.

    Head order: strategy → card (including noop) → tile_x → tile_y.
    """

    strategy: Strategy
    card_idx: int     # 0-3 = play card, 4 = noop
    tile_x: int       # 0-17
    tile_y: int       # 0-31

    @property
    def is_noop(self) -> bool:
        return self.card_idx == NOOP_IDX

    def to_engine_action(self) -> tuple[int, int, int] | None:
        """Convert to engine-compatible ``(tile_x, tile_y, card_idx)`` or None."""
        if self.is_noop:
            return None
        return (self.tile_x, self.tile_y, self.card_idx)


@dataclass(slots=True)
class ActionMask:
    """Boolean masks for each hierarchical head.

    All masks are ``True`` where the action is **valid**.
    """

    strategy: np.ndarray  # (N_STRATEGIES,) bool — always all-True
    card: np.ndarray      # (N_CARDS + 1,) bool — 0-3 playable cards + noop
    tile_x: np.ndarray    # (N_TILE_X,) bool — valid x positions
    tile_y: np.ndarray    # (N_TILE_Y,) bool — valid y positions

    def to_dict(self) -> dict[str, np.ndarray]:
        """Flat dict for ``gymnasium.spaces.Dict``."""
        return {
            "strategy": self.strategy,
            "card": self.card,
            "tile_x": self.tile_x,
            "tile_y": self.tile_y,
        }
