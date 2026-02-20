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
N_HAND_SIZE: int = 4
N_DECK_SIZE: int = 8
# Legacy alias kept for backwards compatibility
N_CARDS: int = N_HAND_SIZE
N_TILE_X: int = N_WIDE_TILES   # 18
N_TILE_Y: int = N_HEIGHT_TILES  # 32
NOOP_IDX: int = N_DECK_SIZE  # card_idx=8 means "do nothing"


@dataclass(slots=True)
class HierarchicalAction:
    """Output of the hierarchical policy heads.

    Head order: strategy → card (including noop) → tile_x → tile_y.

    ``card_idx`` is a **deck index** (0-7) identifying the card by its
    position in the fixed 8-card deck, **not** a hand-slot index.  This
    lets the model learn card-specific strategies regardless of hand
    rotation.  Index 8 = noop.
    """

    strategy: Strategy
    card_idx: int     # 0-7 = deck card, 8 = noop
    tile_x: int       # 0-17
    tile_y: int       # 0-31

    @property
    def is_noop(self) -> bool:
        return self.card_idx == NOOP_IDX

    def to_engine_action(
        self,
        deck: list[str] | None = None,
        hand: list[str] | None = None,
    ) -> tuple[int, int, int] | None:
        """Convert to engine-compatible ``(tile_x, tile_y, hand_slot_idx)`` or None.

        Parameters
        ----------
        deck : list[str], optional
            The full 8-card deck (needed to map deck index → card name).
        hand : list[str], optional
            The current 4-card hand (needed to find the hand slot).
        """
        if self.is_noop:
            return None
        if deck is not None and hand is not None:
            card_name = deck[self.card_idx]
            if card_name not in hand:
                return None  # card not in hand — treat as noop
            hand_slot = hand.index(card_name)
            return (self.tile_x, self.tile_y, hand_slot)
        # Fallback: treat card_idx as hand-slot (legacy)
        return (self.tile_x, self.tile_y, self.card_idx)


@dataclass(slots=True)
class ActionMask:
    """Boolean masks for each hierarchical head.

    All masks are ``True`` where the action is **valid**.
    """

    strategy: np.ndarray      # (N_STRATEGIES,) bool — always all-True
    card: np.ndarray          # (N_DECK_SIZE + 1,) bool — 0-7 deck cards + noop
    tile_x_per_card: np.ndarray  # (N_DECK_SIZE + 1, N_TILE_X) bool
    tile_y_per_card: np.ndarray  # (N_DECK_SIZE + 1, N_TILE_Y) bool

    def to_dict(self) -> dict[str, np.ndarray]:
        """Flat dict for ``gymnasium.spaces.Dict``."""
        return {
            "strategy": self.strategy,
            "card": self.card,
            "tile_x_per_card": self.tile_x_per_card,
            "tile_y_per_card": self.tile_y_per_card,
        }
