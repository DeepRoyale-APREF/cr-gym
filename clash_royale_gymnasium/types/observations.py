"""Observation types — structured dict observation for AlphaStar-like models.

Observations are **partial** by design:
- Enemy elixir is hidden (always 0).
- Only enemies **visible** to the player's towers/troops are included.
- The opponent's hand/deck is never exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_TROOPS: int = 100
"""Maximum number of troops in the observation arrays (padded with zeros)."""

TROOP_FEATURE_DIM: int = 14
"""Per-troop feature vector size: [name_idx, category, target, transport,
tile_x, tile_y, hp_ratio, is_ally, cost, damage, hit_speed, range, speed,
hitbox_radius].  All normalised to roughly [0, 1]."""

SCALAR_DIM: int = 16
"""Scalar features: elixir(1) + tower_hp(6) + time(1)
+ king_active(2) + double_elixir(1) + overtime(1) + overtime_remaining(1)
+ troop_elixir_value(1) + leaked_elixir(1) + current_frame_ratio(1).

Note: ``enemy_elixir`` is **excluded** (fog-of-war)."""

DECK_SIZE: int = 8
"""Number of cards in the deck (observation encodes ALL deck cards)."""

CARD_FEATURE_DIM: int = 5
"""Per-card feature: [name_idx, cost, is_spell, is_in_hand, is_affordable]."""


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TroopInfo:
    """Single troop in the arena — ready for embedding."""

    name_idx: int        # index into the card vocabulary
    category: int        # 0=troop, 1=building
    target: int          # 0=all, 1=ground, 2=buildings
    transport: int       # 0=ground, 1=air
    tile_x: float        # [0, 17]
    tile_y: float        # [0, 31]
    hp_ratio: float      # current_hp / max_hp  ∈ [0, 1]
    is_ally: bool
    cost: float          # elixir cost normalised /10
    damage: float        # normalised /400
    hit_speed: float     # seconds normalised /2
    attack_range: float  # tiles normalised /7
    speed: float         # normalised /100
    hitbox_radius: float # normalised /2

    def to_array(self) -> np.ndarray:
        """Return a (TROOP_FEATURE_DIM,) float32 array."""
        return np.array(
            [
                self.name_idx / 8.0,
                float(self.category),
                self.target / 2.0,
                float(self.transport),
                self.tile_x / 17.0,
                self.tile_y / 31.0,
                self.hp_ratio,
                float(self.is_ally),
                self.cost,
                self.damage,
                self.hit_speed,
                self.attack_range,
                self.speed,
                self.hitbox_radius,
            ],
            dtype=np.float32,
        )


@dataclass(slots=True)
class CardInfo:
    """A card in the deck — encodes identity, cost, and availability."""

    name_idx: int      # vocabulary index
    cost: int          # raw elixir cost
    is_spell: bool
    is_in_hand: bool   # True if this deck card is currently in the 4-card hand
    is_affordable: bool  # True if in hand AND affordable

    def to_array(self) -> np.ndarray:
        """Return a (CARD_FEATURE_DIM,) float32 array."""
        return np.array(
            [
                self.name_idx / 8.0,
                self.cost / 10.0,
                float(self.is_spell),
                float(self.is_in_hand),
                float(self.is_affordable),
            ],
            dtype=np.float32,
        )


@dataclass(slots=True)
class ScalarFeatures:
    """All non-entity, non-card numeric features.

    Partial observability: enemy elixir is **not** included.
    Enemy tower HP **is** visible (towers are always on the map in CR).
    """

    elixir: float
    # enemy_elixir omitted — fog-of-war
    left_princess_hp: float
    right_princess_hp: float
    king_hp: float
    left_enemy_princess_hp: float
    right_enemy_princess_hp: float
    enemy_king_hp: float
    time_remaining: float
    king_active: bool
    enemy_king_active: bool
    is_double_elixir: bool
    is_overtime: bool
    overtime_remaining: float
    troop_elixir_value: float   # own troops only
    leaked_elixir: float        # own leaked elixir only
    frame_ratio: float          # current_frame / total_frames ∈ [0, 1]

    def to_array(self) -> np.ndarray:
        """Return a (SCALAR_DIM,) float32 array."""
        return np.array(
            [
                self.elixir / 10.0,
                self.left_princess_hp / 1400.0,
                self.right_princess_hp / 1400.0,
                self.king_hp / 2400.0,
                self.left_enemy_princess_hp / 1400.0,
                self.right_enemy_princess_hp / 1400.0,
                self.enemy_king_hp / 2400.0,
                self.time_remaining / 240.0,
                float(self.king_active),
                float(self.enemy_king_active),
                float(self.is_double_elixir),
                float(self.is_overtime),
                self.overtime_remaining / 60.0,
                self.troop_elixir_value / 30.0,
                min(self.leaked_elixir / 20.0, 1.0),
                self.frame_ratio,
            ],
            dtype=np.float32,
        )


@dataclass(slots=True)
class Observation:
    """Structured observation dict used as the Gymnasium observation.

    **Partial observability** (like a real CR match):
    - Enemy elixir is unknown (not included in scalars).
    - Only **visible** enemies are in ``troops`` (filtered by tower/troop sight).
    - The opponent's hand and deck are never exposed.
    - Own hand cards are fully visible to the agent.

    Designed for AlphaStar-like architectures:
    - ``troops``:  (MAX_TROOPS, TROOP_FEATURE_DIM) — positional-encoded entities
    - ``troop_mask``: (MAX_TROOPS,) bool — True where a real troop exists
    - ``scalars``: (SCALAR_DIM,) — feed-forward scalar features
    - ``cards``:   (DECK_SIZE, CARD_FEATURE_DIM) — all 8 deck cards
    """

    troops: np.ndarray           # (MAX_TROOPS, TROOP_FEATURE_DIM) float32
    troop_mask: np.ndarray       # (MAX_TROOPS,) bool
    scalars: np.ndarray          # (SCALAR_DIM,) float32
    cards: np.ndarray            # (DECK_SIZE, CARD_FEATURE_DIM) float32
    card_names: List[str] = field(default_factory=list)  # for debugging

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to flat dict suitable for ``gymnasium.spaces.Dict``."""
        return {
            "troops": self.troops,
            "troop_mask": self.troop_mask,
            "scalars": self.scalars,
            "cards": self.cards,
        }
