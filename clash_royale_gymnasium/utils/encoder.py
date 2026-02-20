"""Encode engine :class:`State` to :class:`Observation` (partial, dict-based).

All encoding respects **fog-of-war**:
- Enemy elixir is excluded from scalars.
- Only enemies present in ``state.enemies`` are encoded (the engine already
  filters by tower/troop vision range).
- Opponent hand/deck is never exposed.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.core.state import State, UnitDetection
from clash_royale_engine.utils.constants import CARD_STATS, CARD_VOCAB

from clash_royale_gymnasium.types.observations import (
    CARD_FEATURE_DIM,
    DECK_SIZE,
    MAX_TROOPS,
    SCALAR_DIM,
    TROOP_FEATURE_DIM,
    CardInfo,
    Observation,
    ScalarFeatures,
    TroopInfo,
)

# ── Card vocabulary lookup ────────────────────────────────────────────────────
_CARD_NAME_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CARD_VOCAB)}

# ── Category / target / transport string→int maps ────────────────────────────
_CATEGORY_MAP: Dict[str, int] = {"troop": 0, "building": 1, "spell": 2}
_TARGET_MAP: Dict[str, int] = {"all": 0, "ground": 1, "buildings": 2}
_TRANSPORT_MAP: Dict[str, int] = {"ground": 0, "air": 1}


def _encode_detection(det: UnitDetection, is_ally: bool) -> TroopInfo:
    """Convert a single :class:`UnitDetection` to a :class:`TroopInfo`."""
    name_idx = _CARD_NAME_TO_IDX.get(det.unit.name, 0)
    stats = CARD_STATS.get(det.unit.name, {})

    hp_ratio = det.hp / max(det.max_hp, 1)
    cost = stats.get("elixir", 0) / 10.0
    damage = stats.get("damage", 0) / 400.0
    hit_speed = stats.get("hit_speed", 1.0) / 2.0
    attack_range = stats.get("range", 1.0) / 7.0
    speed = stats.get("speed", 60.0) / 100.0
    hitbox = stats.get("hitbox_radius", 0.5) / 2.0

    return TroopInfo(
        name_idx=name_idx,
        category=_CATEGORY_MAP.get(det.unit.category, 0),
        target=_TARGET_MAP.get(det.unit.target, 0),
        transport=_TRANSPORT_MAP.get(det.unit.transport, 0),
        tile_x=float(det.position.tile_x),
        tile_y=float(det.position.tile_y),
        hp_ratio=hp_ratio,
        is_ally=is_ally,
        cost=cost,
        damage=damage,
        hit_speed=hit_speed,
        attack_range=attack_range,
        speed=speed,
        hitbox_radius=hitbox,
    )


def encode_observation(
    state: State,
    engine: ClashRoyaleEngine,
    player_id: int = 0,
) -> Observation:
    """Build a partial :class:`Observation` from engine state.

    Parameters
    ----------
    state : State
        The player-perspective state (already fog-filtered by the engine).
    engine : ClashRoyaleEngine
        For accessing leaked elixir, troop elixir value, and frame info.
    player_id : int
        Which player's perspective (0 or 1).
    """
    # ── Troops (allies + visible enemies) ─────────────────────────────────
    troop_infos: List[TroopInfo] = []

    for det in state.allies:
        troop_infos.append(_encode_detection(det, is_ally=True))
    for det in state.enemies:
        troop_infos.append(_encode_detection(det, is_ally=False))

    # Pad / truncate to MAX_TROOPS
    troops = np.zeros((MAX_TROOPS, TROOP_FEATURE_DIM), dtype=np.float32)
    troop_mask = np.zeros(MAX_TROOPS, dtype=bool)

    for i, ti in enumerate(troop_infos[:MAX_TROOPS]):
        troops[i] = ti.to_array()
        troop_mask[i] = True

    # ── Scalars (no enemy elixir — fog-of-war) ───────────────────────────
    total_frames = engine.scheduler.game_duration * engine.fps
    frame_ratio = engine.current_frame / max(total_frames, 1) if total_frames > 0 else 0.0

    scalars = ScalarFeatures(
        elixir=state.numbers.elixir,
        left_princess_hp=state.numbers.left_princess_hp,
        right_princess_hp=state.numbers.right_princess_hp,
        king_hp=state.numbers.king_hp,
        left_enemy_princess_hp=state.numbers.left_enemy_princess_hp,
        right_enemy_princess_hp=state.numbers.right_enemy_princess_hp,
        enemy_king_hp=state.numbers.enemy_king_hp,
        time_remaining=state.numbers.time_remaining,
        king_active=state.numbers.king_active,
        enemy_king_active=state.numbers.enemy_king_active,
        is_double_elixir=state.numbers.is_double_elixir,
        is_overtime=state.numbers.is_overtime,
        overtime_remaining=state.numbers.overtime_remaining,
        troop_elixir_value=engine.get_alive_troop_elixir_value(player_id),
        leaked_elixir=engine.get_leaked_elixir(player_id),
        frame_ratio=frame_ratio,
    )

    # ── Cards (all 8 deck cards with hand/affordability info) ───────────
    deck = state.deck if state.deck else [c.name for c in state.cards]
    hand_names: List[str] = [c.name for c in state.cards]
    ready_set = set(state.ready)

    cards_arr = np.zeros((DECK_SIZE, CARD_FEATURE_DIM), dtype=np.float32)
    card_names: List[str] = []

    for i, deck_card_name in enumerate(deck[:DECK_SIZE]):
        stats = CARD_STATS.get(deck_card_name, {})
        is_in_hand = deck_card_name in hand_names
        # A card is affordable only if it's in hand and its hand slot is ready
        is_affordable = False
        if is_in_hand:
            hand_slot = hand_names.index(deck_card_name)
            is_affordable = hand_slot in ready_set
        ci = CardInfo(
            name_idx=_CARD_NAME_TO_IDX.get(deck_card_name, 0),
            cost=stats.get("elixir", 0),
            is_spell=stats.get("is_spell", False),
            is_in_hand=is_in_hand,
            is_affordable=is_affordable,
        )
        cards_arr[i] = ci.to_array()
        card_names.append(deck_card_name)

    return Observation(
        troops=troops,
        troop_mask=troop_mask,
        scalars=scalars.to_array(),
        cards=cards_arr,
        card_names=card_names,
    )
