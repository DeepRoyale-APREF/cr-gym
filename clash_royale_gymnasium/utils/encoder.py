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
    ARENA_H,
    ARENA_MAP_CHANNELS,
    ARENA_W,
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

    # ── Arena spatial map (C, H, W) ──────────────────────────────────────
    # Channels: 0=ally_tower_hp, 1=enemy_tower_hp,
    #   2=ally_troop_hp, 3=ally_troop_damage, 4=ally_troop_elixir,
    #   5=enemy_troop_hp, 6=enemy_troop_damage, 7=enemy_troop_elixir
    arena_map = _build_arena_map(state)

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
        arena_map=arena_map,
        card_names=card_names,
    )


# ── Arena spatial map builder ─────────────────────────────────────────────────

# Tower positions (tile_x, tile_y) for player 0's perspective
# Player 0 (ally) towers are at the bottom, player 1 (enemy) at the top.
_ALLY_TOWER_POSITIONS = {
    "left_princess": (3, 3),
    "right_princess": (14, 3),
    "king": (8, 0),
}
_ENEMY_TOWER_POSITIONS = {
    "left_princess": (3, 28),
    "right_princess": (14, 28),
    "king": (8, 31),
}

# Princess tower max HP / King tower max HP for normalisation
_PRINCESS_MAX_HP = 3052.0
_KING_MAX_HP = 4824.0


def _build_arena_map(state: State) -> np.ndarray:
    """Build the (C, H, W) spatial arena map from engine state.

    Channels
    --------
    0 : ally_tower_hp       — normalised HP of ally towers at their tile positions
    1 : enemy_tower_hp      — normalised HP of enemy towers at their tile positions
    2 : ally_troop_hp       — HP ratio of ally troops (accumulated per tile)
    3 : ally_troop_damage   — normalised damage of ally troops at their positions
    4 : ally_troop_elixir   — normalised elixir cost of ally troops at their positions
    5 : enemy_troop_hp      — HP ratio of enemy troops (accumulated per tile)
    6 : enemy_troop_damage  — normalised damage of enemy troops at their positions
    7 : enemy_troop_elixir  — normalised elixir cost of enemy troops at their positions

    Parameters
    ----------
    state : State
        Player-perspective game state.

    Returns
    -------
    np.ndarray
        Shape ``(ARENA_MAP_CHANNELS, ARENA_H, ARENA_W)`` float32.
    """
    arena = np.zeros((ARENA_MAP_CHANNELS, ARENA_H, ARENA_W), dtype=np.float32)
    n = state.numbers

    # ── Ally towers (channel 0) ───────────────────────────────────────────
    _place_tower(arena, 0, _ALLY_TOWER_POSITIONS["left_princess"],
                 n.left_princess_hp, _PRINCESS_MAX_HP)
    _place_tower(arena, 0, _ALLY_TOWER_POSITIONS["right_princess"],
                 n.right_princess_hp, _PRINCESS_MAX_HP)
    _place_tower(arena, 0, _ALLY_TOWER_POSITIONS["king"],
                 n.king_hp, _KING_MAX_HP)

    # ── Enemy towers (channel 1) ─────────────────────────────────────────
    _place_tower(arena, 1, _ENEMY_TOWER_POSITIONS["left_princess"],
                 n.left_enemy_princess_hp, _PRINCESS_MAX_HP)
    _place_tower(arena, 1, _ENEMY_TOWER_POSITIONS["right_princess"],
                 n.right_enemy_princess_hp, _PRINCESS_MAX_HP)
    _place_tower(arena, 1, _ENEMY_TOWER_POSITIONS["king"],
                 n.enemy_king_hp, _KING_MAX_HP)

    # ── Ally troops (channels 2, 3, 4) ────────────────────────────────────
    for det in state.allies:
        _place_troop(arena, det, ally_offset=2)

    # ── Enemy troops (channels 5, 6, 7) ──────────────────────────────────
    for det in state.enemies:
        _place_troop(arena, det, ally_offset=5)

    return arena


def _place_tower(
    arena: np.ndarray,
    channel: int,
    pos: tuple[int, int],
    hp: float,
    max_hp: float,
) -> None:
    """Place normalised tower HP at a tile position."""
    x, y = pos
    if 0 <= x < ARENA_W and 0 <= y < ARENA_H and max_hp > 0:
        arena[channel, y, x] = max(hp, 0.0) / max_hp


def _place_troop(
    arena: np.ndarray,
    det: UnitDetection,
    ally_offset: int,
) -> None:
    """Accumulate troop stats at its tile position.

    ally_offset=2 for ally channels {2,3,4}, ally_offset=5 for enemy {5,6,7}.
    """
    tx = int(round(det.position.tile_x))
    ty = int(round(det.position.tile_y))
    if not (0 <= tx < ARENA_W and 0 <= ty < ARENA_H):
        return

    stats = CARD_STATS.get(det.unit.name, {})
    hp_ratio = det.hp / max(det.max_hp, 1)

    arena[ally_offset + 0, ty, tx] += hp_ratio                    # hp
    arena[ally_offset + 1, ty, tx] += stats.get("damage", 0) / 400.0  # damage
    arena[ally_offset + 2, ty, tx] += stats.get("elixir", 0) / 10.0   # elixir
