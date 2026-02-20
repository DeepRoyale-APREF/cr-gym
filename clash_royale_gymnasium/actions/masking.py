"""Action masking — compute per-head masks from engine state.

Masks enforce **only valid actions** so the policy never proposes an
illegal placement.  The hierarchical heads are masked autoregressively:

1. **strategy** — always all-valid (the agent can choose any intent).
2. **card** — deck cards that are in hand AND affordable, + noop.
3. **tile_x / tile_y per card** — per-card placement masks so the model
   learns card-specific tile rules (spells -> anywhere, troops -> own side
   + pocket if unlocked).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from clash_royale_engine.core.state import State
from clash_royale_engine.utils.constants import (
    BRIDGE_Y,
    CARD_STATS,
    LANE_DIVIDER_X,
    N_HEIGHT_TILES,
    N_WIDE_TILES,
    POCKET_DEPTH,
    RIVER_Y_MAX,
)

from clash_royale_gymnasium.types.actions import (
    ActionMask,
    HierarchicalAction,
    N_DECK_SIZE,
    N_HAND_SIZE,
    N_STRATEGIES,
    N_TILE_X,
    N_TILE_Y,
    NOOP_IDX,
    Strategy,
)


def _troop_tile_masks(
    *,
    enemy_left_princess_dead: bool,
    enemy_right_princess_dead: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tile masks for a troop card (own side + pocket)."""
    tx = np.ones(N_TILE_X, dtype=bool)
    ty = np.zeros(N_TILE_Y, dtype=bool)

    for y in range(BRIDGE_Y):
        ty[y] = True

    pocket_min_y = int(RIVER_Y_MAX)
    pocket_max_y = int(RIVER_Y_MAX) + POCKET_DEPTH - 1
    if enemy_left_princess_dead or enemy_right_princess_dead:
        for y in range(pocket_min_y, pocket_max_y + 1):
            if y < N_TILE_Y:
                ty[y] = True

    return tx, ty


def _spell_tile_masks() -> Tuple[np.ndarray, np.ndarray]:
    """Tile masks for a spell card (anywhere)."""
    return np.ones(N_TILE_X, dtype=bool), np.ones(N_TILE_Y, dtype=bool)


def compute_action_mask(
    state: State,
    *,
    enemy_left_princess_dead: bool = False,
    enemy_right_princess_dead: bool = False,
) -> ActionMask:
    """Build masks for all hierarchical heads.

    Card indices are **deck-indexed** (0-7) referencing the full 8-card
    deck, not the 4-card hand.  A deck card is valid only when it is
    currently in the hand AND affordable.  Tile masks are computed
    per-card so the autoregressive model can condition placement on the
    chosen card.

    Parameters
    ----------
    state : State
        The agent's (partial) game state (must include ``state.deck``).
    enemy_left_princess_dead, enemy_right_princess_dead : bool
        Whether opponent princess towers have been destroyed (enables pocket).
    """
    strategy_mask = np.ones(N_STRATEGIES, dtype=bool)

    n_options = N_DECK_SIZE + 1  # 8 deck + 1 noop
    card_mask = np.zeros(n_options, dtype=bool)
    card_mask[NOOP_IDX] = True

    hand_names: List[str] = [c.name for c in state.cards]
    ready_set = set(state.ready)
    deck = state.deck if state.deck else [c.name for c in state.cards]

    for deck_idx, deck_card_name in enumerate(deck[:N_DECK_SIZE]):
        if deck_card_name in hand_names:
            hand_slot = hand_names.index(deck_card_name)
            if hand_slot in ready_set:
                card_mask[deck_idx] = True

    tile_x_per_card = np.zeros((n_options, N_TILE_X), dtype=bool)
    tile_y_per_card = np.zeros((n_options, N_TILE_Y), dtype=bool)

    for deck_idx, deck_card_name in enumerate(deck[:N_DECK_SIZE]):
        stats = CARD_STATS.get(deck_card_name)
        if stats is None:
            continue
        is_spell = stats.get("is_spell", False)
        if is_spell:
            tx, ty = _spell_tile_masks()
        else:
            tx, ty = _troop_tile_masks(
                enemy_left_princess_dead=enemy_left_princess_dead,
                enemy_right_princess_dead=enemy_right_princess_dead,
            )
        tile_x_per_card[deck_idx] = tx
        tile_y_per_card[deck_idx] = ty

    tile_x_per_card[NOOP_IDX, :] = True
    tile_y_per_card[NOOP_IDX, :] = True

    return ActionMask(
        strategy=strategy_mask,
        card=card_mask,
        tile_x_per_card=tile_x_per_card,
        tile_y_per_card=tile_y_per_card,
    )


def validate_hierarchical(
    action: HierarchicalAction,
    state: State,
    *,
    enemy_left_princess_dead: bool = False,
    enemy_right_princess_dead: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Check whether a hierarchical action is valid.

    Returns ``(True, None)`` if valid, ``(False, reason)`` otherwise.
    """
    if action.is_noop:
        return True, None

    deck = state.deck if state.deck else [c.name for c in state.cards]

    if action.card_idx < 0 or action.card_idx >= len(deck):
        return False, f"Invalid card index: {action.card_idx}"

    card_name = deck[action.card_idx]
    hand_names = [c.name for c in state.cards]

    if card_name not in hand_names:
        return False, f"Card {card_name} not in hand (deck_idx={action.card_idx})"

    hand_slot = hand_names.index(card_name)
    if hand_slot not in state.ready:
        return False, f"Card {card_name} not affordable (hand_slot={hand_slot})"

    stats = CARD_STATS.get(card_name)
    if stats is None:
        return False, f"Unknown card: {card_name}"

    if not (0 <= action.tile_x < N_TILE_X and 0 <= action.tile_y < N_TILE_Y):
        return False, f"Tile ({action.tile_x}, {action.tile_y}) out of bounds"

    is_spell = stats.get("is_spell", False)
    if not is_spell and action.tile_y >= BRIDGE_Y:
        in_left = action.tile_x < LANE_DIVIDER_X
        lane_ok = (in_left and enemy_left_princess_dead) or (
            not in_left and enemy_right_princess_dead
        )
        if not lane_ok:
            return False, "Cannot place troop on enemy side -- tower still standing"

        pocket_min = int(RIVER_Y_MAX)
        pocket_max = int(RIVER_Y_MAX) + POCKET_DEPTH - 1
        if not (pocket_min <= action.tile_y <= pocket_max):
            return False, "Troop outside pocket depth"

    return True, None
