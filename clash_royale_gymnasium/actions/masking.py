"""Action masking — compute per-head masks from engine state.

Masks enforce **only valid actions** so the policy never proposes an
illegal placement.  The hierarchical heads are masked sequentially:

1. **strategy** — always all-valid (the agent can choose any intent).
2. **card** — affordable hand slots + noop.  Noop is always valid.
3. **tile_x / tile_y** — own-side tiles + unlocked pocket tiles.
   Spells can target any tile.
"""

from __future__ import annotations

from typing import Optional, Tuple

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
    N_CARDS,
    N_STRATEGIES,
    N_TILE_X,
    N_TILE_Y,
    NOOP_IDX,
    Strategy,
)


def compute_action_mask(
    state: State,
    *,
    enemy_left_princess_dead: bool = False,
    enemy_right_princess_dead: bool = False,
) -> ActionMask:
    """Build masks for all four hierarchical heads.

    Parameters
    ----------
    state : State
        The agent's (partial) game state.
    enemy_left_princess_dead, enemy_right_princess_dead : bool
        Whether opponent princess towers have been destroyed (enables pocket).
    """
    # ── Strategy: always all valid ────────────────────────────────────────
    strategy_mask = np.ones(N_STRATEGIES, dtype=bool)

    # ── Card mask: playable slots + noop ──────────────────────────────────
    card_mask = np.zeros(N_CARDS + 1, dtype=bool)
    card_mask[NOOP_IDX] = True  # noop always valid

    for idx in state.ready:
        if 0 <= idx < N_CARDS:
            card_mask[idx] = True

    # ── Tile masks: valid placement zones ─────────────────────────────────
    # Compute the union of valid tiles across all playable cards.
    # A tile is valid if ANY playable card can be placed there.
    tile_x_mask = np.zeros(N_TILE_X, dtype=bool)
    tile_y_mask = np.zeros(N_TILE_Y, dtype=bool)

    has_playable_spell = False
    has_playable_troop = False

    for idx in state.ready:
        if idx < 0 or idx >= N_CARDS:
            continue
        card = state.cards[idx]
        stats = CARD_STATS.get(card.name)
        if stats is None:
            continue
        if stats.get("is_spell", False):
            has_playable_spell = True
        else:
            has_playable_troop = True

    if has_playable_spell:
        # Spells can target anywhere on the grid
        tile_x_mask[:] = True
        tile_y_mask[:] = True
    elif has_playable_troop:
        # Troops: own side always valid
        tile_x_mask[:] = True  # all columns are valid on own side
        for y in range(BRIDGE_Y):
            tile_y_mask[y] = True

        # Pocket tiles (enemy side, post-river)
        pocket_min_y = int(RIVER_Y_MAX)  # 17
        pocket_max_y = int(RIVER_Y_MAX) + POCKET_DEPTH - 1  # 19

        if enemy_left_princess_dead or enemy_right_princess_dead:
            for y in range(pocket_min_y, pocket_max_y + 1):
                if y < N_TILE_Y:
                    tile_y_mask[y] = True
            # x is already all-True (filtered per-card at validation)

    # If nothing is playable, only noop — tiles don't matter but mark
    # at least one so the model has a valid sample.
    if not tile_x_mask.any():
        tile_x_mask[0] = True
    if not tile_y_mask.any():
        tile_y_mask[0] = True

    return ActionMask(
        strategy=strategy_mask,
        card=card_mask,
        tile_x=tile_x_mask,
        tile_y=tile_y_mask,
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

    if action.card_idx < 0 or action.card_idx >= N_CARDS:
        return False, f"Invalid card index: {action.card_idx}"

    card = state.cards[action.card_idx]
    if action.card_idx not in state.ready:
        return False, f"Card {card.name} not affordable (idx={action.card_idx})"

    stats = CARD_STATS.get(card.name)
    if stats is None:
        return False, f"Unknown card: {card.name}"

    # Bounds
    if not (0 <= action.tile_x < N_TILE_X and 0 <= action.tile_y < N_TILE_Y):
        return False, f"Tile ({action.tile_x}, {action.tile_y}) out of bounds"

    # Placement zone (spells bypass)
    is_spell = stats.get("is_spell", False)
    if not is_spell and action.tile_y >= BRIDGE_Y:
        # Must be in a valid pocket
        in_left = action.tile_x < LANE_DIVIDER_X
        lane_ok = (in_left and enemy_left_princess_dead) or (
            not in_left and enemy_right_princess_dead
        )
        if not lane_ok:
            return False, "Cannot place troop on enemy side — tower still standing"

        pocket_min = int(RIVER_Y_MAX)
        pocket_max = int(RIVER_Y_MAX) + POCKET_DEPTH - 1
        if not (pocket_min <= action.tile_y <= pocket_max):
            return False, "Troop outside pocket depth"

    return True, None
