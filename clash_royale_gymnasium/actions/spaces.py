"""Gymnasium action space definition for hierarchical actions."""

from __future__ import annotations

from gymnasium import spaces

from clash_royale_gymnasium.types.actions import N_DECK_SIZE, N_TILE_X, N_TILE_Y


def build_action_space() -> spaces.Dict:
    """Return the hierarchical ``Dict`` action space.

    Keys
    ----
    card : Discrete(9)
        0-7 = deck card, 8 = noop.
    tile_x : Discrete(18)
    tile_y : Discrete(32)
    """
    return spaces.Dict(
        {
            "card": spaces.Discrete(N_DECK_SIZE + 1),  # +1 for noop
            "tile_x": spaces.Discrete(N_TILE_X),
            "tile_y": spaces.Discrete(N_TILE_Y),
        }
    )
