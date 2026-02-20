"""Gymnasium action space definition for hierarchical actions."""

from __future__ import annotations

from gymnasium import spaces

from clash_royale_gymnasium.types.actions import N_DECK_SIZE, N_STRATEGIES, N_TILE_X, N_TILE_Y


def build_action_space() -> spaces.Dict:
    """Return the hierarchical ``Dict`` action space.

    Keys
    ----
    strategy : Discrete(3)
        AGGRESSIVE (0), DEFENSIVE (1), FARMING (2).
    card : Discrete(9)
        0-7 = deck card, 8 = noop.
    tile_x : Discrete(18)
    tile_y : Discrete(32)
    """
    return spaces.Dict(
        {
            "strategy": spaces.Discrete(N_STRATEGIES),
            "card": spaces.Discrete(N_DECK_SIZE + 1),  # +1 for noop
            "tile_x": spaces.Discrete(N_TILE_X),
            "tile_y": spaces.Discrete(N_TILE_Y),
        }
    )
