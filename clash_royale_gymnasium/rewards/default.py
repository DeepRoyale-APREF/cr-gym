"""Default reward function with balanced weights."""

from __future__ import annotations

from clash_royale_gymnasium.rewards.base import RewardFunction
from clash_royale_gymnasium.rewards.components import (
    DamageComponent,
    DefensiveComponent,
    ElixirComponent,
    TerminalComponent,
)


def default_reward_function(
    damage_weight=5.0,
    defensive_weight=0.5,
    elixir_weight=0.2,
    terminal_weight=1,
    princess_reward=5.0,
    win_reward=10.0,
    comfort_threshold=6.0,
) -> RewardFunction:
    """Create the default reward function with tuneable weights.

    Parameters
    ----------
    damage_weight : float
        Weight for tower-damage component.
    defensive_weight : float
        Weight for killing enemy troops (defensive play).
    elixir_weight : float
        Weight for elixir-patience signal (penalise spending at low elixir).
    terminal_weight : float
        Weight for princess-destroy and win/loss rewards.
    princess_reward : float
        Reward per princess tower destroyed.
    win_reward : float
        Reward for winning the match.
    comfort_threshold : float
        Elixir level at which spending is neutral (default 7.0).
    """
    return RewardFunction(
        [
            DamageComponent(weight=damage_weight),
            DefensiveComponent(weight=defensive_weight),
            ElixirComponent(
                weight=elixir_weight,
                comfort_threshold=comfort_threshold,
            ),
            TerminalComponent(
                weight=terminal_weight,
                princess_reward=princess_reward,
                win_reward=win_reward,
            ),
        ]
    )
