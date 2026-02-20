"""Default reward function with balanced weights."""

from __future__ import annotations

from clash_royale_gymnasium.rewards.base import RewardFunction
from clash_royale_gymnasium.rewards.components import (
    DamageComponent,
    ElixirComponent,
    TerminalComponent,
)


def default_reward_function(
    damage_weight=5.0,
    elixir_weight=0.2,
    terminal_weight=0.5,
    princess_reward=5.0,
    win_reward=10.0,
    leak_sensitivity=0.5,
) -> RewardFunction:
    """Create the default reward function with tuneable weights.

    Parameters
    ----------
    damage_weight : float
        Weight for tower-damage component.
    elixir_weight : float
        Weight for elixir-efficiency component.
    terminal_weight : float
        Weight for princess-destroy and win/loss rewards.
    princess_reward : float
        Reward per princess tower destroyed.
    win_reward : float
        Reward for winning the match.
    leak_sensitivity : float
        How aggressively leaked elixir is penalised (0.1=gentle, 2.0=harsh).
    """
    return RewardFunction(
        [
            DamageComponent(weight=damage_weight),
            ElixirComponent(
                weight=elixir_weight,
                leak_sensitivity=leak_sensitivity,
            ),
            TerminalComponent(
                weight=terminal_weight,
                princess_reward=princess_reward,
                win_reward=win_reward,
            ),
        ]
    )
