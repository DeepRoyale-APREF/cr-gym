"""Default reward function with balanced weights."""

from __future__ import annotations

from clash_royale_gymnasium.rewards.base import RewardFunction
from clash_royale_gymnasium.rewards.components import (
    DamageComponent,
    ElixirComponent,
    StrategyComponent,
    TerminalComponent,
)


def default_reward_function(
    damage_weight: float = 1.0,
    elixir_weight: float = 0.3,
    terminal_weight: float = 1.0,
    strategy_weight: float = 0.5,
    princess_reward: float = 10.0,
    win_reward: float = 20.0,
    leak_sensitivity: float = 0.5,
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
    strategy_weight : float
        Weight for strategy-conditioned bonus.
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
            ElixirComponent(weight=elixir_weight, leak_sensitivity=leak_sensitivity),
            TerminalComponent(
                weight=terminal_weight,
                princess_reward=princess_reward,
                win_reward=win_reward,
            ),
            StrategyComponent(weight=strategy_weight),
        ]
    )
