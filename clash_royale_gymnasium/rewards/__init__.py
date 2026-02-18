"""Reward system — composable, typed, strategy-aware."""

from clash_royale_gymnasium.rewards.base import RewardComponent, RewardFunction
from clash_royale_gymnasium.rewards.components import (
    DamageComponent,
    ElixirComponent,
    StrategyComponent,
    TerminalComponent,
)
from clash_royale_gymnasium.rewards.default import default_reward_function

__all__ = [
    "DamageComponent",
    "ElixirComponent",
    "RewardComponent",
    "RewardFunction",
    "StrategyComponent",
    "TerminalComponent",
    "default_reward_function",
]
