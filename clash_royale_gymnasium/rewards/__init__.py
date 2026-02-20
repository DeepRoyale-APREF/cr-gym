"""Reward system — composable and typed."""

from clash_royale_gymnasium.rewards.base import RewardComponent, RewardFunction
from clash_royale_gymnasium.rewards.components import (
    DamageComponent,
    ElixirComponent,
    TerminalComponent,
)
from clash_royale_gymnasium.rewards.default import default_reward_function

__all__ = [
    "DamageComponent",
    "ElixirComponent",
    "RewardComponent",
    "RewardFunction",
    "TerminalComponent",
    "default_reward_function",
]
