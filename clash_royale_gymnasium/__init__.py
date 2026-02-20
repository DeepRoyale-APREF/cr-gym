"""Clash Royale Gymnasium — AlphaStar-style training environment.

Wraps ``clash-royale-engine`` with:
- **Dict observations** (partial / fog-of-war) with entity lists for transformer encoders.
- **Hierarchical action space** with per-head action masking.
- **Composable reward system** with typed callbacks.
- **League tournament** system for round-robin evaluation.
- **PDF reporting** for performance analysis.
"""

from clash_royale_gymnasium.env.clash_env import ClashRoyaleGymEnv
from clash_royale_gymnasium.league.player_slot import (
    ExternalAgentSlot,
    HeuristicSlot,
    PlayerSlot,
)
from clash_royale_gymnasium.league.tournament import LeagueTournament
from clash_royale_gymnasium.reporting.tracker import TrainingTracker
from clash_royale_gymnasium.rewards.base import RewardComponent, RewardFunction
from clash_royale_gymnasium.rewards.default import default_reward_function
from clash_royale_gymnasium.types.actions import ActionMask, HierarchicalAction
from clash_royale_gymnasium.types.observations import Observation
from clash_royale_gymnasium.types.reward_context import RewardContext

__version__ = "0.1.0"

__all__ = [
    # Environment
    "ClashRoyaleGymEnv",
    # Reward
    "RewardComponent",
    "RewardContext",
    "RewardFunction",
    "default_reward_function",
    # Actions
    "ActionMask",
    "HierarchicalAction",
    # Observation
    "Observation",
    # League
    "ExternalAgentSlot",
    "HeuristicSlot",
    "LeagueTournament",
    "PlayerSlot",
    # Reporting
    "TrainingTracker",
]
