"""Context dataclass passed to reward callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from clash_royale_gymnasium.types.actions import HierarchicalAction


@dataclass(slots=True)
class RewardContext:
    """All information a reward function may need.

    Built by the environment each step and passed to the
    :class:`~clash_royale_gymnasium.rewards.base.RewardFunction` callback.

    Note: reward functions have access to **privileged** engine data (e.g.
    exact tower HP deltas) for shaping.  The *observation* the agent sees
    remains partial (no enemy elixir, no hidden troops, no opponent hand).
    """

    # ── Tower damage (per-tower, this frame) ──────────────────────────────
    damage_dealt: dict[str, float] = field(default_factory=dict)
    """HP lost by **opponent** towers: ``{left_princess, right_princess, king}``."""

    damage_received: dict[str, float] = field(default_factory=dict)
    """HP lost by **own** towers."""

    # ── Tower HP (absolute, for destroy detection) ────────────────────────
    own_tower_hp: dict[str, float] = field(default_factory=dict)
    enemy_tower_hp: dict[str, float] = field(default_factory=dict)

    # ── Elixir ────────────────────────────────────────────────────────────
    current_elixir: float = 0.0
    troop_elixir_value: float = 0.0
    leaked_elixir: float = 0.0
    prev_leaked_elixir: float = 0.0
    mean_deck_cost: float = 0.0

    # ── Game state ────────────────────────────────────────────────────────
    game_done: bool = False
    winner: Optional[int] = None  # 0, 1, None (draw)
    player_id: int = 0

    # ── Action ────────────────────────────────────────────────────────────
    action: Optional[HierarchicalAction] = None
    action_valid: bool = True

    # ── Frame info ────────────────────────────────────────────────────────
    is_action_frame: bool = True
    """True on the sub-frame where the agent's action is applied (sub_frame 0).
    False on subsequent frame-skip sub-frames.  Useful for event-driven
    reward components that should only fire once per agent decision."""

    # ── Opponent towers destroyed this step ───────────────────────────────
    towers_destroyed_this_step: int = 0
    own_towers_lost_this_step: int = 0

    # ── Previous frame counts (for delta) ─────────────────────────────────
    prev_towers_destroyed: int = 0
    prev_own_towers_alive: int = 3
