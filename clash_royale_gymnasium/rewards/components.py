"""Built-in reward components for Clash Royale training."""

from __future__ import annotations

import math

from clash_royale_gymnasium.rewards.base import RewardComponent
from clash_royale_gymnasium.types.reward_context import RewardContext

# ── Tower max HP for normalisation ────────────────────────────────────────────
_PRINCESS_MAX_HP = 1400.0
_KING_MAX_HP = 2400.0

_TOWER_MAX: dict[str, float] = {
    "left_princess": _PRINCESS_MAX_HP,
    "right_princess": _PRINCESS_MAX_HP,
    "king": _KING_MAX_HP,
}


class DamageComponent(RewardComponent):
    """Reward for tower damage dealt minus tower damage received.

    For each tower:
        ``+ diff_hp_dealt / max_hp_of_that_tower``
        ``- diff_hp_received / max_hp_of_that_tower``

    Summed across all 3 towers on each side.
    """

    def compute(self, ctx: RewardContext) -> float:
        reward = 0.0
        for tower, max_hp in _TOWER_MAX.items():
            reward += ctx.damage_dealt.get(tower, 0.0) / max_hp
            reward -= ctx.damage_received.get(tower, 0.0) / max_hp
        return reward


class ElixirComponent(RewardComponent):
    """Elixir management reward with two signals.

    1. **Leaked elixir penalty** (every frame, when ``leaked_delta > 0``):
       Penalises wasted elixir at the 10-cap.  Uses a sigmoid mapping
       so small leaks cost little but large ones hurt.
       ``-sigmoid_norm(leaked_delta)``

    2. **Elixir efficiency signal** (only on the action-frame of a card play):
       Measures how well the agent is investing elixir into the field
       relative to its deck's average cost.

       ``(mean_deck_cost - (current_elixir + troop_elixir_value)) / 10``

       - Positive  → agent has room to invest more (low field value / high elixir).
       - Negative  → agent has over-committed / lots of troops already deployed.

    Parameters
    ----------
    leak_sensitivity : float
        Sigmoid steepness for leaked-elixir penalty (0.1 = gentle, 2.0 = harsh).
    """

    def __init__(
        self,
        weight: float = 1.0,
        leak_sensitivity: float = 0.5,
    ) -> None:
        super().__init__(weight)
        self._leak_k = leak_sensitivity

    def compute(self, ctx: RewardContext) -> float:
        reward = 0.0

        # 1. Leaked elixir penalty — fires every frame when there is new leakage
        leaked_delta = ctx.leaked_elixir - ctx.prev_leaked_elixir
        if leaked_delta > 0:
            reward -= self._sigmoid_norm(leaked_delta)

        # 2. Elixir efficiency — fires only when the agent plays a valid card
        if (
            ctx.is_action_frame
            and ctx.action is not None
            and not ctx.action.is_noop
            and ctx.action_valid
        ):
            reward += (
                ctx.mean_deck_cost - (ctx.current_elixir + ctx.troop_elixir_value)
            ) / 10.0

        return reward

    def _sigmoid_norm(self, x: float) -> float:
        """Map [0, ∞) → [0, 1) with tuneable sensitivity."""
        if x <= 0:
            return 0.0
        return 2.0 / (1.0 + math.exp(-self._leak_k * x)) - 1.0


class TerminalComponent(RewardComponent):
    """Large reward/penalty at game end.

    +/- ``princess_reward`` per princess tower destroyed / lost.
    +/- ``win_reward`` for winning / losing the match.
    """

    def __init__(
        self,
        weight: float = 1.0,
        princess_reward: float = 10.0,
        win_reward: float = 20.0,
    ) -> None:
        super().__init__(weight)
        self.princess_reward = princess_reward
        self.win_reward = win_reward

    def compute(self, ctx: RewardContext) -> float:
        reward = 0.0

        # Princess tower destroy/loss events (detected via delta)
        reward += ctx.towers_destroyed_this_step * self.princess_reward
        reward -= ctx.own_towers_lost_this_step * self.princess_reward

        # Win/loss
        if ctx.game_done:
            if ctx.winner == ctx.player_id:
                reward += self.win_reward
            elif ctx.winner is not None:
                reward -= self.win_reward
            # draw: 0

        return reward

