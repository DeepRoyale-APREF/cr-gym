"""Built-in reward components for Clash Royale training."""

from __future__ import annotations

import math

from clash_royale_gymnasium.rewards.base import RewardComponent
from clash_royale_gymnasium.types.actions import Strategy
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
    """Penalise poor elixir management.

    Two sub-signals:

    1. **Elixir advantage deficit** (continuous):
       ``-(mean_deck_cost - (current_elixir + troop_value)) / MAX_ELIXIR``
       Negative when you have less elixir invested than your average card cost.

    2. **Leaked elixir** (continuous):
       ``-sigmoid_norm(leaked_delta)``  where ``sigmoid_norm`` maps [0, ∞) → [0, 1)
       via ``2/(1+exp(-k*x)) - 1`` so small leaks are cheap but large ones hurt.
    """

    def __init__(self, weight: float = 1.0, leak_sensitivity: float = 0.5) -> None:
        super().__init__(weight)
        self._leak_k = leak_sensitivity

    def compute(self, ctx: RewardContext) -> float:
        max_elixir = 10.0

        # 1. Elixir deficit
        invested = ctx.current_elixir + ctx.troop_elixir_value
        deficit = (ctx.mean_deck_cost - invested) / max_elixir
        deficit = max(0.0, deficit)  # only penalise deficit, not surplus

        # 2. Leaked elixir (this frame's increment)
        leaked_delta = ctx.leaked_elixir - ctx.prev_leaked_elixir
        leak_penalty = self._sigmoid_norm(leaked_delta)

        return -(deficit + leak_penalty)

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


class StrategyComponent(RewardComponent):
    """Reward that biases components by the agent's declared strategy.

    The hierarchical policy first picks a :class:`Strategy`.  This component
    adjusts rewards so each strategy is incentivised correctly:

    - **AGGRESSIVE**: amplify ``DamageComponent`` dealt factor.
    - **DEFENSIVE**: amplify penalty for damage received.
    - **FARMING**: amplify elixir efficiency.

    Parameters
    ----------
    aggressive_bonus, defensive_bonus, farming_bonus : float
        Additive weights applied to the relevant sub-signal when the
        matching strategy is active.
    """

    def __init__(
        self,
        weight: float = 1.0,
        aggressive_bonus: float = 0.3,
        defensive_bonus: float = 0.3,
        farming_bonus: float = 0.3,
    ) -> None:
        super().__init__(weight)
        self.aggressive_bonus = aggressive_bonus
        self.defensive_bonus = defensive_bonus
        self.farming_bonus = farming_bonus

    def compute(self, ctx: RewardContext) -> float:
        strategy = ctx.strategy
        reward = 0.0

        if strategy == Strategy.AGGRESSIVE:
            # Bonus for damage dealt
            for tower, max_hp in _TOWER_MAX.items():
                reward += ctx.damage_dealt.get(tower, 0.0) / max_hp * self.aggressive_bonus

        elif strategy == Strategy.DEFENSIVE:
            # Reduced penalty for damage received (relative to baseline)
            for tower, max_hp in _TOWER_MAX.items():
                dmg_recv = ctx.damage_received.get(tower, 0.0) / max_hp
                # If defending well (low dmg received), get positive reward
                reward += (0.01 - dmg_recv) * self.defensive_bonus

        elif strategy == Strategy.FARMING:
            # Reward for having high elixir + troop investment
            invested = (ctx.current_elixir + ctx.troop_elixir_value) / 10.0
            leaked_norm = min(ctx.leaked_elixir / 20.0, 1.0)
            reward += (invested - leaked_norm) * self.farming_bonus

        return reward
