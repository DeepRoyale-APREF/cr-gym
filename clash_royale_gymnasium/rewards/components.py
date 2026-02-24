"""Built-in reward components for Clash Royale training."""

from __future__ import annotations

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
    """Elixir management reward — penalises impatient spending.

    Fires once per action frame when a card is played (not noop).
    The reward depends on how much elixir the agent had *before*
    playing the card, relative to a comfortable threshold:

    - Playing at high elixir (≥ ``comfort_threshold``) → small positive
      reward: the agent is spending surplus wisely.
    - Playing at low elixir (just barely enough) → penalty: the agent
      is spending the instant it can afford anything.

    Formula::

        signal = (current_elixir - comfort_threshold) / 10.0

    At ``comfort_threshold=7``:

    ========  =======  ===========
    Elixir    Signal   Meaning
    ========  =======  ===========
     10        +0.30   great — spending surplus
      8        +0.10   fine — comfortable
      7         0.00   neutral
      5        −0.20   bad — spending while poor
      3        −0.40   very bad — desperate play
    ========  =======  ===========

    This teaches the agent to **save up** before deploying, which
    naturally leads to building proper pushes (Giant + support) rather
    than instantly dropping the cheapest troop available.  It does NOT
    penalise expensive cards — a Giant at 8 elixir is rewarded the
    same as skeletons at 8 elixir.

    Parameters
    ----------
    comfort_threshold : float
        Elixir level at which spending is neutral (default 7.0).
    """

    def __init__(
        self,
        weight: float = 1.0,
        comfort_threshold: float = 6.0,
    ) -> None:
        super().__init__(weight)
        self._threshold = comfort_threshold

    def compute(self, ctx: RewardContext) -> float:
        # Only fires on the action frame when a card was actually played
        if not ctx.is_action_frame or ctx.played_card_cost <= 0:
            return 0.0
        return (ctx.current_elixir - self._threshold) / 10.0


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


class DefensiveComponent(RewardComponent):
    """Reward for eliminating enemy troops (defensive play).

    Tracks the delta in opponent's alive troop elixir value between
    frames.  When enemy troops die (value drops), the agent receives
    a positive reward proportional to the elixir value destroyed.
    This incentivises using spells and troops to defend against pushes.

    ``reward = max(0, prev_enemy_value - current_enemy_value) / normaliser``

    The signal fires whether the agent killed the troops directly
    (spells, defensive troop combat) or indirectly (tower fire), but
    combined with the damage component it teaches the agent to
    proactively defend rather than passively let towers take hits.
    """

    def __init__(self, weight: float = 1.0, normaliser: float = 20.0) -> None:
        super().__init__(weight)
        self._normaliser = normaliser

    def compute(self, ctx: RewardContext) -> float:
        killed_value = ctx.prev_enemy_troop_elixir_value - ctx.enemy_troop_elixir_value
        if killed_value > 0:
            return killed_value / self._normaliser
        return 0.0
