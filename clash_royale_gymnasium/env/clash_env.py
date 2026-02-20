"""Gymnasium environment for Clash Royale — dict observations, hierarchical actions.

Designed for RL training:
- **Observation**: partial (fog-of-war) ``Dict`` with entity list, scalars, cards.
- **Action**: hierarchical ``Dict`` (card → tile_x → tile_y) with masks.
- **Reward**: customisable via :class:`RewardFunction` callback.
- **Opponent**: any :class:`PlayerInterface` (heuristic bot, another agent, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.core.state import State
from clash_royale_engine.players.player_interface import (
    HeuristicBot,
    PlayerInterface,
    RLAgentPlayer,
)
from clash_royale_engine.utils.constants import (
    CARD_STATS,
    DEFAULT_DECK,
    DEFAULT_FPS,
    GAME_DURATION,
    MAX_ELIXIR,
)
from clash_royale_engine.utils.validators import InvalidActionError

from clash_royale_gymnasium.actions.masking import compute_action_mask, validate_hierarchical
from clash_royale_gymnasium.actions.spaces import build_action_space
from clash_royale_gymnasium.rewards.base import RewardFunction
from clash_royale_gymnasium.rewards.default import default_reward_function
from clash_royale_gymnasium.types.actions import (
    ActionMask,
    HierarchicalAction,
    N_DECK_SIZE,
    N_TILE_X,
    N_TILE_Y,
    NOOP_IDX,
)
from clash_royale_gymnasium.types.observations import (
    CARD_FEATURE_DIM,
    DECK_SIZE,
    MAX_TROOPS,
    SCALAR_DIM,
    TROOP_FEATURE_DIM,
)
from clash_royale_gymnasium.types.reward_context import RewardContext
from clash_royale_gymnasium.utils.encoder import encode_observation


class ClashRoyaleGymEnv(gym.Env):
    """Gymnasium environment for Clash Royale Arena 1.

    Action space (``Dict``)
    -----------------------
    ============ ============ ===============================================
    Key          Type         Description
    ============ ============ ===============================================
    ``card``     Discrete(9)  Deck index 0-7 or noop (8)
    ``tile_x``   Discrete(18) Tile column
    ``tile_y``   Discrete(32) Tile row
    ============ ============ ===============================================

    Observation space (``Dict``)
    ----------------------------
    ============== =================== ===================================
    Key            Shape               Description
    ============== =================== ===================================
    ``troops``     (100, 14) float32   Entity features (padded)
    ``troop_mask`` (100,)    bool      True where a real troop exists
    ``scalars``    (16,)     float32   Elixir, tower HP, time, flags
    ``cards``      (8, 5)    float32   Deck cards (all 8, with hand flag)
    ``action_mask``  Dict of bool arrays  Per-head validity masks
    ============== =================== ===================================

    All observations are **partial** — enemy elixir is hidden,
    only visible enemies are included, opponent hand is never exposed.

    Parameters
    ----------
    opponent : PlayerInterface, optional
        Opponent controller (default: ``HeuristicBot()``).
    reward_fn : RewardFunction, optional
        Custom reward callback.  Default: :func:`default_reward_function`.
    deck, opponent_deck : list[str], optional
        Card decks (default: 8 Arena-1 cards).
    fps : int
        Simulation framerate.
    time_limit : float
        Game duration in seconds.
    speed_multiplier : float
        Internal simulation speed-up factor.
    frame_skip : int
        Number of engine frames per ``step()`` call.  The RL action is
        applied on the **first** frame; subsequent frames advance with
        noop while the opponent keeps acting normally.  Rewards are
        **accumulated** across all skipped frames.  Default ``30``
        (≈1 decision/second at 30 fps, ~180 RL steps per match).
    speed_multiplier : float
        Number of physics ticks to run per frame-skip tick.  Compounds
        with ``frame_skip``: total ticks per RL step =
        ``frame_skip × speed_multiplier``.  Use ``1.0`` (default) for
        normal physics; ``2.0`` runs the engine at 2× game-time speed
        between decisions (troops move faster, elixir fills faster).
    seed : int
        Random seed.
    """

    metadata: Dict[str, Any] = {"render_modes": [None], "render_fps": 30}

    def __init__(
        self,
        opponent: Optional[PlayerInterface] = None,
        reward_fn: Optional[RewardFunction] = None,
        deck: Optional[List[str]] = None,
        opponent_deck: Optional[List[str]] = None,
        fps: int = DEFAULT_FPS,
        time_limit: float = GAME_DURATION,
        speed_multiplier: float = 1.0,
        frame_skip: int = 1,
        seed: int = 0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self._player_id = 0
        self._frame_skip = max(1, frame_skip)
        # How many raw physics ticks to run per frame-skip tick.
        # Compounds with frame_skip for total ticks per RL step.
        self._speed_ticks = max(1, int(speed_multiplier))

        # Reward
        self._reward_fn = reward_fn or default_reward_function()

        # Decks
        deck = deck or list(DEFAULT_DECK)
        opponent_deck = opponent_deck or list(DEFAULT_DECK)
        self._deck = deck
        self._mean_deck_cost = float(
            np.mean([CARD_STATS[c]["elixir"] for c in deck])
        )

        # Engine
        opponent = opponent or HeuristicBot()
        self.engine = ClashRoyaleEngine(
            player1=RLAgentPlayer(),
            player2=opponent,
            deck1=deck,
            deck2=opponent_deck,
            fps=fps,
            time_limit=time_limit,
            speed_multiplier=speed_multiplier,
            seed=seed,
        )

        # ── Spaces ────────────────────────────────────────────────────────
        self.action_space = build_action_space()

        self.observation_space = spaces.Dict(
            {
                "troops": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(MAX_TROOPS, TROOP_FEATURE_DIM), dtype=np.float32,
                ),
                "troop_mask": spaces.MultiBinary(MAX_TROOPS),
                "scalars": spaces.Box(
                    low=0.0, high=1.0, shape=(SCALAR_DIM,), dtype=np.float32,
                ),
                "cards": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(DECK_SIZE, CARD_FEATURE_DIM), dtype=np.float32,
                ),
                "action_mask": spaces.Dict(
                    {
                        "card": spaces.MultiBinary(N_DECK_SIZE + 1),
                        "tile_x_per_card": spaces.MultiBinary(
                            [N_DECK_SIZE + 1, N_TILE_X]
                        ),
                        "tile_y_per_card": spaces.MultiBinary(
                            [N_DECK_SIZE + 1, N_TILE_Y]
                        ),
                    }
                ),
            }
        )

        # ── Episode tracking ─────────────────────────────────────────────
        self._prev_towers_destroyed = 0
        self._prev_own_towers_alive = 3
        self._prev_leaked_elixir = 0.0
        self._total_reward = 0.0
        self._n_actions = 0
        self._step_count = 0

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        # Increment seed each episode for deck-shuffle variety
        if seed is not None:
            self.engine.seed = seed
        else:
            self._episode_seed = getattr(self, "_episode_seed", 0) + 1
            self.engine.seed = self._episode_seed

        self.engine.reset()
        self._reward_fn.reset()
        self._prev_towers_destroyed = 0
        self._prev_own_towers_alive = 3
        self._prev_leaked_elixir = 0.0
        self._total_reward = 0.0
        self._n_actions = 0
        self._step_count = 0

        state = self.engine.get_state(self._player_id)
        obs = self._build_obs(state)
        info = self._build_info(state, action_valid=True)
        return obs, info

    def step(
        self, action: Dict[str, int],
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._step_count += 1

        # ── Decode hierarchical action ────────────────────────────────────
        h_action = HierarchicalAction(
            card_idx=action["card"],
            tile_x=action["tile_x"],
            tile_y=action["tile_y"],
        )

        # ── Validate against masks (forbid invalid) ──────────────────────
        state_pre = self.engine.get_state(self._player_id)
        valid, reason = validate_hierarchical(
            h_action,
            state_pre,
            **self._enemy_tower_flags(),
        )

        action_valid = True
        state_pre = self.engine.get_state(self._player_id)
        deck = state_pre.deck if state_pre.deck else self._deck
        hand_names = [c.name for c in state_pre.cards]
        engine_action = h_action.to_engine_action(deck=deck, hand=hand_names)

        if not valid:
            # Masked invalid → treat as noop + flag
            engine_action = None
            action_valid = False

        # ── Step engine (with frame skipping + speed ticks) ─────────────
        #
        # Outer loop  (frame_skip iterations): one reward measurement each.
        #   sub_frame 0: apply the RL agent's card action on the FIRST physics tick.
        #   sub_frame 1+: RL agent does noop every tick.
        # Inner loop (speed_ticks per sub_frame): runs extra physics ticks
        #   between decisions so the simulation advances faster than real-time.
        #   The opponent's PlayerInterface get_action() is called every tick.
        # Rewards are accumulated at sub_frame granularity (outer loop).
        #
        cumulative_reward = 0.0
        cumulative_breakdown: Dict[str, float] = {}
        done = False

        for sub_frame in range(self._frame_skip):
            frame_action = engine_action if sub_frame == 0 else None

            # Inner physics-tick loop driven by speed_multiplier
            for tick in range(self._speed_ticks):
                tick_action = frame_action if tick == 0 else None
                try:
                    state_p0, _, done = self.engine.step_with_action(
                        player_id=self._player_id, action=tick_action,
                    )
                except InvalidActionError:
                    state_p0, _, done = self.engine.step_with_action(
                        player_id=self._player_id, action=None,
                    )
                    if sub_frame == 0 and tick == 0:
                        action_valid = False
                if done:
                    break

            # Compute reward for this sub-frame
            ctx = self._build_reward_context(
                state_p0, h_action, action_valid, done,
                is_action_frame=(sub_frame == 0),
            )
            sub_reward = self._reward_fn(ctx)
            sub_bd = self._reward_fn.breakdown(ctx)
            cumulative_reward += sub_reward
            for k, v in sub_bd.items():
                cumulative_breakdown[k] = cumulative_breakdown.get(k, 0.0) + v

            # Update trackers after each sub-frame so next sub-frame gets
            # correct deltas
            self._prev_towers_destroyed = self.engine.count_towers_destroyed(
                self._player_id,
            )
            self._prev_own_towers_alive = self._count_own_towers_alive()
            self._prev_leaked_elixir = self.engine.get_leaked_elixir(
                self._player_id,
            )

            if done:
                break

        if not h_action.is_noop and action_valid:
            self._n_actions += 1

        reward = cumulative_reward
        self._reward_breakdown = cumulative_breakdown
        self._total_reward += reward

        # ── Observation ───────────────────────────────────────────────────
        obs = self._build_obs(state_p0)

        terminated = done and self.engine.has_winner()
        truncated = done and not self.engine.has_winner()
        info = self._build_info(state_p0, action_valid=action_valid)

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        pass

    # ── Observation builder ───────────────────────────────────────────────

    def _build_obs(self, state: State) -> Dict[str, Any]:
        obs = encode_observation(state, self.engine, self._player_id)
        mask = compute_action_mask(state, **self._enemy_tower_flags())
        d = obs.to_dict()
        d["action_mask"] = mask.to_dict()
        return d

    # ── Reward context builder ────────────────────────────────────────────

    def _build_reward_context(
        self,
        state: State,
        action: HierarchicalAction,
        action_valid: bool,
        done: bool,
        *,
        is_action_frame: bool = True,
    ) -> RewardContext:
        pid = self._player_id
        damage_dealt = self.engine.get_tower_damage_per_tower(pid)
        damage_received = self.engine.get_tower_loss_per_tower(pid)

        # Tower destroy events (delta)
        current_destroyed = self.engine.count_towers_destroyed(pid)
        towers_destroyed_step = current_destroyed - self._prev_towers_destroyed

        current_own_alive = self._count_own_towers_alive()
        own_lost_step = self._prev_own_towers_alive - current_own_alive

        return RewardContext(
            damage_dealt=damage_dealt,
            damage_received=damage_received,
            own_tower_hp={
                "left_princess": state.numbers.left_princess_hp,
                "right_princess": state.numbers.right_princess_hp,
                "king": state.numbers.king_hp,
            },
            enemy_tower_hp={
                "left_princess": state.numbers.left_enemy_princess_hp,
                "right_princess": state.numbers.right_enemy_princess_hp,
                "king": state.numbers.enemy_king_hp,
            },
            current_elixir=state.numbers.elixir,
            troop_elixir_value=self.engine.get_alive_troop_elixir_value(pid),
            leaked_elixir=self.engine.get_leaked_elixir(pid),
            prev_leaked_elixir=self._prev_leaked_elixir,
            mean_deck_cost=self._mean_deck_cost,
            game_done=done,
            winner=self.engine.get_winner(),
            player_id=pid,
            action=action,
            action_valid=action_valid,
            is_action_frame=is_action_frame,
            towers_destroyed_this_step=max(0, towers_destroyed_step),
            own_towers_lost_this_step=max(0, own_lost_step),
            prev_towers_destroyed=self._prev_towers_destroyed,
            prev_own_towers_alive=self._prev_own_towers_alive,
        )

    # ── Info dict (stats visible to logger / league) ──────────────────────

    def _build_info(self, state: State, *, action_valid: bool) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "action_valid": action_valid,
            "elixir": state.numbers.elixir,
            "towers_destroyed": self.engine.count_towers_destroyed(self._player_id),
            "own_towers_alive": self._count_own_towers_alive(),
            "total_reward": self._total_reward,
            "n_actions": self._n_actions,
            "step_count": self._step_count,
            "time_remaining": state.numbers.time_remaining,
            "reward_breakdown": getattr(self, "_reward_breakdown", {}),
        }
        # Engine-level debug signals (if method exists in engine version)
        if hasattr(self.engine, "debug_reward_signals"):
            info["engine_debug"] = self.engine.debug_reward_signals(self._player_id)
        return info

    # ── Helpers ───────────────────────────────────────────────────────────

    def _enemy_tower_flags(self) -> Dict[str, bool]:
        opp = 1 - self._player_id
        return {
            "enemy_left_princess_dead": self.engine.arena.tower_hp(opp, "left_princess") <= 0,
            "enemy_right_princess_dead": self.engine.arena.tower_hp(opp, "right_princess") <= 0,
        }

    def _count_own_towers_alive(self) -> int:
        pid = self._player_id
        alive = 0
        for t in ("left_princess", "right_princess", "king"):
            if self.engine.arena.tower_hp(pid, t) > 0:
                alive += 1
        return alive

    # ── Episode statistics (for league / reporting) ───────────────────────

    @property
    def episode_stats(self) -> Dict[str, Any]:
        """Summary statistics for the completed episode."""
        return {
            "total_reward": self._total_reward,
            "n_actions": self._n_actions,
            "steps": self._step_count,
            "game_duration": self._step_count / max(self.engine.fps, 1),
            "winner": self.engine.get_winner(),
            "towers_destroyed": self.engine.count_towers_destroyed(self._player_id),
            "own_towers_alive": self._count_own_towers_alive(),
            "leaked_elixir": self.engine.get_leaked_elixir(self._player_id),
        }
