"""Tests for the Clash Royale Gymnasium package."""

from __future__ import annotations

import numpy as np
import pytest

from clash_royale_engine.core.state import State
from clash_royale_engine.players.player_interface import HeuristicBot

from clash_royale_gymnasium.actions.masking import compute_action_mask, validate_hierarchical
from clash_royale_gymnasium.actions.spaces import build_action_space
from clash_royale_gymnasium.env.clash_env import ClashRoyaleGymEnv
from clash_royale_gymnasium.league.match import run_match
from clash_royale_gymnasium.league.player_slot import HeuristicSlot
from clash_royale_gymnasium.league.tournament import LeagueTournament
from clash_royale_gymnasium.rewards.base import RewardComponent, RewardFunction
from clash_royale_gymnasium.rewards.components import (
    DamageComponent,
    ElixirComponent,
    TerminalComponent,
)
from clash_royale_gymnasium.rewards.default import default_reward_function
from clash_royale_gymnasium.types.actions import (
    ActionMask,
    HierarchicalAction,
    N_CARDS,
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
    CardInfo,
    Observation,
    ScalarFeatures,
    TroopInfo,
)
from clash_royale_gymnasium.types.reward_context import RewardContext


# ══════════════════════════════════════════════════════════════════════════════
# Observation types
# ══════════════════════════════════════════════════════════════════════════════


class TestObservationTypes:
    """Test the typed observation dataclasses."""

    def test_scalar_dim_excludes_enemy_elixir(self) -> None:
        """SCALAR_DIM should be 16 (no enemy elixir in partial obs)."""
        assert SCALAR_DIM == 16

    def test_scalar_features_array_shape(self) -> None:
        sf = ScalarFeatures(
            elixir=5.0,
            left_princess_hp=1400.0,
            right_princess_hp=1400.0,
            king_hp=2400.0,
            left_enemy_princess_hp=1400.0,
            right_enemy_princess_hp=1400.0,
            enemy_king_hp=2400.0,
            time_remaining=180.0,
            king_active=False,
            enemy_king_active=False,
            is_double_elixir=False,
            is_overtime=False,
            overtime_remaining=0.0,
            troop_elixir_value=0.0,
            leaked_elixir=0.0,
            frame_ratio=0.0,
        )
        arr = sf.to_array()
        assert arr.shape == (SCALAR_DIM,)
        assert arr.dtype == np.float32
        # First element is elixir/10 = 0.5
        assert abs(arr[0] - 0.5) < 1e-5

    def test_scalar_no_enemy_elixir_field(self) -> None:
        """ScalarFeatures should NOT have an enemy_elixir attribute."""
        assert not hasattr(ScalarFeatures, "enemy_elixir") or "enemy_elixir" not in [
            f.name for f in ScalarFeatures.__dataclass_fields__.values()  # type: ignore[attr-defined]
        ]

    def test_troop_info_array_shape(self) -> None:
        ti = TroopInfo(
            name_idx=0, category=0, target=0, transport=0,
            tile_x=9.0, tile_y=5.0, hp_ratio=1.0, is_ally=True,
            cost=0.5, damage=0.3, hit_speed=0.5, attack_range=0.5,
            speed=0.6, hitbox_radius=0.25,
        )
        arr = ti.to_array()
        assert arr.shape == (TROOP_FEATURE_DIM,)

    def test_card_info_array_shape(self) -> None:
        ci = CardInfo(name_idx=0, cost=5, is_spell=False, is_in_hand=True, is_affordable=True)
        arr = ci.to_array()
        assert arr.shape == (CARD_FEATURE_DIM,)

    def test_observation_to_dict(self) -> None:
        obs = Observation(
            troops=np.zeros((MAX_TROOPS, TROOP_FEATURE_DIM), dtype=np.float32),
            troop_mask=np.zeros(MAX_TROOPS, dtype=bool),
            scalars=np.zeros(SCALAR_DIM, dtype=np.float32),
            cards=np.zeros((DECK_SIZE, CARD_FEATURE_DIM), dtype=np.float32),
        )
        d = obs.to_dict()
        assert "troops" in d
        assert "troop_mask" in d
        assert "scalars" in d
        assert "cards" in d
        assert d["troops"].shape == (MAX_TROOPS, TROOP_FEATURE_DIM)


# ══════════════════════════════════════════════════════════════════════════════
# Action types and masking
# ══════════════════════════════════════════════════════════════════════════════


class TestActionTypes:
    """Test hierarchical action types."""

    def test_noop_action(self) -> None:
        a = HierarchicalAction(
            card_idx=NOOP_IDX, tile_x=0, tile_y=0,
        )
        assert a.is_noop
        assert a.to_engine_action() is None

    def test_play_action(self) -> None:
        a = HierarchicalAction(
            card_idx=2, tile_x=9, tile_y=5,
        )
        assert not a.is_noop
        assert a.to_engine_action() == (9, 5, 2)

    def test_action_space_structure(self) -> None:
        space = build_action_space()
        assert "card" in space.spaces
        assert "tile_x" in space.spaces
        assert "tile_y" in space.spaces
        assert space["card"].n == N_DECK_SIZE + 1  # +1 noop
        assert space["tile_x"].n == N_TILE_X
        assert space["tile_y"].n == N_TILE_Y


class TestActionMaskShape:
    """Test mask shapes and noop guarantee."""

    def test_mask_to_dict_keys(self) -> None:
        n_options = N_DECK_SIZE + 1
        mask = ActionMask(
            card=np.ones(n_options, dtype=bool),
            tile_x_per_card=np.ones((n_options, N_TILE_X), dtype=bool),
            tile_y_per_card=np.ones((n_options, N_TILE_Y), dtype=bool),
        )
        d = mask.to_dict()
        assert d["card"].shape == (n_options,)
        assert d["tile_x_per_card"].shape == (n_options, N_TILE_X)
        assert d["tile_y_per_card"].shape == (n_options, N_TILE_Y)


# ══════════════════════════════════════════════════════════════════════════════
# Reward system
# ══════════════════════════════════════════════════════════════════════════════


class TestRewardComponents:
    """Test individual reward components."""

    def _base_ctx(self) -> RewardContext:
        return RewardContext(
            damage_dealt={"left_princess": 0.0, "right_princess": 0.0, "king": 0.0},
            damage_received={"left_princess": 0.0, "right_princess": 0.0, "king": 0.0},
            own_tower_hp={"left_princess": 1400, "right_princess": 1400, "king": 2400},
            enemy_tower_hp={"left_princess": 1400, "right_princess": 1400, "king": 2400},
            current_elixir=5.0,
            troop_elixir_value=0.0,
            leaked_elixir=0.0,
            prev_leaked_elixir=0.0,
            mean_deck_cost=3.375,
            game_done=False,
            winner=None,
            player_id=0,
        )

    def test_damage_component_zero_when_no_damage(self) -> None:
        comp = DamageComponent(weight=1.0)
        ctx = self._base_ctx()
        assert comp.compute(ctx) == 0.0

    def test_damage_component_positive_for_dealt(self) -> None:
        comp = DamageComponent(weight=1.0)
        ctx = self._base_ctx()
        ctx.damage_dealt["left_princess"] = 100.0
        reward = comp.compute(ctx)
        assert reward > 0

    def test_damage_component_negative_for_received(self) -> None:
        comp = DamageComponent(weight=1.0)
        ctx = self._base_ctx()
        ctx.damage_received["left_princess"] = 100.0
        reward = comp.compute(ctx)
        assert reward < 0

    def test_terminal_win_reward(self) -> None:
        comp = TerminalComponent(weight=1.0, win_reward=20.0)
        ctx = self._base_ctx()
        ctx.game_done = True
        ctx.winner = 0
        reward = comp.compute(ctx)
        assert reward == 20.0

    def test_terminal_loss_penalty(self) -> None:
        comp = TerminalComponent(weight=1.0, win_reward=20.0)
        ctx = self._base_ctx()
        ctx.game_done = True
        ctx.winner = 1
        reward = comp.compute(ctx)
        assert reward == -20.0

    def test_terminal_draw_zero(self) -> None:
        comp = TerminalComponent(weight=1.0, win_reward=20.0)
        ctx = self._base_ctx()
        ctx.game_done = True
        ctx.winner = None
        reward = comp.compute(ctx)
        assert reward == 0.0

    def test_terminal_tower_destroy_reward(self) -> None:
        comp = TerminalComponent(weight=1.0, princess_reward=10.0)
        ctx = self._base_ctx()
        ctx.towers_destroyed_this_step = 1
        reward = comp.compute(ctx)
        assert reward == 10.0

    def test_elixir_component_no_penalty_at_start(self) -> None:
        comp = ElixirComponent(weight=1.0)
        ctx = self._base_ctx()
        ctx.current_elixir = 5.0
        ctx.troop_elixir_value = 0.0
        reward = comp.compute(ctx)
        # small deficit penalty only, not huge
        assert reward <= 0

    def test_elixir_leaked_penalty(self) -> None:
        comp = ElixirComponent(weight=1.0, leak_sensitivity=1.0)
        ctx = self._base_ctx()
        ctx.leaked_elixir = 5.0
        ctx.prev_leaked_elixir = 0.0
        reward = comp.compute(ctx)
        assert reward < -0.1  # should be penalised

    def test_composite_reward_function(self) -> None:
        rf = default_reward_function()
        ctx = self._base_ctx()
        reward = rf(ctx)
        assert isinstance(reward, float)

    def test_reward_breakdown(self) -> None:
        rf = default_reward_function()
        ctx = self._base_ctx()
        bd = rf.breakdown(ctx)
        assert "DamageComponent" in bd
        assert "ElixirComponent" in bd
        assert "TerminalComponent" in bd


class TestCustomReward:
    """Test that custom reward components can be created."""

    def test_custom_component(self) -> None:
        class MyReward(RewardComponent):
            def compute(self, ctx: RewardContext) -> float:
                return 42.0

        rf = RewardFunction([MyReward(weight=2.0)])
        ctx = RewardContext()  # defaults
        assert rf(ctx) == 84.0


# ══════════════════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════════════════


class TestEnvironment:
    """Test the Gymnasium environment."""

    @pytest.fixture()
    def env(self) -> ClashRoyaleGymEnv:
        return ClashRoyaleGymEnv(
            opponent=HeuristicBot(aggression=0.5, seed=42),
            fps=30,
            time_limit=10.0,
            speed_multiplier=1.0,
            seed=0,
        )

    def test_reset_returns_dict_obs(self, env: ClashRoyaleGymEnv) -> None:
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "troops" in obs
        assert "troop_mask" in obs
        assert "scalars" in obs
        assert "cards" in obs
        assert "action_mask" in obs

    def test_obs_shapes(self, env: ClashRoyaleGymEnv) -> None:
        obs, _ = env.reset()
        assert obs["troops"].shape == (MAX_TROOPS, TROOP_FEATURE_DIM)
        assert obs["troop_mask"].shape == (MAX_TROOPS,)
        assert obs["scalars"].shape == (SCALAR_DIM,)
        assert obs["cards"].shape == (DECK_SIZE, CARD_FEATURE_DIM)

    def test_obs_partial_no_enemy_elixir(self, env: ClashRoyaleGymEnv) -> None:
        """Scalars should NOT contain enemy elixir (fog-of-war)."""
        obs, _ = env.reset()
        # SCALAR_DIM is 16 (not 18 which would include enemy_elixir)
        assert obs["scalars"].shape[0] == 16

    def test_action_mask_in_obs(self, env: ClashRoyaleGymEnv) -> None:
        obs, _ = env.reset()
        am = obs["action_mask"]
        n_options = N_DECK_SIZE + 1
        assert am["card"].shape == (n_options,)
        assert am["tile_x_per_card"].shape == (n_options, N_TILE_X)
        assert am["tile_y_per_card"].shape == (n_options, N_TILE_Y)

    def test_noop_always_valid(self, env: ClashRoyaleGymEnv) -> None:
        obs, _ = env.reset()
        assert obs["action_mask"]["card"][NOOP_IDX]  # noop always True

    def test_step_with_noop(self, env: ClashRoyaleGymEnv) -> None:
        env.reset()
        action = {"card": NOOP_IDX, "tile_x": 0, "tile_y": 0}
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(info, dict)
        assert "action_valid" in info

    def test_step_with_valid_play(self, env: ClashRoyaleGymEnv) -> None:
        obs, _ = env.reset()
        card_mask = obs["action_mask"]["card"]

        # Find a playable card
        playable = [i for i in range(N_DECK_SIZE) if card_mask[i]]
        if playable:
            card_idx = playable[0]
            action = {"card": card_idx, "tile_x": 9, "tile_y": 5}
            obs2, reward, terminated, truncated, info = env.step(action)
            assert isinstance(obs2, dict)

    def test_episode_completes(self, env: ClashRoyaleGymEnv) -> None:
        env.reset()
        done = False
        steps = 0
        while not done and steps < 20000:
            action = {"card": NOOP_IDX, "tile_x": 0, "tile_y": 0}
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert done  # should finish within time limit

    def test_episode_stats(self, env: ClashRoyaleGymEnv) -> None:
        env.reset()
        # Run a few steps
        for _ in range(10):
            env.step({"card": NOOP_IDX, "tile_x": 0, "tile_y": 0})
        stats = env.episode_stats
        assert "total_reward" in stats
        assert "n_actions" in stats
        assert "game_duration" in stats

    def test_obs_space_contains_observation(self, env: ClashRoyaleGymEnv) -> None:
        obs, _ = env.reset()
        # Check that obs matches the declared observation space structure
        assert "troops" in obs
        assert obs["troops"].shape == env.observation_space["troops"].shape

    def test_action_space_sample(self, env: ClashRoyaleGymEnv) -> None:
        """Verify that the action space can be sampled."""
        env.reset()
        action = env.action_space.sample()
        assert "card" in action


# ══════════════════════════════════════════════════════════════════════════════
# League
# ══════════════════════════════════════════════════════════════════════════════


class TestLeague:
    """Test league / tournament system."""

    def test_single_match(self) -> None:
        p0 = HeuristicSlot("Bot-A", aggression=0.3, seed=1)
        p1 = HeuristicSlot("Bot-B", aggression=0.7, seed=2)
        result = run_match(p0, p1, time_limit=10.0, seed=0)
        assert result.player_0_name == "Bot-A"
        assert result.player_1_name == "Bot-B"
        assert result.winner in (0, 1, None)
        assert result.total_frames > 0
        assert result.game_duration > 0

    def test_tournament_basic(self) -> None:
        players = [
            HeuristicSlot("Passive", aggression=0.2, seed=10),
            HeuristicSlot("Balanced", aggression=0.5, seed=20),
        ]
        league = LeagueTournament(
            players=players, matches_per_pair=2, time_limit=10.0,
        )
        results = league.run(seed_start=0)
        assert len(results) == 2

        standings = league.get_standings()
        assert len(standings) == 2

        summary = league.summary()
        assert summary["total_matches"] == 2

    def test_tournament_three_players(self) -> None:
        players = [
            HeuristicSlot("A", aggression=0.2, seed=1),
            HeuristicSlot("B", aggression=0.5, seed=2),
            HeuristicSlot("C", aggression=0.9, seed=3),
        ]
        league = LeagueTournament(
            players=players, matches_per_pair=2, time_limit=10.0,
        )
        results = league.run()
        # 3 pairs × 2 matches = 6
        assert len(results) == 6

    def test_tournament_needs_min_players(self) -> None:
        with pytest.raises(ValueError):
            LeagueTournament(players=[HeuristicSlot("Solo")])

    def test_standings_sorted(self) -> None:
        players = [
            HeuristicSlot("A", aggression=0.2, seed=1),
            HeuristicSlot("B", aggression=0.8, seed=2),
        ]
        league = LeagueTournament(
            players=players, matches_per_pair=4, time_limit=10.0,
        )
        league.run()
        standings = league.get_standings()
        # First should have highest win rate
        assert standings[0].win_rate >= standings[-1].win_rate

    def test_progress_callback(self) -> None:
        calls: list[int] = []
        players = [
            HeuristicSlot("A", aggression=0.3, seed=1),
            HeuristicSlot("B", aggression=0.7, seed=2),
        ]
        league = LeagueTournament(
            players=players, matches_per_pair=2, time_limit=10.0,
        )

        def cb(idx: int, total: int, result: object) -> None:
            calls.append(idx)

        league.run(progress_cb=cb)
        assert len(calls) == 2
