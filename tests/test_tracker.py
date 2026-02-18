"""Tests for the TrainingTracker module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from clash_royale_gymnasium.reporting.tracker import (
    EpisodeRecord,
    EvalSnapshot,
    TrainingTracker,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _episode_stats(
    reward: float = 5.0,
    n_actions: int = 10,
    steps: int = 100,
    duration: float = 3.33,
    winner: int | None = 0,
    towers: int = 1,
    alive: int = 3,
    leaked: float = 0.5,
) -> dict:
    return {
        "total_reward": reward,
        "n_actions": n_actions,
        "steps": steps,
        "game_duration": duration,
        "winner": winner,
        "towers_destroyed": towers,
        "own_towers_alive": alive,
        "leaked_elixir": leaked,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestTrainingTracker:
    """Core tracker functionality."""

    def test_empty_tracker(self) -> None:
        tracker = TrainingTracker("test")
        assert tracker.total_episodes == 0
        assert tracker.total_simulated_time == 0.0
        assert tracker.total_matches == 0
        assert tracker.win_rate() == 0.0
        assert tracker.avg_reward() == 0.0

    def test_begin_end_episode(self) -> None:
        tracker = TrainingTracker("test")
        tracker.begin_episode()
        rec = tracker.end_episode(_episode_stats(reward=10.0, winner=0))
        assert isinstance(rec, EpisodeRecord)
        assert rec.episode == 1
        assert rec.total_reward == 10.0
        assert rec.winner == 0
        assert rec.wall_time > 0

    def test_episode_counter(self) -> None:
        tracker = TrainingTracker("test")
        for i in range(5):
            tracker.begin_episode()
            tracker.end_episode(_episode_stats())
        assert tracker.total_episodes == 5
        assert tracker.total_matches == 5

    def test_simulated_time(self) -> None:
        tracker = TrainingTracker("test")
        for _ in range(3):
            tracker.begin_episode()
            tracker.end_episode(_episode_stats(duration=60.0))
        assert tracker.total_simulated_time == pytest.approx(180.0)
        assert tracker.total_simulated_time_hours == pytest.approx(0.05)

    def test_win_rate(self) -> None:
        tracker = TrainingTracker("test")
        # 3 wins, 2 losses
        for w in [0, 0, 0, 1, 1]:
            tracker.begin_episode()
            tracker.end_episode(_episode_stats(winner=w))
        assert tracker.win_rate() == pytest.approx(0.6)

    def test_win_rate_last_n(self) -> None:
        tracker = TrainingTracker("test")
        for w in [0, 0, 0, 1, 1]:
            tracker.begin_episode()
            tracker.end_episode(_episode_stats(winner=w))
        # Last 2 are losses → 0% win rate
        assert tracker.win_rate(last_n=2) == pytest.approx(0.0)
        # Last 3: W, L, L → 33%
        assert tracker.win_rate(last_n=3) == pytest.approx(1 / 3)

    def test_avg_reward(self) -> None:
        tracker = TrainingTracker("test")
        for r in [10.0, 20.0, 30.0]:
            tracker.begin_episode()
            tracker.end_episode(_episode_stats(reward=r))
        assert tracker.avg_reward() == pytest.approx(20.0)

    def test_avg_towers_destroyed(self) -> None:
        tracker = TrainingTracker("test")
        for t in [0, 1, 2, 3]:
            tracker.begin_episode()
            tracker.end_episode(_episode_stats(towers=t))
        assert tracker.avg_towers_destroyed() == pytest.approx(1.5)

    def test_avg_leaked_elixir(self) -> None:
        tracker = TrainingTracker("test")
        for lk in [1.0, 2.0, 3.0]:
            tracker.begin_episode()
            tracker.end_episode(_episode_stats(leaked=lk))
        assert tracker.avg_leaked_elixir() == pytest.approx(2.0)

    def test_avg_actions(self) -> None:
        tracker = TrainingTracker("test")
        for a in [10, 20, 30]:
            tracker.begin_episode()
            tracker.end_episode(_episode_stats(n_actions=a))
        assert tracker.avg_actions() == pytest.approx(20.0)

    def test_avg_game_duration(self) -> None:
        tracker = TrainingTracker("test")
        for d in [60.0, 120.0, 180.0]:
            tracker.begin_episode()
            tracker.end_episode(_episode_stats(duration=d))
        assert tracker.avg_game_duration() == pytest.approx(120.0)


class TestTrackerSummary:
    """current_summary() output."""

    def test_summary_keys(self) -> None:
        tracker = TrainingTracker("test")
        tracker.begin_episode()
        tracker.end_episode(_episode_stats())
        s = tracker.current_summary()
        expected_keys = {
            "name", "total_episodes", "total_matches",
            "total_simulated_time_h", "total_wall_time_h",
            "sim_to_wall_ratio", "episodes_per_hour",
            "rolling_win_rate", "rolling_avg_reward",
            "rolling_avg_towers", "rolling_avg_duration",
            "rolling_avg_leaked", "rolling_avg_actions",
            "n_evals", "latest_eval_win_rate",
        }
        assert set(s.keys()) == expected_keys

    def test_summary_values(self) -> None:
        tracker = TrainingTracker("test")
        for _ in range(10):
            tracker.begin_episode()
            tracker.end_episode(_episode_stats(reward=5.0, winner=0, towers=2))
        s = tracker.current_summary()
        assert s["total_episodes"] == 10
        assert s["rolling_win_rate"] == 1.0
        assert s["rolling_avg_towers"] == 2.0
        assert s["latest_eval_win_rate"] is None  # no evals yet


class TestTrackerEval:
    """Evaluation snapshot recording."""

    def test_record_eval(self) -> None:
        tracker = TrainingTracker("test")
        tracker.begin_episode()
        tracker.end_episode(_episode_stats())

        snap = tracker.record_eval(
            win_rate=0.75,
            avg_reward=15.0,
            avg_towers_destroyed=1.5,
            avg_game_duration=120.0,
            avg_leaked_elixir=0.3,
            avg_actions=20.0,
            total_eval_matches=10,
            opponents=["BotA", "BotB"],
            wall_time=5.0,
            elo=1200,
        )
        assert isinstance(snap, EvalSnapshot)
        assert snap.eval_id == 1
        assert snap.episode == 1
        assert snap.win_rate == 0.75
        assert snap.extra == {"elo": 1200}

    def test_record_eval_from_tournament_dict(self) -> None:
        tracker = TrainingTracker("test")
        tracker.begin_episode()
        tracker.end_episode(_episode_stats())

        summary = {
            "total_matches": 20,
            "elapsed_seconds": 10.0,
            "matches_per_hour": 7200.0,
            "standings": [
                {
                    "name": "MyAgent",
                    "wins": 8, "losses": 2, "draws": 0,
                    "total_matches": 10,
                    "win_rate": 0.8,
                    "avg_towers_destroyed": 1.5,
                    "avg_game_duration": 150.0,
                    "avg_actions_per_match": 25.0,
                    "avg_leaked_elixir": 0.2,
                },
                {
                    "name": "Opponent1",
                    "wins": 2, "losses": 8, "draws": 0,
                    "total_matches": 10,
                    "win_rate": 0.2,
                    "avg_towers_destroyed": 0.5,
                    "avg_game_duration": 150.0,
                    "avg_actions_per_match": 15.0,
                    "avg_leaked_elixir": 1.5,
                },
            ],
        }
        snap = tracker.record_eval_from_tournament(
            agent_name="MyAgent",
            tournament_summary=summary,
            wall_time=10.0,
        )
        assert snap.win_rate == 0.8
        assert snap.opponents == ["Opponent1"]
        assert snap.total_eval_matches == 10

    def test_record_eval_unknown_agent_raises(self) -> None:
        tracker = TrainingTracker("test")
        with pytest.raises(ValueError, match="not found"):
            tracker.record_eval_from_tournament(
                agent_name="Unknown",
                tournament_summary={"standings": []},
                wall_time=0.0,
            )

    def test_total_eval_matches(self) -> None:
        tracker = TrainingTracker("test")
        tracker.record_eval(
            win_rate=0.5, avg_reward=0, avg_towers_destroyed=0,
            avg_game_duration=0, avg_leaked_elixir=0, avg_actions=0,
            total_eval_matches=20, opponents=[], wall_time=1.0,
        )
        tracker.record_eval(
            win_rate=0.6, avg_reward=0, avg_towers_destroyed=0,
            avg_game_duration=0, avg_leaked_elixir=0, avg_actions=0,
            total_eval_matches=30, opponents=[], wall_time=1.0,
        )
        assert tracker.total_eval_matches == 50

    def test_latest_eval_in_summary(self) -> None:
        tracker = TrainingTracker("test")
        tracker.begin_episode()
        tracker.end_episode(_episode_stats())
        tracker.record_eval(
            win_rate=0.85, avg_reward=0, avg_towers_destroyed=0,
            avg_game_duration=0, avg_leaked_elixir=0, avg_actions=0,
            total_eval_matches=10, opponents=[], wall_time=1.0,
        )
        s = tracker.current_summary()
        assert s["latest_eval_win_rate"] == 0.85
        assert s["n_evals"] == 1


class TestTrackerPersistence:
    """Save / load round-trip."""

    def test_save_creates_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TrainingTracker("test-save", log_dir=tmpdir)
            for i in range(3):
                tracker.begin_episode()
                tracker.end_episode(_episode_stats(reward=float(i)))
            tracker.record_eval(
                win_rate=0.5, avg_reward=1.0, avg_towers_destroyed=1.0,
                avg_game_duration=100.0, avg_leaked_elixir=0.5, avg_actions=10.0,
                total_eval_matches=5, opponents=["BotA"], wall_time=2.0,
            )

            out = tracker.save()
            assert (out / "test-save_episodes.csv").exists()
            assert (out / "test-save_evals.json").exists()
            assert (out / "test-save_summary.json").exists()

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            tracker1 = TrainingTracker("rt", log_dir=tmpdir)
            for w in [0, 1, 0, None]:
                tracker1.begin_episode()
                tracker1.end_episode(_episode_stats(winner=w))
            tracker1.save()

            # Load into fresh tracker
            tracker2 = TrainingTracker("rt-loaded")
            tracker2.load_episodes(Path(tmpdir) / "rt_episodes.csv")

            assert tracker2.total_episodes == 4
            assert tracker2.win_rate() == pytest.approx(0.5)

    def test_save_without_dir_raises(self) -> None:
        tracker = TrainingTracker("no-dir")
        with pytest.raises(ValueError, match="No log_dir"):
            tracker.save()

    def test_eval_json_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TrainingTracker("json-test", log_dir=tmpdir)
            tracker.record_eval(
                win_rate=0.7, avg_reward=5.0, avg_towers_destroyed=2.0,
                avg_game_duration=150.0, avg_leaked_elixir=0.3, avg_actions=18.0,
                total_eval_matches=12, opponents=["X", "Y"], wall_time=3.0,
                custom_metric=42,
            )
            tracker.save()

            with open(Path(tmpdir) / "json-test_evals.json") as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["win_rate"] == 0.7
            assert data[0]["extra"]["custom_metric"] == 42


class TestTrackerRepr:
    """String representation."""

    def test_repr(self) -> None:
        tracker = TrainingTracker("repr-test")
        r = repr(tracker)
        assert "repr-test" in r
        assert "episodes=0" in r
