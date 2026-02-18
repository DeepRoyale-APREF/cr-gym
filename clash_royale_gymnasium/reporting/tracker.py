"""Training session tracker — accumulates stats across episodes and evaluations.

Keeps a running log of:
- Total simulated game-time and wall-clock time.
- Per-episode metrics (reward, actions, duration, win/loss).
- Periodic evaluation snapshots from league tournaments.

Designed to be plugged into any training loop and queried for live dashboards
or post-training analysis.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(slots=True)
class EpisodeRecord:
    """One completed episode."""

    episode: int
    total_reward: float
    n_actions: int
    steps: int
    game_duration: float  # simulated seconds
    winner: Optional[int]  # 0 = agent, 1 = opponent, None = draw
    towers_destroyed: int
    own_towers_alive: int
    leaked_elixir: float
    wall_time: float  # wall-clock seconds for this episode


@dataclass(slots=True)
class EvalSnapshot:
    """Snapshot of an evaluation round (league tournament)."""

    eval_id: int
    episode: int  # training episode at which eval happened
    win_rate: float
    avg_reward: float
    avg_towers_destroyed: float
    avg_game_duration: float
    avg_leaked_elixir: float
    avg_actions: float
    total_eval_matches: int
    opponents: List[str]
    wall_time: float  # wall-clock seconds for the evaluation
    extra: Dict[str, Any] = field(default_factory=dict)


class TrainingTracker:
    """Accumulates training statistics across episodes and evaluations.

    Parameters
    ----------
    name : str
        Identifier for the training run (e.g. ``"PPO-v1-run3"``).
    log_dir : Path or str, optional
        Directory to persist CSV / JSON logs.  Created on first write.

    Example
    -------
    >>> tracker = TrainingTracker("PPO-experiment-1", log_dir="./logs")
    >>> tracker.begin_episode()
    >>> # ... run env steps ...
    >>> tracker.end_episode(env.episode_stats)
    >>> tracker.current_summary()
    """

    def __init__(self, name: str, log_dir: Optional[str | Path] = None) -> None:
        self.name = name
        self._log_dir = Path(log_dir) if log_dir else None
        self._episodes: List[EpisodeRecord] = []
        self._evals: List[EvalSnapshot] = []
        self._episode_counter = 0
        self._eval_counter = 0
        self._t0_run = time.monotonic()
        self._t0_episode: float = 0.0

    # ── Episode tracking ──────────────────────────────────────────────────

    def begin_episode(self) -> None:
        """Call at the start of each training episode."""
        self._t0_episode = time.monotonic()

    def end_episode(self, stats: Dict[str, Any]) -> EpisodeRecord:
        """Call at the end of each training episode.

        Parameters
        ----------
        stats : dict
            The ``episode_stats`` dict from :class:`ClashRoyaleGymEnv`.

        Returns
        -------
        EpisodeRecord
            The recorded data for this episode.
        """
        wall = time.monotonic() - self._t0_episode
        self._episode_counter += 1

        rec = EpisodeRecord(
            episode=self._episode_counter,
            total_reward=stats.get("total_reward", 0.0),
            n_actions=stats.get("n_actions", 0),
            steps=stats.get("steps", 0),
            game_duration=stats.get("game_duration", 0.0),
            winner=stats.get("winner"),
            towers_destroyed=stats.get("towers_destroyed", 0),
            own_towers_alive=stats.get("own_towers_alive", 0),
            leaked_elixir=stats.get("leaked_elixir", 0.0),
            wall_time=wall,
        )
        self._episodes.append(rec)
        return rec

    # ── Evaluation tracking ───────────────────────────────────────────────

    def record_eval(
        self,
        win_rate: float,
        avg_reward: float,
        avg_towers_destroyed: float,
        avg_game_duration: float,
        avg_leaked_elixir: float,
        avg_actions: float,
        total_eval_matches: int,
        opponents: List[str],
        wall_time: float,
        **extra: Any,
    ) -> EvalSnapshot:
        """Record an evaluation snapshot (e.g. after a league tournament).

        Parameters
        ----------
        win_rate : float
            Agent win rate in the evaluation round.
        avg_reward : float
            Mean episode reward during evaluation.
        avg_towers_destroyed : float
            Mean towers destroyed per match.
        avg_game_duration : float
            Mean game duration (simulated seconds) per match.
        avg_leaked_elixir : float
            Mean leaked elixir per match.
        avg_actions : float
            Mean actions per match.
        total_eval_matches : int
            How many matches were played in this eval round.
        opponents : list[str]
            Names of opponents faced.
        wall_time : float
            Wall-clock time for the full evaluation.
        **extra
            Any additional metrics (e.g. ELO, loss values).

        Returns
        -------
        EvalSnapshot
        """
        self._eval_counter += 1
        snap = EvalSnapshot(
            eval_id=self._eval_counter,
            episode=self._episode_counter,
            win_rate=win_rate,
            avg_reward=avg_reward,
            avg_towers_destroyed=avg_towers_destroyed,
            avg_game_duration=avg_game_duration,
            avg_leaked_elixir=avg_leaked_elixir,
            avg_actions=avg_actions,
            total_eval_matches=total_eval_matches,
            opponents=opponents,
            wall_time=wall_time,
            extra=dict(extra),
        )
        self._evals.append(snap)
        return snap

    def record_eval_from_tournament(
        self,
        agent_name: str,
        tournament_summary: Dict[str, Any],
        wall_time: float,
        **extra: Any,
    ) -> EvalSnapshot:
        """Convenience — extract agent stats from :meth:`LeagueTournament.summary`.

        Parameters
        ----------
        agent_name : str
            Name of the agent in the tournament standings.
        tournament_summary : dict
            Output of ``tournament.summary()``.
        wall_time : float
            Wall-clock seconds for the tournament.
        **extra
            Additional metrics.
        """
        standings = tournament_summary["standings"]
        agent_stats: Optional[Dict[str, Any]] = None
        for s in standings:
            if s["name"] == agent_name:
                agent_stats = s
                break
        if agent_stats is None:
            raise ValueError(f"Agent '{agent_name}' not found in standings")

        opponents = [s["name"] for s in standings if s["name"] != agent_name]

        # Compute avg reward from episodes if available, else 0
        recent_eps = self._episodes[-50:] if self._episodes else []
        avg_reward = (
            sum(e.total_reward for e in recent_eps) / max(len(recent_eps), 1)
            if recent_eps
            else 0.0
        )

        return self.record_eval(
            win_rate=agent_stats["win_rate"],
            avg_reward=avg_reward,
            avg_towers_destroyed=agent_stats["avg_towers_destroyed"],
            avg_game_duration=agent_stats["avg_game_duration"],
            avg_leaked_elixir=agent_stats["avg_leaked_elixir"],
            avg_actions=agent_stats["avg_actions_per_match"],
            total_eval_matches=agent_stats.get("total_matches", 0),
            opponents=opponents,
            wall_time=wall_time,
            **extra,
        )

    # ── Queries ───────────────────────────────────────────────────────────

    @property
    def total_episodes(self) -> int:
        return self._episode_counter

    @property
    def total_simulated_time(self) -> float:
        """Total simulated game-time in seconds across all episodes."""
        return sum(e.game_duration for e in self._episodes)

    @property
    def total_simulated_time_hours(self) -> float:
        """Total simulated game-time in hours."""
        return self.total_simulated_time / 3600.0

    @property
    def total_wall_time(self) -> float:
        """Wall-clock seconds since tracker creation."""
        return time.monotonic() - self._t0_run

    @property
    def total_wall_time_hours(self) -> float:
        return self.total_wall_time / 3600.0

    @property
    def total_matches(self) -> int:
        """Total training episodes (each episode = 1 match)."""
        return self._episode_counter

    @property
    def total_eval_matches(self) -> int:
        """Total evaluation matches across all eval rounds."""
        return sum(e.total_eval_matches for e in self._evals)

    @property
    def episodes(self) -> List[EpisodeRecord]:
        return list(self._episodes)

    @property
    def evals(self) -> List[EvalSnapshot]:
        return list(self._evals)

    def win_rate(self, last_n: Optional[int] = None) -> float:
        """Fraction of episodes where the agent (player 0) won.

        Parameters
        ----------
        last_n : int, optional
            Only consider the last *n* episodes.  Default: all.
        """
        eps = self._episodes[-last_n:] if last_n else self._episodes
        if not eps:
            return 0.0
        wins = sum(1 for e in eps if e.winner == 0)
        return wins / len(eps)

    def avg_reward(self, last_n: Optional[int] = None) -> float:
        """Mean total reward per episode."""
        eps = self._episodes[-last_n:] if last_n else self._episodes
        if not eps:
            return 0.0
        return sum(e.total_reward for e in eps) / len(eps)

    def avg_towers_destroyed(self, last_n: Optional[int] = None) -> float:
        """Mean towers destroyed per episode."""
        eps = self._episodes[-last_n:] if last_n else self._episodes
        if not eps:
            return 0.0
        return sum(e.towers_destroyed for e in eps) / len(eps)

    def avg_game_duration(self, last_n: Optional[int] = None) -> float:
        """Mean game duration (simulated seconds) per episode."""
        eps = self._episodes[-last_n:] if last_n else self._episodes
        if not eps:
            return 0.0
        return sum(e.game_duration for e in eps) / len(eps)

    def avg_leaked_elixir(self, last_n: Optional[int] = None) -> float:
        """Mean leaked elixir per episode."""
        eps = self._episodes[-last_n:] if last_n else self._episodes
        if not eps:
            return 0.0
        return sum(e.leaked_elixir for e in eps) / len(eps)

    def avg_actions(self, last_n: Optional[int] = None) -> float:
        """Mean number of actions per episode."""
        eps = self._episodes[-last_n:] if last_n else self._episodes
        if not eps:
            return 0.0
        return sum(e.n_actions for e in eps) / len(eps)

    def current_summary(self, last_n: int = 100) -> Dict[str, Any]:
        """Snapshot of current training progress.

        Parameters
        ----------
        last_n : int
            Window size for rolling averages.

        Returns
        -------
        dict
            Keys: ``total_episodes``, ``total_matches``, ``total_simulated_time_h``,
            ``total_wall_time_h``, ``sim_to_wall_ratio``, ``episodes_per_hour``,
            ``rolling_win_rate``, ``rolling_avg_reward``, ``rolling_avg_towers``,
            ``rolling_avg_duration``, ``rolling_avg_leaked``, ``rolling_avg_actions``,
            ``n_evals``, ``latest_eval_win_rate``.
        """
        wall_h = self.total_wall_time_hours
        sim_h = self.total_simulated_time_hours
        return {
            "name": self.name,
            "total_episodes": self.total_episodes,
            "total_matches": self.total_matches,
            "total_simulated_time_h": round(sim_h, 3),
            "total_wall_time_h": round(wall_h, 4),
            "sim_to_wall_ratio": round(sim_h / max(wall_h, 1e-9), 1),
            "episodes_per_hour": round(self.total_episodes / max(wall_h, 1e-9), 0),
            "rolling_win_rate": round(self.win_rate(last_n), 4),
            "rolling_avg_reward": round(self.avg_reward(last_n), 2),
            "rolling_avg_towers": round(self.avg_towers_destroyed(last_n), 2),
            "rolling_avg_duration": round(self.avg_game_duration(last_n), 1),
            "rolling_avg_leaked": round(self.avg_leaked_elixir(last_n), 2),
            "rolling_avg_actions": round(self.avg_actions(last_n), 1),
            "n_evals": len(self._evals),
            "latest_eval_win_rate": (
                round(self._evals[-1].win_rate, 4) if self._evals else None
            ),
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, directory: Optional[str | Path] = None) -> Path:
        """Write episode log and eval snapshots to disk.

        Parameters
        ----------
        directory : Path or str, optional
            Override output directory (default: ``self._log_dir``).

        Returns
        -------
        Path
            The directory where files were written.
        """
        out = Path(directory) if directory else self._log_dir
        if out is None:
            raise ValueError("No log_dir configured and no directory given")
        out.mkdir(parents=True, exist_ok=True)

        # Episode CSV
        ep_path = out / f"{self.name}_episodes.csv"
        with open(ep_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode", "total_reward", "n_actions", "steps",
                    "game_duration", "winner", "towers_destroyed",
                    "own_towers_alive", "leaked_elixir", "wall_time",
                ],
            )
            writer.writeheader()
            for e in self._episodes:
                writer.writerow({
                    "episode": e.episode,
                    "total_reward": round(e.total_reward, 4),
                    "n_actions": e.n_actions,
                    "steps": e.steps,
                    "game_duration": round(e.game_duration, 2),
                    "winner": e.winner,
                    "towers_destroyed": e.towers_destroyed,
                    "own_towers_alive": e.own_towers_alive,
                    "leaked_elixir": round(e.leaked_elixir, 4),
                    "wall_time": round(e.wall_time, 3),
                })

        # Eval JSON
        eval_path = out / f"{self.name}_evals.json"
        eval_data = []
        for ev in self._evals:
            eval_data.append({
                "eval_id": ev.eval_id,
                "episode": ev.episode,
                "win_rate": round(ev.win_rate, 4),
                "avg_reward": round(ev.avg_reward, 2),
                "avg_towers_destroyed": round(ev.avg_towers_destroyed, 2),
                "avg_game_duration": round(ev.avg_game_duration, 1),
                "avg_leaked_elixir": round(ev.avg_leaked_elixir, 2),
                "avg_actions": round(ev.avg_actions, 1),
                "total_eval_matches": ev.total_eval_matches,
                "opponents": ev.opponents,
                "wall_time": round(ev.wall_time, 2),
                "extra": ev.extra,
            })
        with open(eval_path, "w") as f:
            json.dump(eval_data, f, indent=2)

        # Summary JSON
        summary_path = out / f"{self.name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.current_summary(), f, indent=2)

        return out

    def load_episodes(self, path: str | Path) -> None:
        """Load episode records from a CSV file (for resuming / analysis).

        Parameters
        ----------
        path : Path or str
            Path to a ``*_episodes.csv`` file.
        """
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = EpisodeRecord(
                    episode=int(row["episode"]),
                    total_reward=float(row["total_reward"]),
                    n_actions=int(row["n_actions"]),
                    steps=int(row["steps"]),
                    game_duration=float(row["game_duration"]),
                    winner=int(row["winner"]) if row["winner"] not in ("", "None") else None,
                    towers_destroyed=int(row["towers_destroyed"]),
                    own_towers_alive=int(row["own_towers_alive"]),
                    leaked_elixir=float(row["leaked_elixir"]),
                    wall_time=float(row["wall_time"]),
                )
                self._episodes.append(rec)
        if self._episodes:
            self._episode_counter = max(e.episode for e in self._episodes)

    def __repr__(self) -> str:
        return (
            f"TrainingTracker(name={self.name!r}, "
            f"episodes={self.total_episodes}, "
            f"sim_time={self.total_simulated_time_hours:.2f}h, "
            f"wall_time={self.total_wall_time_hours:.2f}h)"
        )
