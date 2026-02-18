"""League tournament — round-robin or custom matchmaking with stats aggregation."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from clash_royale_gymnasium.league.match import MatchResult, run_match
from clash_royale_gymnasium.league.player_slot import PlayerSlot


@dataclass
class PlayerStats:
    """Aggregated statistics for a single player across all matches."""

    name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_towers_destroyed: int = 0
    total_towers_lost: int = 0
    total_leaked_elixir: float = 0.0
    total_actions: int = 0
    total_game_duration: float = 0.0
    total_matches: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.total_matches, 1)

    @property
    def avg_towers_destroyed(self) -> float:
        return self.total_towers_destroyed / max(self.total_matches, 1)

    @property
    def avg_game_duration(self) -> float:
        return self.total_game_duration / max(self.total_matches, 1)

    @property
    def avg_actions_per_match(self) -> float:
        return self.total_actions / max(self.total_matches, 1)

    @property
    def avg_leaked_elixir(self) -> float:
        return self.total_leaked_elixir / max(self.total_matches, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "total_matches": self.total_matches,
            "win_rate": round(self.win_rate, 4),
            "avg_towers_destroyed": round(self.avg_towers_destroyed, 2),
            "avg_game_duration": round(self.avg_game_duration, 1),
            "avg_actions_per_match": round(self.avg_actions_per_match, 1),
            "avg_leaked_elixir": round(self.avg_leaked_elixir, 2),
        }


ProgressCallback = Callable[[int, int, MatchResult], None]
"""``(match_index, total_matches, result) -> None``"""


class LeagueTournament:
    """Round-robin (or custom) tournament between :class:`PlayerSlot` instances.

    Parameters
    ----------
    players : list[PlayerSlot]
        All participants.
    matches_per_pair : int
        How many games each pair plays (alternating sides).
    fps : int
        Simulation framerate.
    time_limit : float
        Game duration.
    speed_multiplier : float
        Simulation speed factor.

    Example
    -------
    >>> from clash_royale_gymnasium.league import LeagueTournament
    >>> from clash_royale_gymnasium.league.player_slot import HeuristicSlot
    >>> league = LeagueTournament(
    ...     players=[
    ...         HeuristicSlot("Passive", aggression=0.2),
    ...         HeuristicSlot("Balanced", aggression=0.5),
    ...         HeuristicSlot("Aggressive", aggression=0.9),
    ...     ],
    ...     matches_per_pair=10,
    ... )
    >>> results = league.run()
    """

    def __init__(
        self,
        players: List[PlayerSlot],
        matches_per_pair: int = 10,
        fps: int = 30,
        time_limit: float = 180.0,
        speed_multiplier: float = 1.0,
    ) -> None:
        if len(players) < 2:
            raise ValueError("League requires at least 2 players")
        self.players = players
        self.matches_per_pair = matches_per_pair
        self.fps = fps
        self.time_limit = time_limit
        self.speed_multiplier = speed_multiplier

        self.results: List[MatchResult] = []
        self.stats: Dict[str, PlayerStats] = {
            p.name: PlayerStats(name=p.name) for p in players
        }
        self._elapsed: float = 0.0

    def run(
        self,
        seed_start: int = 0,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> List[MatchResult]:
        """Execute all matches and return results.

        Parameters
        ----------
        seed_start : int
            Base seed; each match gets ``seed_start + match_index``.
        progress_cb : callable, optional
            Called after each match with ``(index, total, result)``.
        """
        pairs = list(itertools.combinations(range(len(self.players)), 2))
        total = len(pairs) * self.matches_per_pair
        match_idx = 0

        t0 = time.monotonic()

        for i, j in pairs:
            for m in range(self.matches_per_pair):
                # Alternate sides each match
                if m % 2 == 0:
                    p0, p1 = self.players[i], self.players[j]
                else:
                    p0, p1 = self.players[j], self.players[i]

                result = run_match(
                    p0,
                    p1,
                    fps=self.fps,
                    time_limit=self.time_limit,
                    speed_multiplier=self.speed_multiplier,
                    seed=seed_start + match_idx,
                )
                self.results.append(result)
                self._update_stats(result)

                if progress_cb is not None:
                    progress_cb(match_idx, total, result)

                match_idx += 1

        self._elapsed = time.monotonic() - t0
        return self.results

    def get_standings(self) -> List[PlayerStats]:
        """Return players sorted by win rate (desc), then avg towers destroyed."""
        return sorted(
            self.stats.values(),
            key=lambda s: (s.win_rate, s.avg_towers_destroyed),
            reverse=True,
        )

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict suitable for reporting."""
        standings = self.get_standings()
        return {
            "total_matches": len(self.results),
            "elapsed_seconds": round(self._elapsed, 2),
            "matches_per_hour": round(len(self.results) / max(self._elapsed, 0.001) * 3600, 0),
            "standings": [s.to_dict() for s in standings],
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _update_stats(self, result: MatchResult) -> None:
        s0 = self.stats[result.player_0_name]
        s1 = self.stats[result.player_1_name]

        s0.total_matches += 1
        s1.total_matches += 1

        if result.winner == 0:
            s0.wins += 1
            s1.losses += 1
        elif result.winner == 1:
            s1.wins += 1
            s0.losses += 1
        else:
            s0.draws += 1
            s1.draws += 1

        s0.total_towers_destroyed += result.p0_towers_destroyed
        s1.total_towers_destroyed += result.p1_towers_destroyed
        s0.total_towers_lost += result.p1_towers_destroyed
        s1.total_towers_lost += result.p0_towers_destroyed
        s0.total_leaked_elixir += result.p0_leaked_elixir
        s1.total_leaked_elixir += result.p1_leaked_elixir
        s0.total_actions += result.p0_actions
        s1.total_actions += result.p1_actions
        s0.total_game_duration += result.game_duration
        s1.total_game_duration += result.game_duration
