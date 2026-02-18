#!/usr/bin/env python3
"""03 — PDF league report generation.

Runs a quick league tournament between heuristic bots of varying aggression,
then generates a PDF report with standings, charts, and match logs.

Requires the ``[report]`` extra:
    pip install clash-royale-gymnasium[report]

Usage:
    python examples/03_league_report.py
"""

from __future__ import annotations

from clash_royale_gymnasium import HeuristicSlot, LeagueTournament
from clash_royale_gymnasium.reporting.pdf_report import generate_report


def main() -> None:
    # ── Build player roster ───────────────────────────────────────────────
    players = [
        HeuristicSlot("Timid",       aggression=0.1, seed=10),
        HeuristicSlot("Cautious",    aggression=0.3, seed=20),
        HeuristicSlot("Balanced",    aggression=0.5, seed=30),
        HeuristicSlot("Bold",        aggression=0.7, seed=40),
        HeuristicSlot("Berserker",   aggression=0.95, seed=50),
    ]

    # ── Run tournament ────────────────────────────────────────────────────
    tournament = LeagueTournament(
        players=players,
        matches_per_pair=6,   # 6 matches per pair → 60 total (C(5,2)=10 pairs)
        speed_multiplier=1.0,
    )

    print(f"Running league: {len(players)} players, "
          f"{tournament.matches_per_pair} matches/pair")

    def on_progress(idx: int, total: int, result: object) -> None:
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  match {idx + 1}/{total}")

    tournament.run(seed_start=0, progress_cb=on_progress)

    # ── Print standings ───────────────────────────────────────────────────
    print("\n── Standings ───────────────────────────────────")
    for rank, ps in enumerate(tournament.get_standings(), 1):
        print(
            f"  #{rank} {ps.name:<12s}  "
            f"W={ps.wins:>2d}  L={ps.losses:>2d}  D={ps.draws:>2d}  "
            f"WR={ps.win_rate:.0%}  "
            f"AvgTow={ps.avg_towers_destroyed:.2f}  "
            f"AvgLeak={ps.avg_leaked_elixir:.2f}"
        )

    # ── Print summary stats ───────────────────────────────────────────────
    s = tournament.summary()
    print(f"\n  Total matches:   {s['total_matches']}")
    print(f"  Elapsed:         {s['elapsed_seconds']:.1f}s")
    print(f"  Throughput:      {s['matches_per_hour']:.0f} matches/h")

    # ── Generate PDF report ───────────────────────────────────────────────
    output = generate_report(
        tournament,
        output_path="./logs/league_report.pdf",
        title="Arena 1 Heuristic Bot League",
    )
    print(f"\nPDF report saved to: {output}")


if __name__ == "__main__":
    main()
