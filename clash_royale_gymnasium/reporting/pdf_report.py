"""Generate a well-designed PDF report from league tournament results.

Requires ``matplotlib`` and ``fpdf2`` (install with ``pip install clash-royale-gymnasium[report]``).
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from clash_royale_gymnasium.league.tournament import LeagueTournament, PlayerStats


def generate_report(
    tournament: LeagueTournament,
    output_path: str | Path = "league_report.pdf",
    title: str = "Clash Royale League Report",
) -> Path:
    """Generate a PDF report from a completed :class:`LeagueTournament`.

    Parameters
    ----------
    tournament : LeagueTournament
        A tournament that has already been run (``tournament.run()`` called).
    output_path : str or Path
        Where to save the PDF.
    title : str
        Report title on the cover page.

    Returns
    -------
    Path
        Absolute path to the generated PDF.
    """
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ImportError(
            "PDF reporting requires fpdf2. Install with: "
            "pip install clash-royale-gymnasium[report]"
        ) from exc

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "PDF reporting requires matplotlib. Install with: "
            "pip install clash-royale-gymnasium[report]"
        ) from exc

    output_path = Path(output_path)
    standings = tournament.get_standings()
    summary = tournament.summary()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Cover page ────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 40, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(
        0, 10,
        f"Total matches: {summary['total_matches']}  |  "
        f"Elapsed: {summary['elapsed_seconds']}s  |  "
        f"Throughput: {summary['matches_per_hour']:.0f} matches/h",
        new_x="LMARGIN", new_y="NEXT", align="C",
    )
    pdf.ln(10)

    # ── Standings table ───────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "Standings", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    _draw_standings_table(pdf, standings)
    pdf.ln(10)

    # ── Win rate bar chart ────────────────────────────────────────────────
    chart_path = _make_win_rate_chart(standings)
    if chart_path:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "Win Rate Comparison", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        pdf.image(str(chart_path), x=20, w=170)
        pdf.ln(10)

    # ── Performance radar chart ───────────────────────────────────────────
    radar_path = _make_radar_chart(standings)
    if radar_path:
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "Performance Profile", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        pdf.image(str(radar_path), x=20, w=170)
        pdf.ln(10)

    # ── Per-player detail pages ───────────────────────────────────────────
    for ps in standings:
        pdf.add_page()
        _draw_player_detail(pdf, ps, tournament)

    # ── Match log ─────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "Match Log", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    _draw_match_log(pdf, tournament)

    pdf.output(str(output_path))
    return output_path.resolve()


# ══════════════════════════════════════════════════════════════════════════════
# Internal drawing helpers
# ══════════════════════════════════════════════════════════════════════════════


def _draw_standings_table(pdf: Any, standings: List[PlayerStats]) -> None:
    """Draw a coloured standings table."""
    headers = ["#", "Player", "W", "L", "D", "Win%", "Avg Towers", "Avg Duration", "Avg Leak"]
    col_widths = [10, 40, 15, 15, 15, 20, 25, 25, 25]

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(52, 73, 94)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 9)

    for rank, ps in enumerate(standings, 1):
        if rank % 2 == 0:
            pdf.set_fill_color(236, 240, 241)
        else:
            pdf.set_fill_color(255, 255, 255)

        row = [
            str(rank),
            ps.name,
            str(ps.wins),
            str(ps.losses),
            str(ps.draws),
            f"{ps.win_rate:.1%}",
            f"{ps.avg_towers_destroyed:.1f}",
            f"{ps.avg_game_duration:.0f}s",
            f"{ps.avg_leaked_elixir:.1f}",
        ]
        for i, val in enumerate(row):
            pdf.cell(col_widths[i], 7, val, border=1, fill=True, align="C")
        pdf.ln()


def _draw_player_detail(pdf: Any, ps: PlayerStats, tournament: LeagueTournament) -> None:
    """Draw a detail section for one player."""
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, f"Player: {ps.name}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    lines = [
        f"Matches: {ps.total_matches}  |  Wins: {ps.wins}  |  "
        f"Losses: {ps.losses}  |  Draws: {ps.draws}",
        f"Win Rate: {ps.win_rate:.1%}",
        f"Avg Towers Destroyed: {ps.avg_towers_destroyed:.2f}",
        f"Avg Towers Lost: {ps.total_towers_lost / max(ps.total_matches, 1):.2f}",
        f"Avg Actions/Match: {ps.avg_actions_per_match:.1f}",
        f"Avg Game Duration: {ps.avg_game_duration:.1f}s",
        f"Avg Leaked Elixir: {ps.avg_leaked_elixir:.2f}",
    ]
    for line in lines:
        pdf.cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")

    # Win rate per opponent
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Head-to-Head", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    h2h = _head_to_head(ps.name, tournament)
    for opp_name, (w, l, d) in h2h.items():
        total = w + l + d
        wr = w / max(total, 1)
        pdf.cell(
            0, 6,
            f"  vs {opp_name}: {w}W / {l}L / {d}D  ({wr:.0%})",
            new_x="LMARGIN", new_y="NEXT",
        )


def _head_to_head(
    player_name: str, tournament: LeagueTournament,
) -> Dict[str, tuple[int, int, int]]:
    """Compute head-to-head win/loss/draw for a player."""
    h2h: Dict[str, list[int]] = {}
    for r in tournament.results:
        if r.player_0_name == player_name:
            opp = r.player_1_name
            h2h.setdefault(opp, [0, 0, 0])
            if r.winner == 0:
                h2h[opp][0] += 1
            elif r.winner == 1:
                h2h[opp][1] += 1
            else:
                h2h[opp][2] += 1
        elif r.player_1_name == player_name:
            opp = r.player_0_name
            h2h.setdefault(opp, [0, 0, 0])
            if r.winner == 1:
                h2h[opp][0] += 1
            elif r.winner == 0:
                h2h[opp][1] += 1
            else:
                h2h[opp][2] += 1
    return {k: (v[0], v[1], v[2]) for k, v in h2h.items()}


def _draw_match_log(pdf: Any, tournament: LeagueTournament) -> None:
    """Draw a compact match log table."""
    headers = ["#", "Player 0", "Player 1", "Winner", "Duration", "P0 Towers", "P1 Towers"]
    widths = [10, 35, 35, 30, 25, 25, 25]

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(52, 73, 94)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 8)

    for idx, r in enumerate(tournament.results):
        if idx % 2 == 0:
            pdf.set_fill_color(255, 255, 255)
        else:
            pdf.set_fill_color(245, 245, 245)

        row = [
            str(idx + 1),
            r.player_0_name[:15],
            r.player_1_name[:15],
            r.winner_name[:15],
            f"{r.game_duration:.0f}s",
            str(r.p0_towers_destroyed),
            str(r.p1_towers_destroyed),
        ]
        for i, val in enumerate(row):
            pdf.cell(widths[i], 6, val, border=1, fill=True, align="C")
        pdf.ln()

        # Page break if near bottom
        if pdf.get_y() > 270:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(52, 73, 94)
            pdf.set_text_color(255, 255, 255)
            for i, h in enumerate(headers):
                pdf.cell(widths[i], 7, h, border=1, fill=True, align="C")
            pdf.ln()
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 8)


# ══════════════════════════════════════════════════════════════════════════════
# Chart helpers (matplotlib → temp PNG → embedded in PDF)
# ══════════════════════════════════════════════════════════════════════════════


def _make_win_rate_chart(standings: List[PlayerStats]) -> Optional[Path]:
    """Bar chart of win rates."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    names = [s.name for s in standings]
    rates = [s.win_rate * 100 for s in standings]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.viridis([i / max(len(names) - 1, 1) for i in range(len(names))])  # type: ignore[attr-defined]
    bars = ax.barh(names, rates, color=colors)
    ax.set_xlabel("Win Rate (%)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    path = Path(tempfile.mktemp(suffix=".png"))
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    return path


def _make_radar_chart(standings: List[PlayerStats]) -> Optional[Path]:
    """Radar chart comparing key metrics across players."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    if len(standings) < 2:
        return None

    categories = ["Win Rate", "Towers Destroyed", "Elixir Efficiency", "Activity"]
    n = len(categories)
    angles = [i * 2 * np.pi / n for i in range(n)]
    angles.append(angles[0])  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Normalise metrics to [0, 1]
    max_towers = max(s.avg_towers_destroyed for s in standings) or 1
    max_actions = max(s.avg_actions_per_match for s in standings) or 1
    max_leak = max(s.avg_leaked_elixir for s in standings) or 1

    for ps in standings:
        values = [
            ps.win_rate,
            ps.avg_towers_destroyed / max_towers,
            1.0 - (ps.avg_leaked_elixir / max_leak),  # lower leak = better
            ps.avg_actions_per_match / max_actions,
        ]
        values.append(values[0])
        ax.plot(angles, values, "o-", linewidth=1.5, label=ps.name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()

    path = Path(tempfile.mktemp(suffix=".png"))
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
