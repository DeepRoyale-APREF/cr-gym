#!/usr/bin/env python3
"""02 — League training loop with periodic evaluation.

Demonstrates the user's primary workflow:
  1. Train agent for N episodes in the Gymnasium environment.
  2. Every ``eval_interval`` episodes, run a league tournament to measure
     performance against a roster of heuristic bots.
  3. Track ALL stats over time: simulated time, matches, rewards, win rates,
     towers, leaked elixir, actions.
  4. Print a live dashboard after each evaluation.

The "agent" here is a random policy (masked random actions).  Replace the
``agent_action_fn`` with your actual model to train for real.

Usage:
    python examples/02_league_training_loop.py
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from clash_royale_engine.core.state import State

from clash_royale_gymnasium import (
    ClashRoyaleGymEnv,
    ExternalAgentSlot,
    HeuristicSlot,
    LeagueTournament,
    TrainingTracker,
)


# ── Agent stub (replace with your model) ─────────────────────────────────────


def sample_masked_action(obs: dict) -> dict[str, int]:
    """Random policy that respects action masks."""
    mask = obs["action_mask"]
    rng = np.random.default_rng()

    card_mask = np.asarray(mask["card"], dtype=bool)
    tile_x_mask = np.asarray(mask["tile_x"], dtype=bool)
    tile_y_mask = np.asarray(mask["tile_y"], dtype=bool)

    return {
        "card": int(rng.choice(np.where(card_mask)[0])),
        "tile_x": int(rng.choice(np.where(tile_x_mask)[0])),
        "tile_y": int(rng.choice(np.where(tile_y_mask)[0])),
    }


class RandomAgentSlot(ExternalAgentSlot):
    """Wraps the random policy as an ExternalAgentSlot for league play."""

    def __init__(self, name: str = "RandomAgent") -> None:
        super().__init__(name=name, action_fn=self._decide)
        self._rng = np.random.default_rng(seed=123)

    def _decide(self, state: State) -> Optional[Tuple[int, int, int]]:
        """Engine-level action: randomly play affordable card or do nothing."""
        affordable = [
            i for i, card_name in enumerate(state.hand)
            if card_name and state.numbers.elixir >= getattr(state.numbers, "elixir", 0)
        ]
        # Simple heuristic: if elixir >= 5, play a random card at a random tile
        if state.numbers.elixir >= 5.0 and state.hand:
            card_idx = int(self._rng.integers(0, len(state.hand)))
            tile_x = int(self._rng.integers(3, 15))
            tile_y = int(self._rng.integers(0, 14))  # own side
            return (tile_x, tile_y, card_idx)
        return None


# ── Dashboard printer ─────────────────────────────────────────────────────────


def print_dashboard(tracker: TrainingTracker, eval_num: int) -> None:
    """Print a live dashboard with current training and evaluation stats."""
    s = tracker.current_summary()
    ev = tracker.evals[-1] if tracker.evals else None

    print("\n" + "=" * 72)
    print(f"  TRAINING DASHBOARD — {s['name']}  |  Eval #{eval_num}")
    print("=" * 72)
    print(f"  Total training episodes:   {s['total_episodes']:>8d}")
    print(f"  Total matches (train):     {s['total_matches']:>8d}")
    print(f"  Total eval matches:        {tracker.total_eval_matches:>8d}")
    print(f"  Simulated game-time:       {s['total_simulated_time_h']:>8.3f} h")
    print(f"  Wall-clock time:           {s['total_wall_time_h']:>8.4f} h")
    print(f"  Sim / Wall ratio:          {s['sim_to_wall_ratio']:>8.1f}x")
    print(f"  Throughput:                {s['episodes_per_hour']:>8.0f} ep/h")
    print("-" * 72)
    print(f"  Rolling win rate (100):    {s['rolling_win_rate']:>8.1%}")
    print(f"  Rolling avg reward:        {s['rolling_avg_reward']:>8.2f}")
    print(f"  Rolling avg towers:        {s['rolling_avg_towers']:>8.2f}")
    print(f"  Rolling avg duration:      {s['rolling_avg_duration']:>8.1f} s")
    print(f"  Rolling avg actions:       {s['rolling_avg_actions']:>8.1f}")
    print(f"  Rolling avg leaked elixir: {s['rolling_avg_leaked']:>8.2f}")
    if ev:
        print("-" * 72)
        print(f"  Eval win rate:             {ev.win_rate:>8.1%}")
        print(f"  Eval avg towers:           {ev.avg_towers_destroyed:>8.2f}")
        print(f"  Eval avg leaked:           {ev.avg_leaked_elixir:>8.2f}")
        print(f"  Eval opponents:            {', '.join(ev.opponents)}")
    print("=" * 72 + "\n")


# ── Main training loop ────────────────────────────────────────────────────────


def main() -> None:
    # ── Config ────────────────────────────────────────────────────────────
    total_episodes = 30        # Total training episodes (increase for real training)
    eval_interval = 10         # Evaluate every N episodes
    eval_matches_per_pair = 4  # Matches per pair during evaluation
    log_dir = "./logs/league_training"

    # ── Environment & Tracker ─────────────────────────────────────────────
    env = ClashRoyaleGymEnv(seed=0)
    tracker = TrainingTracker("RandomAgent-v0", log_dir=log_dir)

    # ── Evaluation roster ─────────────────────────────────────────────────
    eval_opponents = [
        HeuristicSlot("Passive-Bot", aggression=0.2, seed=1),
        HeuristicSlot("Balanced-Bot", aggression=0.5, seed=2),
        HeuristicSlot("Aggressive-Bot", aggression=0.8, seed=3),
    ]
    agent_slot = RandomAgentSlot("RandomAgent-v0")

    eval_count = 0

    print(f"Starting training: {total_episodes} episodes, eval every {eval_interval}")
    print(f"Eval opponents: {[o.name for o in eval_opponents]}")
    print()

    # ── Training loop ─────────────────────────────────────────────────────
    for ep in range(1, total_episodes + 1):
        tracker.begin_episode()
        obs, info = env.reset(seed=ep)
        done = False

        while not done:
            # ── Agent decides ─────────────────────────────────────────────
            action = sample_masked_action(obs)

            # ── (Here you would: store transition, update model, etc.) ────
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # ── Record episode ────────────────────────────────────────────────
        rec = tracker.end_episode(env.episode_stats)
        winner_str = (
            "W" if rec.winner == 0
            else "L" if rec.winner == 1
            else "D"
        )
        print(
            f"  ep {rec.episode:>4d} | "
            f"R={rec.total_reward:>+7.2f} | "
            f"A={rec.n_actions:>3d} | "
            f"T={rec.towers_destroyed} | "
            f"{winner_str} | "
            f"{rec.game_duration:.0f}s"
        )

        # ── Periodic evaluation ───────────────────────────────────────────
        if ep % eval_interval == 0:
            eval_count += 1
            print(f"\n>>> Running evaluation #{eval_count} at episode {ep} ...")

            t_eval = time.monotonic()

            tournament = LeagueTournament(
                players=[agent_slot] + eval_opponents,
                matches_per_pair=eval_matches_per_pair,
                speed_multiplier=1.0,
            )

            def _progress(idx: int, total: int, result: Any) -> None:
                print(
                    f"    eval match {idx + 1}/{total}: "
                    f"{result.player_0_name} vs {result.player_1_name} → "
                    f"{result.winner_name}"
                )

            tournament.run(seed_start=ep * 1000, progress_cb=_progress)

            eval_wall = time.monotonic() - t_eval

            # Record evaluation
            tracker.record_eval_from_tournament(
                agent_name="RandomAgent-v0",
                tournament_summary=tournament.summary(),
                wall_time=eval_wall,
            )

            print_dashboard(tracker, eval_count)

    env.close()

    # ── Persist all logs ──────────────────────────────────────────────────
    out = tracker.save()
    print(f"Training complete.  Logs saved to {out}/")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n── Final Stats ─────────────────────────────────")
    s = tracker.current_summary()
    for k, v in s.items():
        print(f"  {k}: {v}")

    # ── Eval progression ──────────────────────────────────────────────────
    if tracker.evals:
        print("\n── Evaluation Progression ──────────────────────")
        print(f"  {'Eval':>4s}  {'Episode':>7s}  {'Win%':>6s}  {'AvgTow':>6s}  {'AvgLeak':>7s}")
        for ev in tracker.evals:
            print(
                f"  {ev.eval_id:>4d}  {ev.episode:>7d}  "
                f"{ev.win_rate:>5.1%}  {ev.avg_towers_destroyed:>6.2f}  "
                f"{ev.avg_leaked_elixir:>7.2f}"
            )


if __name__ == "__main__":
    main()
