#!/usr/bin/env python3
"""01 — Environment quickstart.

Demonstrates the basic Gymnasium loop:
  reset → step (random masked actions) → log episode stats.

Usage:
    python examples/01_env_quickstart.py
"""

from __future__ import annotations

import numpy as np

from clash_royale_gymnasium import ClashRoyaleGymEnv, TrainingTracker


def sample_masked_action(obs: dict) -> dict[str, int]:
    """Sample a random action respecting the action masks."""
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


def main() -> None:
    env = ClashRoyaleGymEnv(seed=42)
    tracker = TrainingTracker("quickstart", log_dir="./logs/quickstart")

    n_episodes = 5
    for ep in range(n_episodes):
        tracker.begin_episode()
        obs, info = env.reset(seed=ep)
        done = False

        while not done:
            action = sample_masked_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        rec = tracker.end_episode(env.episode_stats)
        print(
            f"Episode {rec.episode:>3d} | "
            f"Reward: {rec.total_reward:>7.2f} | "
            f"Actions: {rec.n_actions:>3d} | "
            f"Duration: {rec.game_duration:>5.1f}s | "
            f"Towers: {rec.towers_destroyed} | "
            f"Winner: {'Agent' if rec.winner == 0 else 'Opponent' if rec.winner == 1 else 'Draw'}"
        )

    env.close()

    # Print summary
    summary = tracker.current_summary()
    print("\n── Training Summary ─────────────────────────")
    print(f"  Episodes:              {summary['total_episodes']}")
    print(f"  Simulated time:        {summary['total_simulated_time_h']:.3f} h")
    print(f"  Wall-clock time:       {summary['total_wall_time_h']:.4f} h")
    print(f"  Sim/Wall ratio:        {summary['sim_to_wall_ratio']}x")
    print(f"  Episodes/hour:         {summary['episodes_per_hour']:.0f}")
    print(f"  Win rate:              {summary['rolling_win_rate']:.1%}")
    print(f"  Avg reward:            {summary['rolling_avg_reward']:.2f}")
    print(f"  Avg towers destroyed:  {summary['rolling_avg_towers']:.2f}")
    print(f"  Avg actions/episode:   {summary['rolling_avg_actions']:.1f}")
    print(f"  Avg leaked elixir:     {summary['rolling_avg_leaked']:.2f}")

    # Persist logs
    out = tracker.save()
    print(f"\nLogs saved to {out}/")


if __name__ == "__main__":
    main()
