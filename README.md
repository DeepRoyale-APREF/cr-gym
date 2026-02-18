# Clash Royale Gymnasium

Gymnasium environment wrapping [clash-royale-engine](https://github.com/DeepRoyale-APREF/cr-engine) for AlphaStar-style reinforcement learning training of Clash Royale Arena 1 agents.

## Features

| Module | What it provides |
|---|---|
| **Environment** | `ClashRoyaleGymEnv` — Gymnasium `Dict` obs/action spaces, partial observations (fog-of-war), hierarchical action masking |
| **Rewards** | Composable `RewardFunction` with pluggable `RewardComponent`s (damage, elixir, terminal, strategy) |
| **Actions** | Hierarchical `Dict` space: strategy → card (masked) → tile_x (masked) → tile_y (masked) |
| **League** | Round-robin `LeagueTournament` with `PlayerSlot` abstraction for heuristic bots, RL agents, or any callable |
| **Tracker** | `TrainingTracker` — accumulates episode/eval stats, rolling metrics, CSV/JSON persistence |
| **Reporting** | PDF report generation with standings tables, win-rate charts, radar plots, and match logs |

## Installation

```bash
# Install cr-engine first (local sibling)
cd ../cr-engine && pip install -e .

# Install cr-gym
cd ../cr-gym && pip install -e .

# With PDF report support
pip install -e ".[report]"

# With dev tools (pytest, ruff, mypy, etc.)
pip install -e ".[dev]"
```

## Quick Start

```python
from clash_royale_gymnasium import ClashRoyaleGymEnv
import numpy as np

env = ClashRoyaleGymEnv(seed=42)
obs, info = env.reset()

done = False
while not done:
    # Sample a masked random action
    mask = obs["action_mask"]
    action = {
        "strategy": int(np.random.choice(np.where(mask["strategy"])[0])),
        "card":     int(np.random.choice(np.where(mask["card"])[0])),
        "tile_x":   int(np.random.choice(np.where(mask["tile_x"])[0])),
        "tile_y":   int(np.random.choice(np.where(mask["tile_y"])[0])),
    }
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(env.episode_stats)
env.close()
```

## Observation Space

All observations are **partial** — enemy elixir is hidden, only visible enemies are included, opponent hand is never exposed.

| Key | Shape | Description |
|---|---|---|
| `troops` | `(100, 14)` float32 | Entity features (padded, normalised) |
| `troop_mask` | `(100,)` bool | `True` where a real troop exists |
| `scalars` | `(16,)` float32 | Elixir, tower HP, time, flags (no enemy elixir) |
| `cards` | `(4, 4)` float32 | Own hand only |
| `action_mask` | `Dict` of bool arrays | Per-head validity masks |

## Action Space

Hierarchical `Dict` with sequential masking for AlphaStar-style policy heads:

| Key | Type | Description |
|---|---|---|
| `strategy` | `Discrete(3)` | AGGRESSIVE / DEFENSIVE / FARMING |
| `card` | `Discrete(5)` | Hand slot 0–3 or noop (4) |
| `tile_x` | `Discrete(18)` | Tile column (masked to valid placement) |
| `tile_y` | `Discrete(32)` | Tile row (masked to own side + pockets) |

## Reward System

Composable reward function with typed callbacks:

```python
from clash_royale_gymnasium import default_reward_function
from clash_royale_gymnasium.rewards.components import DamageComponent

# Use defaults
reward_fn = default_reward_function(damage_weight=1.5, elixir_weight=0.2)

# Or build custom
from clash_royale_gymnasium import RewardComponent, RewardFunction, RewardContext

class MyComponent(RewardComponent):
    def compute(self, ctx: RewardContext) -> float:
        return ctx.towers_destroyed_this_step * 5.0

reward_fn = RewardFunction([DamageComponent(weight=1.0), MyComponent(weight=2.0)])
env = ClashRoyaleGymEnv(reward_fn=reward_fn)
```

## League Training Loop

The primary workflow: train → evaluate periodically via league → track performance over time.

```python
from clash_royale_gymnasium import (
    ClashRoyaleGymEnv, ExternalAgentSlot, HeuristicSlot,
    LeagueTournament, TrainingTracker,
)

tracker = TrainingTracker("PPO-v1", log_dir="./logs")
env = ClashRoyaleGymEnv(seed=0)

for ep in range(1, 1001):
    tracker.begin_episode()
    obs, info = env.reset(seed=ep)
    done = False
    while not done:
        action = your_agent.act(obs)  # your model here
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    tracker.end_episode(env.episode_stats)

    # Evaluate every 100 episodes
    if ep % 100 == 0:
        tournament = LeagueTournament(
            players=[
                ExternalAgentSlot("PPO-v1", action_fn=your_agent.raw_act),
                HeuristicSlot("Passive", aggression=0.2),
                HeuristicSlot("Balanced", aggression=0.5),
                HeuristicSlot("Aggressive", aggression=0.8),
            ],
            matches_per_pair=10,
        )
        tournament.run()
        tracker.record_eval_from_tournament("PPO-v1", tournament.summary(), wall_time=0)
        print(tracker.current_summary())

tracker.save()
```

### Dashboard Output

`tracker.current_summary()` returns:

```
{
  "total_episodes": 1000,
  "total_matches": 1000,
  "total_simulated_time_h": 50.0,
  "total_wall_time_h": 0.42,
  "sim_to_wall_ratio": 119.0,
  "episodes_per_hour": 2380,
  "rolling_win_rate": 0.62,
  "rolling_avg_reward": 12.5,
  "rolling_avg_towers": 1.3,
  "rolling_avg_duration": 180.0,
  "rolling_avg_leaked": 0.8,
  "rolling_avg_actions": 22.0,
  "n_evals": 10,
  "latest_eval_win_rate": 0.65
}
```

## PDF Reports

```python
from clash_royale_gymnasium import HeuristicSlot, LeagueTournament
from clash_royale_gymnasium.reporting import generate_report

tournament = LeagueTournament(
    players=[HeuristicSlot("A", aggression=0.3), HeuristicSlot("B", aggression=0.7)],
    matches_per_pair=20,
)
tournament.run()
generate_report(tournament, output_path="report.pdf")
```

Requires `pip install clash-royale-gymnasium[report]`.

## Examples

| File | Description |
|---|---|
| `examples/01_env_quickstart.py` | Basic Gym loop with masked random actions and stats |
| `examples/02_league_training_loop.py` | Train → evaluate via league → dashboard + persistence |
| `examples/03_league_report.py` | Run a 5-player league and generate a PDF report |

## Project Structure

```
clash_royale_gymnasium/
├── env/              ClashRoyaleGymEnv
├── types/            Observation, HierarchicalAction, ActionMask, RewardContext
├── actions/          build_action_space(), compute_action_mask()
├── rewards/          RewardComponent, RewardFunction, built-in components
├── league/           PlayerSlot, HeuristicSlot, ExternalAgentSlot, LeagueTournament
├── reporting/        TrainingTracker, generate_report (PDF)
└── utils/            encode_observation (State → partial Observation)
```

## Testing

```bash
pytest -v               # 64 tests, ~15s
pytest -m "not slow"    # skip slow tests
```

## Requirements

- Python ≥ 3.10 (target 3.12)
- [clash-royale-engine](https://github.com/DeepRoyale-APREF/cr-engine) ≥ 0.1.0
- gymnasium ≥ 1.0
- numpy ≥ 1.26

## License

MIT
