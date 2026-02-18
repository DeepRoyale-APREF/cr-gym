"""Single match runner — executes one full game between two PlayerSlots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.players.player_interface import RLAgentPlayer
from clash_royale_engine.utils.constants import DEFAULT_DECK, DEFAULT_FPS, GAME_DURATION

from clash_royale_gymnasium.league.player_slot import PlayerSlot


@dataclass
class MatchResult:
    """Outcome of a single match between two players."""

    player_0_name: str
    player_1_name: str
    winner: Optional[int]  # 0, 1, or None (draw)
    total_frames: int = 0
    game_duration: float = 0.0  # seconds
    p0_towers_destroyed: int = 0
    p1_towers_destroyed: int = 0
    p0_leaked_elixir: float = 0.0
    p1_leaked_elixir: float = 0.0
    p0_actions: int = 0
    p1_actions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_draw(self) -> bool:
        return self.winner is None

    @property
    def winner_name(self) -> str:
        if self.winner == 0:
            return self.player_0_name
        elif self.winner == 1:
            return self.player_1_name
        return "Draw"


def run_match(
    player_0: PlayerSlot,
    player_1: PlayerSlot,
    *,
    deck_0: Optional[List[str]] = None,
    deck_1: Optional[List[str]] = None,
    fps: int = DEFAULT_FPS,
    time_limit: float = GAME_DURATION,
    speed_multiplier: float = 1.0,
    seed: int = 0,
) -> MatchResult:
    """Run a complete match and return the result.

    Both players act through their :class:`PlayerSlot` interface.
    """
    player_0.reset()
    player_1.reset()

    engine = ClashRoyaleEngine(
        player1=player_0.to_player_interface(),
        player2=player_1.to_player_interface(),
        deck1=deck_0 or list(DEFAULT_DECK),
        deck2=deck_1 or list(DEFAULT_DECK),
        fps=fps,
        time_limit=time_limit,
        speed_multiplier=speed_multiplier,
        seed=seed,
    )

    p0_actions = 0
    p1_actions = 0

    while not engine.is_done():
        s0_pre = engine.get_state(0)
        s1_pre = engine.get_state(1)

        a0 = player_0.get_action(s0_pre)
        a1 = player_1.get_action(s1_pre)

        if a0 is not None:
            p0_actions += 1
        if a1 is not None:
            p1_actions += 1

        # Use engine.step() which handles both players internally.
        # But since we already got actions, use step_with_action for p0,
        # and let p1 act through its interface.
        # Actually, since both are registered as player interfaces,
        # we can just call step().
        engine.step(frames=1)

    return MatchResult(
        player_0_name=player_0.name,
        player_1_name=player_1.name,
        winner=engine.get_winner(),
        total_frames=engine.current_frame,
        game_duration=engine.current_frame / fps,
        p0_towers_destroyed=engine.count_towers_destroyed(0),
        p1_towers_destroyed=engine.count_towers_destroyed(1),
        p0_leaked_elixir=engine.get_leaked_elixir(0),
        p1_leaked_elixir=engine.get_leaked_elixir(1),
        p0_actions=p0_actions,
        p1_actions=p1_actions,
        metadata={
            "p0": player_0.metadata(),
            "p1": player_1.metadata(),
            "seed": seed,
        },
    )
