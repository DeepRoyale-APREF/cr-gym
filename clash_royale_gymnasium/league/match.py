"""Single match runner — executes one full game between two PlayerSlots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.core.state import State
from clash_royale_engine.players.player_interface import PlayerInterface, RLAgentPlayer
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


class _FrameSkipCountingAdapter(PlayerInterface):
    """Wraps a PlayerInterface with optional frame-skip and action counting.

    When ``frame_skip > 1``, the inner agent is only queried every
    *frame_skip* frames.  On intermediate frames the adapter returns
    ``None`` (noop), avoiding expensive NN forward passes per frame.
    """

    def __init__(self, inner: PlayerInterface, frame_skip: int = 1) -> None:
        self._inner = inner
        self._frame_skip = max(1, frame_skip)
        self._frame_count = 0
        self.action_count = 0

    def get_action(self, state: State) -> Optional[tuple]:
        self._frame_count += 1
        # Only query the real agent every `frame_skip` frames
        if self._frame_skip > 1 and (self._frame_count % self._frame_skip) != 1:
            return None
        action = self._inner.get_action(state)
        if action is not None:
            self.action_count += 1
        return action

    def reset(self) -> None:
        self._frame_count = 0
        self.action_count = 0
        self._inner.reset()


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
    frame_skip: int = 1,
) -> MatchResult:
    """Run a complete match and return the result.

    Both players act through their :class:`PlayerSlot` interface.
    The engine calls each player's ``get_action`` internally every frame;
    we do **not** call it separately to avoid double-calling (which would
    corrupt stateful agents like LSTM-based models).

    Parameters
    ----------
    frame_skip : int
        If > 1, agents are only queried every *frame_skip* frames.
        This dramatically speeds up matches involving neural-network
        agents while barely affecting gameplay quality.
    """
    player_0.reset()
    player_1.reset()

    adapter_0 = _FrameSkipCountingAdapter(
        player_0.to_player_interface(), frame_skip=frame_skip,
    )
    adapter_1 = _FrameSkipCountingAdapter(
        player_1.to_player_interface(), frame_skip=frame_skip,
    )

    engine = ClashRoyaleEngine(
        player1=adapter_0,
        player2=adapter_1,
        deck1=deck_0 or list(DEFAULT_DECK),
        deck2=deck_1 or list(DEFAULT_DECK),
        fps=fps,
        time_limit=time_limit,
        speed_multiplier=speed_multiplier,
        seed=seed,
    )

    # Let the engine drive both players — no explicit get_action calls
    # to avoid double-calling stateful agents (e.g. LSTM hidden state).
    while not engine.is_done():
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
        p0_actions=adapter_0.action_count,
        p1_actions=adapter_1.action_count,
        metadata={
            "p0": player_0.metadata(),
            "p1": player_1.metadata(),
            "seed": seed,
        },
    )
