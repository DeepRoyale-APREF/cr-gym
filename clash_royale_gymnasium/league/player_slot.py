"""Player slot — wraps any agent (heuristic, RL, or external) for the league."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from clash_royale_engine.core.state import State
from clash_royale_engine.players.player_interface import HeuristicBot, PlayerInterface


class PlayerSlot(ABC):
    """Abstract slot in a league tournament.

    Subclass this to plug any agent type (heuristic bot, trained RL model,
    scripted bot, etc.) into the tournament system.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"PPO-v3"``, ``"HeuristicBot-0.8"``).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        """Return an engine action ``(tile_x, tile_y, card_idx)`` or ``None``."""

    @abstractmethod
    def reset(self) -> None:
        """Called before each match."""

    def to_player_interface(self) -> PlayerInterface:
        """Wrap this slot as a :class:`PlayerInterface` for the engine."""
        return _SlotAdapter(self)

    def metadata(self) -> Dict[str, Any]:
        """Optional metadata for reporting (e.g. model version, ELO)."""
        return {"name": self.name}


class HeuristicSlot(PlayerSlot):
    """League slot wrapping a :class:`HeuristicBot`.

    Parameters
    ----------
    aggression : float
        Bot aggression level (0=passive, 1=aggressive).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "HeuristicBot",
        aggression: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__(name)
        self._bot = HeuristicBot(aggression=aggression, seed=seed)

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        return self._bot.get_action(state)

    def reset(self) -> None:
        self._bot.reset()

    def metadata(self) -> Dict[str, Any]:
        return {"name": self.name, "type": "heuristic", "aggression": self._bot.aggression}


class ExternalAgentSlot(PlayerSlot):
    """League slot for an external RL agent.

    The agent receives the raw :class:`State` and returns engine actions.
    This slot is model-agnostic — the model package provides the callback.

    Parameters
    ----------
    name : str
        Agent identifier.
    action_fn : callable
        ``(State) -> Optional[Tuple[int, int, int]]``
    """

    def __init__(
        self,
        name: str,
        action_fn: Any,  # Callable[[State], Optional[Tuple[int, int, int]]]
    ) -> None:
        super().__init__(name)
        self._action_fn = action_fn

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        return self._action_fn(state)

    def reset(self) -> None:
        pass


# ── Internal adapter ──────────────────────────────────────────────────────────


class _SlotAdapter(PlayerInterface):
    """Make a :class:`PlayerSlot` look like a :class:`PlayerInterface`."""

    def __init__(self, slot: PlayerSlot) -> None:
        self._slot = slot

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        return self._slot.get_action(state)

    def reset(self) -> None:
        self._slot.reset()
