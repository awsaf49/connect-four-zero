from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .game import ConnectFour
from .mcts import MCTS
from .network import PolicyValueEvaluator


@dataclass(frozen=True, slots=True)
class MatchResult:
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "score": self.score,
        }


def play_vs_random(
    evaluator: PolicyValueEvaluator,
    games: int = 200,
    simulations: int = 64,
    seed: int = 31,
) -> MatchResult:
    if games < 2:
        raise ValueError("at least two games are required for paired colors")
    rng = np.random.default_rng(seed)
    search = MCTS(evaluator, simulations=simulations, seed=seed)
    states = [ConnectFour.new() for _ in range(games)]
    model_players = [1 if index % 2 == 0 else 2 for index in range(games)]
    active = list(range(games))

    while active:
        model_turns = [index for index in active if states[index].current_player == model_players[index]]
        random_turns = [index for index in active if index not in model_turns]
        if model_turns:
            results, _ = search.search_many([states[index] for index in model_turns])
            for index, result in zip(model_turns, results):
                states[index] = states[index].play(search.select_action(result, 0.0))
        for index in random_turns:
            action = int(rng.choice(states[index].legal_actions))
            states[index] = states[index].play(action)
        active = [index for index in active if not states[index].is_terminal]

    wins = sum(state.winner == model_players[index] for index, state in enumerate(states))
    draws = sum(state.winner == 0 for state in states)
    return MatchResult(wins=wins, draws=draws, losses=games - wins - draws)

