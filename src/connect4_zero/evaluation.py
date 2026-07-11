from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import numpy as np

from .game import ConnectFour
from .mcts import MCTS
from .network import PolicyValueEvaluator
from .auditor import NegamaxSolver, Position, SearchLimits


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


@dataclass(frozen=True, slots=True)
class SolverAccuracy:
    correct: int
    positions: int
    simulations: int
    seed: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.positions if self.positions else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "correct": self.correct,
            "positions": self.positions,
            "accuracy": self.accuracy,
            "simulations": self.simulations,
            "seed": self.seed,
        }


def exact_probe_suite(count: int = 40, seed: int = 41) -> list[tuple[ConnectFour, tuple[int, ...]]]:
    """Build a deterministic, discriminative late-game suite using only the auditor."""

    if count < 1:
        raise ValueError("at least one probe position is required")
    rng = random.Random(seed)
    solver = NegamaxSolver(limits=SearchLimits(max_nodes=500_000, max_seconds=2.0))
    probes: list[tuple[ConnectFour, tuple[int, ...]]] = []
    attempts = 0
    while len(probes) < count and attempts < count * 500:
        attempts += 1
        state = ConnectFour.new()
        target_plies = rng.randint(20, 26)
        while not state.is_terminal and state.move_count < target_plies:
            state = state.play(rng.choice(state.legal_actions))
        if state.is_terminal or len(state.legal_actions) < 2:
            continue
        if any(state.play(action).winner == state.current_player for action in state.legal_actions):
            continue
        grade = solver.grade_moves(Position.from_game(state))
        if not grade.complete or not grade.best_moves or len(grade.best_moves) == len(state.legal_actions):
            continue
        probes.append((state, grade.best_moves))
    if len(probes) != count:
        raise RuntimeError(f"only generated {len(probes)} exact probes after {attempts} attempts")
    return probes


def solver_move_accuracy(
    evaluator: PolicyValueEvaluator,
    positions: int = 40,
    simulations: int = 64,
    seed: int = 41,
) -> SolverAccuracy:
    probes = exact_probe_suite(positions, seed)
    search = MCTS(evaluator, simulations=simulations, seed=seed)
    results, _ = search.search_many([state for state, _ in probes])
    correct = sum(
        search.select_action(result, 0.0) in best_moves
        for result, (_, best_moves) in zip(results, probes)
    )
    return SolverAccuracy(correct=correct, positions=positions, simulations=simulations, seed=seed)


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
