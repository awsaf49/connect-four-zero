import numpy as np

from connect4_zero.evaluation import MatchResult, play_vs_random
from connect4_zero.game import ConnectFour


class FlatEvaluator:
    def predict(self, states: list[ConnectFour]) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((len(states), 7), dtype=np.float32), np.zeros(len(states), dtype=np.float32)


def test_match_result_score() -> None:
    result = MatchResult(wins=3, draws=2, losses=5)
    assert result.games == 10
    assert result.score == 0.4


def test_random_benchmark_completes_paired_games() -> None:
    result = play_vs_random(FlatEvaluator(), games=4, simulations=2, seed=4)  # type: ignore[arg-type]
    assert result.games == 4

