from collections.abc import Sequence

import numpy as np

from connect4_zero.game import ConnectFour
from connect4_zero.mcts import MCTS


class CenterPredictor:
    def predict(self, states: Sequence[ConnectFour]) -> tuple[np.ndarray, np.ndarray]:
        logits = np.tile(np.array([0, 1, 2, 4, 2, 1, 0], dtype=np.float32), (len(states), 1))
        return logits, np.zeros(len(states), dtype=np.float32)


def test_search_masks_illegal_moves_and_visits_tree() -> None:
    state = ConnectFour.from_moves([0, 0, 0, 0, 0, 0])
    result, _ = MCTS(CenterPredictor(), simulations=32, seed=1).search(state)
    assert result.visits[0] == 0
    assert result.visits.sum() == 32
    assert np.isclose(result.policy.sum(), 1.0)
    assert result.tree_size > 1


def test_search_finds_immediate_win_with_correct_backup_sign() -> None:
    state = ConnectFour.from_moves([0, 6, 1, 6, 2, 5])
    result, _ = MCTS(CenterPredictor(), simulations=80, seed=2).search(state)
    assert int(np.argmax(result.visits)) == 3
    assert result.q_values[3] > 0.9


def test_search_blocks_opponent_win_through_alternating_backup() -> None:
    state = ConnectFour.from_moves([0, 6, 1, 6, 2])
    result, _ = MCTS(CenterPredictor(), simulations=220, seed=12).search(state)
    assert int(np.argmax(result.visits)) == 3


def test_reused_subtree_keeps_prior_search_work() -> None:
    search = MCTS(CenterPredictor(), simulations=32, seed=14)
    state = ConnectFour.new()
    first, root = search.search(state)
    action = int(np.argmax(first.visits))
    reused = MCTS.advance_root(root, action)
    assert reused is not None
    prior_visits = sum(child.visits for child in reused.children.values())
    second, returned = search.search(state.play(action), root=reused)
    assert returned is reused
    assert int(second.visits.sum()) >= prior_visits + 32


def test_batched_search_uses_batched_predictor_calls() -> None:
    predictor = CenterPredictor()
    states = [ConnectFour.from_moves([column]) for column in range(4)]
    results, roots = MCTS(predictor, simulations=8, seed=3).search_many(states)
    assert len(results) == len(roots) == 4
    assert all(result.visits.sum() == 8 for result in results)


def test_noise_and_snapshots_are_seeded() -> None:
    state = ConnectFour.new()
    left, _ = MCTS(CenterPredictor(), simulations=16, seed=9).search(state, add_noise=True, record_snapshots=True)
    right, _ = MCTS(CenterPredictor(), simulations=16, seed=9).search(state, add_noise=True, record_snapshots=True)
    assert np.allclose(left.priors, right.priors)
    assert [item.simulation for item in left.snapshots] == [1, 2, 4, 8, 16]
