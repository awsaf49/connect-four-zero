import numpy as np
import pytest

from connect4_zero.game import ConnectFour


def test_gravity_and_turns() -> None:
    state = ConnectFour.from_moves([3, 3, 2])
    assert state.board[5, 3] == 1
    assert state.board[4, 3] == 2
    assert state.board[5, 2] == 1
    assert state.current_player == 2


@pytest.mark.parametrize(
    "moves,winner",
    [
        ([0, 0, 1, 1, 2, 2, 3], 1),
        ([0, 1, 0, 1, 2, 1, 2, 1], 2),
        ([0, 1, 1, 2, 5, 2, 2, 3, 5, 3, 6, 3, 3], 1),
    ],
)
def test_wins(moves: list[int], winner: int) -> None:
    state = ConnectFour.from_moves(moves)
    assert state.is_terminal
    assert state.winner == winner
    assert state.terminal_value() == -1.0


def test_illegal_column_and_terminal_rejected() -> None:
    full_column = ConnectFour.from_moves([0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        full_column.play(0)
    won = ConnectFour.from_moves([0, 0, 1, 1, 2, 2, 3])
    with pytest.raises(ValueError):
        won.play(4)


def test_encoding_mirror_and_color_swap() -> None:
    state = ConnectFour.from_moves([0, 1, 0])
    encoded = state.encode()
    assert encoded.shape == (3, 6, 7)
    assert np.array_equal(state.mirrored().mirrored().board, state.board)
    swapped = state.color_swapped()
    assert swapped.current_player == 1
    assert np.array_equal(swapped.color_swapped().board, state.board)
    assert state.mirrored().legal_actions == tuple(6 - action for action in reversed(state.legal_actions))


def test_bitboards_use_bottom_first_column_layout() -> None:
    state = ConnectFour.from_moves([0, 1, 0])
    player_one, player_two = state.bitboards()
    assert player_one == (1 << 0) | (1 << 1)
    assert player_two == 1 << 7

