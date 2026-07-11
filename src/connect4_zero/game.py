from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

ROWS = 6
COLS = 7
CONNECT = 4


@dataclass(frozen=True, slots=True)
class ConnectFour:
    """Immutable standard 7x6 Connect Four position."""

    board: np.ndarray
    current_player: int = 1
    winner: int = 0
    last_move: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        board = np.asarray(self.board, dtype=np.int8)
        if board.shape != (ROWS, COLS):
            raise ValueError(f"board must have shape {(ROWS, COLS)}")
        if self.current_player not in (1, 2):
            raise ValueError("current_player must be 1 or 2")
        if self.winner not in (0, 1, 2):
            raise ValueError("winner must be 0, 1, or 2")
        if np.any((board < 0) | (board > 2)):
            raise ValueError("board cells must be 0, 1, or 2")
        board = board.copy()
        board.setflags(write=False)
        object.__setattr__(self, "board", board)

    @classmethod
    def new(cls) -> ConnectFour:
        return cls(np.zeros((ROWS, COLS), dtype=np.int8))

    @classmethod
    def from_moves(cls, moves: Iterable[int]) -> ConnectFour:
        state = cls.new()
        for action in moves:
            state = state.play(action)
        return state

    @property
    def move_count(self) -> int:
        return int(np.count_nonzero(self.board))

    @property
    def is_terminal(self) -> bool:
        return self.winner != 0 or self.move_count == ROWS * COLS

    @property
    def legal_actions(self) -> tuple[int, ...]:
        if self.is_terminal:
            return ()
        return tuple(int(c) for c in np.flatnonzero(self.board[0] == 0))

    def play(self, action: int) -> ConnectFour:
        if self.is_terminal:
            raise ValueError("cannot play from a terminal position")
        if action < 0 or action >= COLS or self.board[0, action] != 0:
            raise ValueError(f"illegal action: {action}")

        empty_rows = np.flatnonzero(self.board[:, action] == 0)
        row = int(empty_rows[-1])
        board = self.board.copy()
        board[row, action] = self.current_player
        winner = self.current_player if _is_winning_move(board, row, action) else 0
        return ConnectFour(
            board=board,
            current_player=3 - self.current_player,
            winner=winner,
            last_move=(row, action),
        )

    def value_for(self, player: int) -> float:
        if not self.is_terminal or self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def terminal_value(self) -> float:
        """Terminal value from the player-to-move perspective."""
        if not self.is_terminal:
            raise ValueError("terminal_value requires a terminal position")
        return self.value_for(self.current_player)

    def encode(self) -> np.ndarray:
        """Raw AGZ-style planes: player one, player two, and side to move."""
        encoded = np.empty((3, ROWS, COLS), dtype=np.float32)
        encoded[0] = self.board == 1
        encoded[1] = self.board == 2
        encoded[2].fill(1.0 if self.current_player == 1 else 0.0)
        return encoded

    def mirrored(self) -> ConnectFour:
        last = None if self.last_move is None else (self.last_move[0], COLS - 1 - self.last_move[1])
        return ConnectFour(
            board=np.fliplr(self.board),
            current_player=self.current_player,
            winner=self.winner,
            last_move=last,
        )

    def color_swapped(self) -> ConnectFour:
        board = np.where(self.board == 1, 2, np.where(self.board == 2, 1, 0)).astype(np.int8)
        winner = 0 if self.winner == 0 else 3 - self.winner
        return ConnectFour(board, 3 - self.current_player, winner, self.last_move)

    def key(self) -> bytes:
        return self.board.tobytes() + bytes((self.current_player,))

    def bitboards(self) -> tuple[int, int]:
        """Return seven-bit-column bitboards, bottom cell at bit c*7."""
        bits = [0, 0]
        for row in range(ROWS):
            for col in range(COLS):
                piece = int(self.board[row, col])
                if piece:
                    bits[piece - 1] |= 1 << (col * 7 + (ROWS - 1 - row))
        return bits[0], bits[1]

    def as_list(self) -> list[list[int]]:
        return self.board.astype(int).tolist()


def _is_winning_move(board: np.ndarray, row: int, col: int) -> bool:
    player = int(board[row, col])
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= CONNECT:
            return True
    return False

