"""Pure-Python Connect Four bitboards used by the auditor.

The representation reserves seven bits per column.  Bits zero through five are
the six playable cells, with the bottom cell at the low bit; bit six is a
sentinel.  Keeping the sentinel makes the usual horizontal, vertical, and
diagonal alignment checks branch-free while preserving the layout used by the
rest of this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Sequence

WIDTH = 7
HEIGHT = 6
CONNECT = 4
STRIDE = HEIGHT + 1

PLAYABLE_COLUMN_MASK = (1 << HEIGHT) - 1
COLUMN_MASKS = tuple(PLAYABLE_COLUMN_MASK << (column * STRIDE) for column in range(WIDTH))
BOTTOM_MASKS = tuple(1 << (column * STRIDE) for column in range(WIDTH))
TOP_MASKS = tuple(1 << (column * STRIDE + HEIGHT - 1) for column in range(WIDTH))
BOARD_MASK = sum(COLUMN_MASKS)
BOTTOM_MASK = sum(BOTTOM_MASKS)
TOP_MASK = sum(TOP_MASKS)
CENTER_FIRST = (3, 2, 4, 1, 5, 0, 6)


def _validate_bits(bits: int, name: str = "bits") -> int:
    if not isinstance(bits, Integral):
        raise TypeError(f"{name} must be an int")
    bits = int(bits)
    if bits < 0 or bits & ~BOARD_MASK:
        raise ValueError(f"{name} contains bits outside the playable board")
    return bits


def mirror_bits(bits: int) -> int:
    """Return the horizontal reflection of a playable bitboard."""

    _validate_bits(bits)
    mirrored = 0
    for column in range(WIDTH):
        mirrored |= ((bits >> (column * STRIDE)) & PLAYABLE_COLUMN_MASK) << ((WIDTH - 1 - column) * STRIDE)
    return mirrored


def mirror_action(action: int) -> int:
    """Return the column corresponding to ``action`` in a mirrored board."""

    if not isinstance(action, int):
        raise TypeError("action must be an int")
    if not 0 <= action < WIDTH:
        raise ValueError(f"action must be between 0 and {WIDTH - 1}")
    return WIDTH - 1 - action


def has_alignment(bits: int) -> bool:
    """Return whether a bitboard contains four connected cells."""

    _validate_bits(bits)
    for shift in (1, STRIDE, HEIGHT, HEIGHT + 2):
        pair = bits & (bits >> shift)
        if pair & (pair >> (2 * shift)):
            return True
    return False


def _alignment_winner(player1: int, player2: int) -> int:
    player1_wins = has_alignment(player1)
    player2_wins = has_alignment(player2)
    if player1_wins and player2_wins:
        raise ValueError("a position cannot contain winning lines for both players")
    if player1_wins:
        return 1
    if player2_wins:
        return 2
    return 0


def _column_height(occupied: int, column: int) -> int:
    return ((occupied >> (column * STRIDE)) & PLAYABLE_COLUMN_MASK).bit_count()


@dataclass(frozen=True, slots=True)
class Position:
    """An immutable, training-independent Connect Four position.

    ``player1`` and ``player2`` contain only playable cells.  ``current_player``
    is the side to move, and ``winner`` is zero for a non-winning position.
    ``last_move`` is the most recently played column, when known.
    """

    player1: int = 0
    player2: int = 0
    current_player: int = 1
    winner: int = 0
    last_move: int | None = None

    def __post_init__(self) -> None:
        player1 = _validate_bits(self.player1, "player1")
        player2 = _validate_bits(self.player2, "player2")
        if player1 & player2:
            raise ValueError("player bitboards must not overlap")
        current_player = int(self.current_player)
        winner = int(self.winner)
        last_move = None if self.last_move is None else int(self.last_move)
        if current_player not in (1, 2):
            raise ValueError("current_player must be 1 or 2")
        if winner not in (0, 1, 2):
            raise ValueError("winner must be 0, 1, or 2")
        if last_move is not None and not 0 <= last_move < WIDTH:
            raise ValueError(f"last_move must be between 0 and {WIDTH - 1}")

        occupied = player1 | player2
        for column in range(WIDTH):
            column_bits = (occupied >> (column * STRIDE)) & PLAYABLE_COLUMN_MASK
            expected = (1 << column_bits.bit_count()) - 1
            if column_bits != expected:
                raise ValueError("bitboards must obey gravity in every column")

        detected_winner = _alignment_winner(player1, player2)
        if winner != detected_winner:
            raise ValueError(f"winner={winner} does not match the bitboards ({detected_winner})")
        object.__setattr__(self, "player1", player1)
        object.__setattr__(self, "player2", player2)
        object.__setattr__(self, "current_player", current_player)
        object.__setattr__(self, "winner", winner)
        object.__setattr__(self, "last_move", last_move)

    @classmethod
    def new(cls) -> Position:
        return cls()

    @classmethod
    def from_moves(cls, moves: Iterable[int]) -> Position:
        position = cls.new()
        for action in moves:
            position = position.play(int(action))
        return position

    @classmethod
    def from_bitboards(
        cls,
        player1: int,
        player2: int,
        *,
        current_player: int = 1,
        winner: int | None = None,
        last_move: int | None = None,
    ) -> Position:
        """Build a position and infer the winner unless one is supplied."""

        detected_winner = _alignment_winner(player1, player2)
        actual_winner = detected_winner if winner is None else winner
        return cls(player1, player2, current_player, actual_winner, last_move)

    @classmethod
    def from_board(
        cls,
        board: Sequence[Sequence[int]],
        *,
        current_player: int = 1,
        winner: int | None = None,
        last_move: int | tuple[int, int] | None = None,
    ) -> Position:
        """Build a position from six top-to-bottom rows of zero/one/two cells."""

        if len(board) != HEIGHT or any(len(row) != WIDTH for row in board):
            raise ValueError(f"board must have shape ({HEIGHT}, {WIDTH})")

        player_bits = [0, 0]
        for row_index, row in enumerate(board):
            for column, cell in enumerate(row):
                value = int(cell)
                if value not in (0, 1, 2):
                    raise ValueError("board cells must be 0, 1, or 2")
                if value:
                    bit = 1 << (column * STRIDE + HEIGHT - 1 - row_index)
                    player_bits[value - 1] |= bit

        action: int | None
        if isinstance(last_move, tuple):
            if len(last_move) != 2:
                raise ValueError("last_move tuple must contain row and column")
            action = int(last_move[1])
        else:
            action = last_move
        return cls.from_bitboards(
            player_bits[0],
            player_bits[1],
            current_player=current_player,
            winner=winner,
            last_move=action,
        )

    @classmethod
    def from_game(cls, game: object) -> Position:
        """Adapt a ConnectFour-like object without importing the training game."""

        board = getattr(game, "board")
        current_player = int(getattr(game, "current_player"))
        winner = int(getattr(game, "winner", 0))
        last_move = getattr(game, "last_move", None)
        return cls.from_board(
            board,
            current_player=current_player,
            winner=winner,
            last_move=last_move,
        )

    @property
    def p1(self) -> int:
        return self.player1

    @property
    def p2(self) -> int:
        return self.player2

    @property
    def occupied(self) -> int:
        return self.player1 | self.player2

    @property
    def move_count(self) -> int:
        return self.occupied.bit_count()

    @property
    def is_full(self) -> bool:
        return self.move_count == WIDTH * HEIGHT

    @property
    def is_terminal(self) -> bool:
        return self.winner != 0 or self.is_full

    @property
    def legal_actions(self) -> tuple[int, ...]:
        if self.is_terminal:
            return ()
        return tuple(column for column in range(WIDTH) if not self.occupied & TOP_MASKS[column])

    @property
    def legal_moves(self) -> tuple[int, ...]:
        return self.legal_actions

    def move_bit(self, action: int) -> int:
        if not isinstance(action, int):
            raise TypeError("action must be an int")
        if not 0 <= action < WIDTH:
            raise ValueError(f"illegal action: {action}")
        if self.occupied & TOP_MASKS[action]:
            raise ValueError(f"illegal action: {action}")
        height = _column_height(self.occupied, action)
        return 1 << (action * STRIDE + height)

    def play(self, action: int) -> Position:
        """Play a legal column and return the resulting immutable position."""

        if self.is_terminal:
            raise ValueError("cannot play from a terminal position")
        bit = self.move_bit(action)
        if self.current_player == 1:
            player1, player2 = self.player1 | bit, self.player2
        else:
            player1, player2 = self.player1, self.player2 | bit
        winner = self.current_player if has_alignment(player1 if self.current_player == 1 else player2) else 0
        return Position(player1, player2, 3 - self.current_player, winner, action)

    @property
    def board(self) -> tuple[tuple[int, ...], ...]:
        """Return rows in the same top-to-bottom order as ``game.ConnectFour``."""

        rows: list[list[int]] = [[0] * WIDTH for _ in range(HEIGHT)]
        for column in range(WIDTH):
            for height in range(HEIGHT):
                bit = 1 << (column * STRIDE + height)
                if self.player1 & bit:
                    rows[HEIGHT - 1 - height][column] = 1
                elif self.player2 & bit:
                    rows[HEIGHT - 1 - height][column] = 2
        return tuple(tuple(row) for row in rows)

    def as_list(self) -> list[list[int]]:
        return [list(row) for row in self.board]

    def bitboards(self) -> tuple[int, int]:
        return self.player1, self.player2

    def mirrored(self) -> Position:
        return Position(
            mirror_bits(self.player1),
            mirror_bits(self.player2),
            self.current_player,
            self.winner,
            None if self.last_move is None else mirror_action(self.last_move),
        )

    def mirror(self) -> Position:
        return self.mirrored()

    def key(self) -> tuple[int, int, int, int]:
        """Return a cache key that excludes the non-semantic last move."""

        return self.player1, self.player2, self.current_player, self.winner

    def mirrored_key(self) -> tuple[int, int, int, int]:
        return mirror_bits(self.player1), mirror_bits(self.player2), self.current_player, self.winner

    def canonical_key(self) -> tuple[int, int, int, int]:
        """Return the lexicographically smaller of this position and its mirror."""

        return min(self.key(), self.mirrored_key())

    def terminal_value(self, player: int | None = None) -> int:
        """Return an outcome from ``player``'s perspective for terminal states."""

        if not self.is_terminal:
            raise ValueError("terminal_value requires a terminal position")
        perspective = self.current_player if player is None else player
        if perspective not in (1, 2):
            raise ValueError("player must be 1 or 2")
        if self.winner == 0:
            return 0
        return 1 if self.winner == perspective else -1


BitboardPosition = Position


__all__ = [
    "BOARD_MASK",
    "BOTTOM_MASK",
    "BOTTOM_MASKS",
    "CENTER_FIRST",
    "COLUMN_MASKS",
    "CONNECT",
    "HEIGHT",
    "PLAYABLE_COLUMN_MASK",
    "Position",
    "STRIDE",
    "TOP_MASK",
    "TOP_MASKS",
    "WIDTH",
    "BitboardPosition",
    "has_alignment",
    "mirror_action",
    "mirror_bits",
]
