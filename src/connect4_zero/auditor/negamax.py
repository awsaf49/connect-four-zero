"""Bounded exact Connect Four negamax with symmetry-aware caching.

This module intentionally imports only the standard library and
``auditor.bitboard``.  It is suitable for an audit endpoint that must be able
to inspect a position without loading a neural network or any training code.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
import time
from typing import Any

from .bitboard import CENTER_FIRST, Position, WIDTH

DEFAULT_NODE_LIMIT = 250_000
_UNSET = object()


class Outcome(str, Enum):
    """A game-theoretic outcome from the side-to-move perspective."""

    WIN = "win"
    DRAW = "draw"
    LOSS = "loss"
    UNKNOWN = "unknown"


MoveOutcome = Outcome
GameOutcome = Outcome


def outcome_for_value(value: int | None) -> Outcome:
    if value is None:
        return Outcome.UNKNOWN
    if value > 0:
        return Outcome.WIN
    if value < 0:
        return Outcome.LOSS
    return Outcome.DRAW


@dataclass(frozen=True, slots=True)
class SearchLimits:
    """Safety limits for one exact search.

    ``max_nodes`` defaults to a finite value so an accidental request to solve
    the empty board returns an explicit incomplete result instead of tying up
    an API worker indefinitely.  Set it to ``None`` when the caller owns the
    runtime budget.
    """

    max_nodes: int | None = DEFAULT_NODE_LIMIT
    max_seconds: float | None = None
    max_depth: int | None = None

    def __post_init__(self) -> None:
        if self.max_nodes is not None and (not isinstance(self.max_nodes, int) or self.max_nodes < 0):
            raise ValueError("max_nodes must be a non-negative integer or None")
        if self.max_seconds is not None and self.max_seconds < 0:
            raise ValueError("max_seconds must be non-negative or None")
        if self.max_depth is not None and (not isinstance(self.max_depth, int) or self.max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")

    @property
    def node_limit(self) -> int | None:
        return self.max_nodes

    @property
    def time_limit(self) -> float | None:
        return self.max_seconds

    @property
    def depth_limit(self) -> int | None:
        return self.max_depth


class Bound(str, Enum):
    EXACT = "exact"
    LOWER = "lower"
    UPPER = "upper"


@dataclass(frozen=True, slots=True)
class CacheEntry:
    value: int
    bound: Bound = Bound.EXACT


class TranspositionCache:
    """Small deterministic cache for scalar negamax results.

    Entries are keyed by the canonical position and, for depth-limited calls,
    the remaining depth.  Eviction is FIFO to keep repeated audits reproducible.
    """

    def __init__(self, max_entries: int | None = 200_000) -> None:
        if max_entries is not None and (not isinstance(max_entries, int) or max_entries < 1):
            raise ValueError("max_entries must be positive or None")
        self.max_entries = max_entries
        self._entries: dict[tuple[tuple[int, int, int, int], int | None], CacheEntry] = {}

    def get(self, key: tuple[tuple[int, int, int, int], int | None]) -> CacheEntry | None:
        return self._entries.get(key)

    def set(self, key: tuple[tuple[int, int, int, int], int | None], entry: CacheEntry) -> None:
        if key in self._entries:
            self._entries[key] = entry
            return
        if self.max_entries is not None and len(self._entries) >= self.max_entries:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = entry

    def clear(self) -> None:
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True, slots=True)
class MoveGrade:
    """The exact or currently-known value of one legal action."""

    action: int
    value: int | None
    outcome: Outcome
    exact: bool

    @property
    def label(self) -> str:
        return self.outcome.value

    @property
    def is_exact(self) -> bool:
        return self.exact


@dataclass(frozen=True, slots=True)
class GradeResult:
    """Move-by-move audit output."""

    position: Position
    grades: dict[int, MoveGrade]
    value: int | None
    outcome: Outcome
    complete: bool
    best_moves: tuple[int, ...]
    nodes: int
    cache_hits: int
    cache_size: int
    aborted_reason: str | None = None

    @property
    def exact(self) -> bool:
        return self.complete

    @property
    def root_value(self) -> int | None:
        return self.value

    @property
    def outcomes(self) -> dict[int, Outcome]:
        return {action: grade.outcome for action, grade in self.grades.items()}

    def __getitem__(self, action: int) -> MoveGrade:
        return self.grades[action]

    def __iter__(self):
        return iter(self.grades)

    def __len__(self) -> int:
        return len(self.grades)

    def items(self):
        return self.grades.items()


MoveGrades = GradeResult


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Scalar solver output with the best move when the search is complete."""

    position: Position
    value: int | None
    outcome: Outcome
    best_move: int | None
    best_moves: tuple[int, ...]
    complete: bool
    nodes: int
    cache_hits: int
    cache_size: int
    move_grades: dict[int, MoveGrade]
    aborted_reason: str | None = None

    @property
    def exact(self) -> bool:
        return self.complete

    @property
    def is_exact(self) -> bool:
        return self.complete

    @property
    def score(self) -> int | None:
        return self.value

    @property
    def grades(self) -> dict[int, MoveGrade]:
        return self.move_grades

    @property
    def root_value(self) -> int | None:
        return self.value


class _SearchLimit(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class NegamaxSolver:
    """Exact outcome solver with bounded execution and transposition caching.

    The solver optimizes the game-theoretic value ``-1, 0, +1``.  It never uses
    a heuristic at a depth boundary: if a limit is reached before a terminal
    proof, the affected move is reported as ``unknown`` and the result is not
    marked exact.
    """

    def __init__(
        self,
        limits: SearchLimits | None = None,
        *,
        max_nodes: int | None = DEFAULT_NODE_LIMIT,
        max_seconds: float | None = None,
        max_depth: int | None = None,
        node_limit: int | None | object = _UNSET,
        time_limit: float | None | object = _UNSET,
        depth_limit: int | None | object = _UNSET,
        cache: TranspositionCache | None = None,
        use_mirror: bool = True,
        move_order: Sequence[int] = CENTER_FIRST,
    ) -> None:
        if limits is not None:
            if any(value is not _UNSET for value in (node_limit, time_limit, depth_limit)):
                raise ValueError("use either limits or the *_limit aliases, not both")
            self.limits = limits
        else:
            if node_limit is not _UNSET:
                max_nodes = node_limit  # type: ignore[assignment]
            if time_limit is not _UNSET:
                max_seconds = time_limit  # type: ignore[assignment]
            if depth_limit is not _UNSET:
                max_depth = depth_limit  # type: ignore[assignment]
            self.limits = SearchLimits(max_nodes, max_seconds, max_depth)

        order = tuple(int(action) for action in move_order)
        if sorted(order) != list(range(WIDTH)):
            raise ValueError(f"move_order must contain each column 0 through {WIDTH - 1} exactly once")
        self.move_order = order
        self.cache = cache if cache is not None else TranspositionCache()
        self.use_mirror = bool(use_mirror)

        self._nodes = 0
        self._cache_hits = 0
        self._max_depth_reached = 0
        self._deadline: float | None = None
        self._active_limits = self.limits

    @property
    def cache_size(self) -> int:
        return self.cache.size

    def clear_cache(self) -> None:
        self.cache.clear()

    reset_cache = clear_cache

    def solve(
        self,
        position: Position | object,
        *,
        limits: SearchLimits | None = None,
        max_nodes: int | None | object = _UNSET,
        max_seconds: float | None | object = _UNSET,
        max_depth: int | None | object = _UNSET,
        node_limit: int | None | object = _UNSET,
        time_limit: float | None | object = _UNSET,
        depth_limit: int | None | object = _UNSET,
    ) -> SolveResult:
        """Solve ``position`` and include the move grades found in this run."""

        result = self.grade_moves(
            position,
            limits=limits,
            max_nodes=max_nodes,
            max_seconds=max_seconds,
            max_depth=max_depth,
            node_limit=node_limit,
            time_limit=time_limit,
            depth_limit=depth_limit,
        )
        best_move = result.best_moves[0] if result.complete and result.best_moves else None
        return SolveResult(
            position=result.position,
            value=result.value,
            outcome=result.outcome,
            best_move=best_move,
            best_moves=result.best_moves,
            complete=result.complete,
            nodes=result.nodes,
            cache_hits=result.cache_hits,
            cache_size=result.cache_size,
            move_grades=result.grades,
            aborted_reason=result.aborted_reason,
        )

    def grade_moves(
        self,
        position: Position | object,
        *,
        limits: SearchLimits | None = None,
        max_nodes: int | None | object = _UNSET,
        max_seconds: float | None | object = _UNSET,
        max_depth: int | None | object = _UNSET,
        node_limit: int | None | object = _UNSET,
        time_limit: float | None | object = _UNSET,
        depth_limit: int | None | object = _UNSET,
    ) -> GradeResult:
        position = _coerce_position(position)
        active_limits = self._resolve_limits(
            limits,
            max_nodes=max_nodes,
            max_seconds=max_seconds,
            max_depth=max_depth,
            node_limit=node_limit,
            time_limit=time_limit,
            depth_limit=depth_limit,
        )
        self._begin_search(active_limits)

        if position.is_terminal:
            self._nodes = 1
            value = position.terminal_value()
            return GradeResult(
                position=position,
                grades={},
                value=value,
                outcome=outcome_for_value(value),
                complete=True,
                best_moves=(),
                nodes=self._nodes,
                cache_hits=self._cache_hits,
                cache_size=self.cache_size,
            )

        grades: dict[int, MoveGrade] = {}
        reason: str | None = None
        legal_actions = position.legal_actions
        ordered_actions = self._ordered_actions(legal_actions)
        for index, action in enumerate(ordered_actions):
            try:
                child_value = self._search(position.play(action), -1, 1, 1)
            except _SearchLimit as error:
                reason = error.reason
                for remaining in ordered_actions[index:]:
                    grades[remaining] = MoveGrade(remaining, None, Outcome.UNKNOWN, False)
                break
            value = -child_value
            grades[action] = MoveGrade(action, value, outcome_for_value(value), True)

        grades = {action: grades[action] for action in legal_actions}
        complete = reason is None
        root_value: int | None = max((grade.value for grade in grades.values() if grade.value is not None), default=None)
        best_moves = (
            tuple(action for action in legal_actions if grades[action].value == root_value)
            if complete and root_value is not None
            else ()
        )
        if not complete:
            root_value = None
        if complete:
            self._store_root(position, root_value)
        return GradeResult(
            position=position,
            grades=grades,
            value=root_value,
            outcome=outcome_for_value(root_value),
            complete=complete,
            best_moves=best_moves,
            nodes=self._nodes,
            cache_hits=self._cache_hits,
            cache_size=self.cache_size,
            aborted_reason=reason,
        )

    audit = grade_moves

    def _resolve_limits(
        self,
        limits: SearchLimits | None,
        *,
        max_nodes: int | None | object,
        max_seconds: float | None | object,
        max_depth: int | None | object,
        node_limit: int | None | object,
        time_limit: float | None | object,
        depth_limit: int | None | object,
    ) -> SearchLimits:
        if limits is not None:
            if any(value is not _UNSET for value in (max_nodes, max_seconds, max_depth, node_limit, time_limit, depth_limit)):
                raise ValueError("use either limits or individual search-limit arguments, not both")
            return limits

        resolved_nodes = self.limits.max_nodes if max_nodes is _UNSET else max_nodes
        resolved_seconds = self.limits.max_seconds if max_seconds is _UNSET else max_seconds
        resolved_depth = self.limits.max_depth if max_depth is _UNSET else max_depth
        if node_limit is not _UNSET:
            resolved_nodes = node_limit
        if time_limit is not _UNSET:
            resolved_seconds = time_limit
        if depth_limit is not _UNSET:
            resolved_depth = depth_limit
        return SearchLimits(resolved_nodes, resolved_seconds, resolved_depth)  # type: ignore[arg-type]

    def _begin_search(self, limits: SearchLimits) -> None:
        self._nodes = 0
        self._cache_hits = 0
        self._max_depth_reached = 0
        self._active_limits = limits
        self._deadline = None if limits.max_seconds is None else time.monotonic() + limits.max_seconds

    def _cache_key(self, position: Position, depth: int) -> tuple[tuple[int, int, int, int], int | None]:
        state_key = position.canonical_key() if self.use_mirror else position.key()
        remaining_depth = None if self._active_limits.max_depth is None else self._active_limits.max_depth - depth
        return state_key, remaining_depth

    def _store_root(self, position: Position, value: int | None) -> None:
        if value is None:
            return
        self.cache.set(self._cache_key(position, 0), CacheEntry(value, Bound.EXACT))

    def _touch(self, depth: int, *, enforce_depth: bool = True) -> None:
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise _SearchLimit("time_limit")
        if self._active_limits.max_nodes is not None and self._nodes >= self._active_limits.max_nodes:
            raise _SearchLimit("node_limit")
        self._nodes += 1
        self._max_depth_reached = max(self._max_depth_reached, depth)
        if enforce_depth and self._active_limits.max_depth is not None and depth >= self._active_limits.max_depth:
            raise _SearchLimit("depth_limit")

    def _search(self, position: Position, alpha: int, beta: int, depth: int) -> int:
        key = self._cache_key(position, depth)
        original_alpha, original_beta = alpha, beta
        entry = self.cache.get(key)
        if entry is not None:
            self._cache_hits += 1
            if entry.bound is Bound.EXACT:
                return entry.value
            if entry.bound is Bound.LOWER:
                alpha = max(alpha, entry.value)
            else:
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

        if position.is_terminal:
            self._touch(depth, enforce_depth=False)
            value = position.terminal_value()
            self.cache.set(key, CacheEntry(value, Bound.EXACT))
            return value
        self._touch(depth)

        best = -1
        for action in self._ordered_actions(position.legal_actions):
            child_value = -self._search(position.play(action), -beta, -alpha, depth + 1)
            if child_value > best:
                best = child_value
            if child_value > alpha:
                alpha = child_value
            if alpha >= beta:
                break

        if best <= original_alpha:
            bound = Bound.UPPER
        elif best >= original_beta:
            bound = Bound.LOWER
        else:
            bound = Bound.EXACT
        self.cache.set(key, CacheEntry(best, bound))
        return best

    def _ordered_actions(self, legal_actions: Iterable[int]) -> tuple[int, ...]:
        legal = set(legal_actions)
        return tuple(action for action in self.move_order if action in legal)


class Auditor:
    """Convenience facade for a reusable solver in a future API layer."""

    def __init__(self, solver: NegamaxSolver | None = None, **solver_kwargs: Any) -> None:
        if solver is not None and solver_kwargs:
            raise ValueError("pass a solver or solver keyword arguments, not both")
        self.solver = solver if solver is not None else NegamaxSolver(**solver_kwargs)

    def audit(self, position: Position | object, **kwargs: Any) -> GradeResult:
        return self.solver.grade_moves(position, **kwargs)

    def solve(self, position: Position | object, **kwargs: Any) -> SolveResult:
        return self.solver.solve(position, **kwargs)

    def __call__(self, position: Position | object, **kwargs: Any) -> GradeResult:
        return self.audit(position, **kwargs)


def _coerce_position(position: Position | object) -> Position:
    return position if isinstance(position, Position) else Position.from_game(position)


def solve_position(position: Position | object, **solver_kwargs: Any) -> SolveResult:
    return NegamaxSolver(**solver_kwargs).solve(position)


def grade_moves(position: Position | object, **solver_kwargs: Any) -> GradeResult:
    return NegamaxSolver(**solver_kwargs).grade_moves(position)


audit_position = grade_moves


__all__ = [
    "Auditor",
    "Bound",
    "CacheEntry",
    "DEFAULT_NODE_LIMIT",
    "GameOutcome",
    "GradeResult",
    "MoveGrade",
    "MoveGrades",
    "MoveOutcome",
    "NegamaxSolver",
    "Outcome",
    "SearchLimits",
    "SolveResult",
    "TranspositionCache",
    "audit_position",
    "grade_moves",
    "outcome_for_value",
    "solve_position",
]
