import json
from pathlib import Path

from connect4_zero.auditor import NegamaxSolver, Outcome, Position, mirror_action
from connect4_zero.export import build_replay_frames, export_replay, render_frame


WINNING_POSITION_MOVES = [
    2,
    2,
    3,
    0,
    3,
    0,
    5,
    4,
    3,
    6,
    6,
    2,
    0,
    0,
    5,
    2,
    1,
    0,
    6,
    3,
    0,
    6,
    6,
    5,
    2,
    6,
    3,
    5,
    5,
    5,
    1,
    2,
    1,
    3,
]


def test_bitboard_layout_and_mirror_match_board_contract() -> None:
    position = Position.from_moves([0, 1, 0])
    assert position.bitboards() == ((1 << 0) | (1 << 1), 1 << 7)
    assert position.board[-1][:2] == (1, 2)
    assert position.mirrored().mirrored() == position
    assert position.mirrored().legal_actions == tuple(mirror_action(action) for action in reversed(position.legal_actions))


def test_terminal_position_is_exact_without_training_imports() -> None:
    position = Position.from_moves([0, 0, 1, 1, 2, 2, 3])
    result = NegamaxSolver(max_nodes=10).solve(position)
    assert result.complete
    assert result.outcome is Outcome.LOSS
    assert result.value == -1
    assert result.best_move is None


def test_exact_grading_and_mirror_cache_reuse() -> None:
    position = Position.from_moves(WINNING_POSITION_MOVES)
    solver = NegamaxSolver(max_nodes=1_000)
    result = solver.grade_moves(position)
    mirrored = solver.grade_moves(position.mirrored())

    assert result.complete
    assert result.outcome is Outcome.WIN
    assert result.best_moves == (1, 4)
    assert mirrored.complete
    assert mirrored.value == result.value
    assert {mirror_action(action): grade.value for action, grade in result.grades.items()} == {
        action: grade.value for action, grade in mirrored.grades.items()
    }
    assert mirrored.cache_hits > 0


def test_budgeted_audit_keeps_known_tactical_grade_and_marks_rest_unknown() -> None:
    position = Position.from_moves([0, 6, 1, 6, 2, 5])
    result = NegamaxSolver(max_nodes=1).grade_moves(position)

    assert not result.complete
    assert result.aborted_reason == "node_limit"
    assert result.grades[3].outcome is Outcome.WIN
    assert all(result.grades[action].outcome is Outcome.UNKNOWN for action in (0, 1, 2, 4, 5, 6))


def test_replay_export_is_json_and_byte_deterministic(tmp_path: Path) -> None:
    moves = [0, 1, 0, 1, 2, 2]
    first = export_replay(moves, tmp_path / "first.mp4", fps=3)
    second = export_replay(moves, tmp_path / "second.mp4", fps=3)

    assert first.frame_count == len(moves) + 1
    assert first.mp4_path.exists()
    assert first.json_path.exists()
    assert first.mp4_path.read_bytes() == second.mp4_path.read_bytes()
    assert first.json_path.read_bytes() == second.json_path.read_bytes()

    payload = json.loads(first.json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["moves"] == moves
    assert len(payload["frames"]) == len(moves) + 1
    assert payload["frames"][0]["action"] is None
    assert payload["frames"][-1]["action"] == 2
    assert build_replay_frames(moves)[-1].board == tuple(tuple(row) for row in payload["frames"][-1]["board"])
    assert render_frame(Position.new()).size == (840, 720)
