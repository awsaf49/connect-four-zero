"""Deterministic Connect Four replay export using Pillow and ffmpeg.

The exporter is deliberately independent of the neural-network and training
modules.  It accepts a move sequence, validates it through the auditor's pure
bitboard position, writes a stable JSON manifest, and encodes the same frames
to an MP4 without a shell.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable, Any

from .auditor import HEIGHT, Position, WIDTH

DEFAULT_FPS = 2
DEFAULT_WIDTH = 840
DEFAULT_HEIGHT = 720

BACKGROUND = (20, 25, 38)
BOARD_COLOR = (32, 92, 157)
BOARD_EDGE = (82, 147, 210)
EMPTY_COLOR = (13, 36, 63)
PLAYER_ONE_COLOR = (226, 68, 72)
PLAYER_TWO_COLOR = (246, 202, 70)
TEXT_COLOR = (235, 240, 248)
LAST_MOVE_EDGE = (255, 255, 255)


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    index: int
    action: int | None
    player: int | None
    current_player: int
    winner: int
    board: tuple[tuple[int, ...], ...]
    legal_actions: tuple[int, ...]
    bitboards: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "bitboards": [self.bitboards[0], self.bitboards[1]],
            "board": [list(row) for row in self.board],
            "current_player": self.current_player,
            "index": self.index,
            "legal_actions": list(self.legal_actions),
            "player": self.player,
            "winner": self.winner,
        }


@dataclass(frozen=True, slots=True)
class ReplayManifest:
    mp4_path: Path
    json_path: Path
    frame_count: int
    fps: int
    width: int
    height: int
    moves: tuple[int, ...]

    @property
    def video_path(self) -> Path:
        return self.mp4_path

    @property
    def manifest_path(self) -> Path:
        return self.json_path

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "fps": self.fps,
            "height": self.height,
            "json_path": str(self.json_path),
            "moves": list(self.moves),
            "mp4_path": str(self.mp4_path),
            "schema_version": 1,
            "width": self.width,
        }


ReplayExport = ReplayManifest


def build_replay_frames(moves: Iterable[int]) -> tuple[ReplayFrame, ...]:
    """Validate moves and return the initial frame plus one frame per move."""

    actions = tuple(int(action) for action in moves)
    position = Position.new()
    frames = [_make_frame(0, position, action=None, player=None)]
    for index, action in enumerate(actions, start=1):
        player = position.current_player
        position = position.play(action)
        frames.append(_make_frame(index, position, action=action, player=player))
    return tuple(frames)


def render_frame(
    position: Position,
    *,
    frame_index: int = 0,
    action: int | None = None,
    player: int | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> Any:
    """Render one deterministic RGB Pillow image for a position."""

    Image, ImageDraw, ImageFont = _pillow()
    _validate_dimensions(width, height)
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    header = f"CONNECT FOUR   FRAME {frame_index:03d}"
    if player is not None and action is not None:
        header += f"   P{player} -> C{action}"
    if position.winner:
        header += f"   P{position.winner} WINS"
    elif position.is_full:
        header += "   DRAW"
    draw.text((32, 28), header, fill=TEXT_COLOR, font=font)

    margin_x = max(24, width // 15)
    top = max(70, height // 9)
    bottom = max(24, height // 18)
    cell = min((width - 2 * margin_x) // WIDTH, (height - top - bottom) // HEIGHT)
    if cell < 12:
        raise ValueError("width and height are too small to render a board")
    board_width = cell * WIDTH
    board_height = cell * HEIGHT
    left = (width - board_width) // 2
    board_top = top + max(0, (height - top - bottom - board_height) // 2)
    board_bottom = board_top + board_height

    draw.rectangle((left - 10, board_top - 10, left + board_width + 10, board_bottom + 10), fill=BOARD_COLOR, outline=BOARD_EDGE, width=3)
    board = position.board
    for row in range(HEIGHT):
        for column in range(WIDTH):
            x0 = left + column * cell + cell // 8
            y0 = board_top + row * cell + cell // 8
            x1 = left + (column + 1) * cell - cell // 8
            y1 = board_top + (row + 1) * cell - cell // 8
            value = board[row][column]
            color = EMPTY_COLOR if value == 0 else PLAYER_ONE_COLOR if value == 1 else PLAYER_TWO_COLOR
            draw.ellipse((x0, y0, x1, y1), fill=color)

    if action is not None:
        last_row = _last_move_row(position, action)
        x0 = left + action * cell + cell // 8 - 3
        y0 = board_top + last_row * cell + cell // 8 - 3
        x1 = left + (action + 1) * cell - cell // 8 + 3
        y1 = board_top + (last_row + 1) * cell - cell // 8 + 3
        draw.ellipse((x0, y0, x1, y1), outline=LAST_MOVE_EDGE, width=3)
    return image


def export_replay(
    moves: Iterable[int],
    mp4_path: str | Path,
    json_path: str | Path | None = None,
    *,
    fps: int = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    ffmpeg_bin: str | Path | None = None,
    overwrite: bool = True,
) -> ReplayManifest:
    """Write a deterministic MP4 and JSON replay for ``moves``.

    The JSON path defaults to the MP4 path with a ``.json`` suffix.  Both
    outputs are replaced atomically only after rendering and encoding succeed.
    """

    _validate_fps(fps)
    _validate_dimensions(width, height)
    output_mp4 = Path(mp4_path).expanduser()
    output_json = Path(json_path).expanduser() if json_path is not None else output_mp4.with_suffix(".json")
    if output_mp4.resolve() == output_json.resolve():
        raise ValueError("MP4 and JSON outputs must be different files")
    if not overwrite and (output_mp4.exists() or output_json.exists()):
        raise FileExistsError("replay output already exists")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    actions = tuple(int(action) for action in moves)
    frames = build_replay_frames(actions)
    manifest = ReplayManifest(output_mp4, output_json, len(frames), fps, width, height, actions)
    payload = {
        "frames": [frame.to_dict() for frame in frames],
        "fps": fps,
        "height": height,
        "moves": list(actions),
        "schema_version": 1,
        "width": width,
    }

    ffmpeg = str(ffmpeg_bin) if ffmpeg_bin is not None else shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to export an MP4 replay")

    temporary_mp4: Path | None = None
    temporary_json: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="connect4-replay-") as frame_directory:
            frame_dir = Path(frame_directory)
            for frame in frames:
                image = render_frame(
                    Position.from_board(frame.board, current_player=frame.current_player, winner=frame.winner, last_move=frame.action),
                    frame_index=frame.index,
                    action=frame.action,
                    player=frame.player,
                    width=width,
                    height=height,
                )
                image.save(frame_dir / f"frame_{frame.index:06d}.png", format="PNG", optimize=False, compress_level=9)

            temporary_mp4 = _temporary_output(output_mp4, ".mp4.tmp")
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-fflags",
                "+bitexact",
                "-f",
                "image2",
                "-framerate",
                str(fps),
                "-i",
                str(frame_dir / "frame_%06d.png"),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                "-g",
                str(max(1, fps * 2)),
                "-keyint_min",
                str(max(1, fps * 2)),
                "-sc_threshold",
                "0",
                "-bf",
                "0",
                "-map_metadata",
                "-1",
                "-metadata",
                "creation_time=0",
                "-flags:v",
                "+bitexact",
                "-video_track_timescale",
                "1000",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                "-y",
                str(temporary_mp4),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                detail = completed.stderr.strip() or "ffmpeg returned a non-zero exit status"
                raise RuntimeError(detail)

        temporary_json = _temporary_output(output_json, ".json.tmp")
        with temporary_json.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_mp4, output_mp4)
        temporary_mp4 = None
        os.replace(temporary_json, output_json)
        temporary_json = None
    finally:
        if temporary_mp4 is not None:
            temporary_mp4.unlink(missing_ok=True)
        if temporary_json is not None:
            temporary_json.unlink(missing_ok=True)

    return manifest


def _make_frame(index: int, position: Position, *, action: int | None, player: int | None) -> ReplayFrame:
    return ReplayFrame(
        index=index,
        action=action,
        player=player,
        current_player=position.current_player,
        winner=position.winner,
        board=position.board,
        legal_actions=position.legal_actions,
        bitboards=position.bitboards(),
    )


def _last_move_row(position: Position, action: int) -> int:
    column_bits = (position.occupied >> (action * (HEIGHT + 1))) & ((1 << HEIGHT) - 1)
    height = column_bits.bit_length() - 1
    return HEIGHT - 1 - height


def _pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError("Pillow is required to render a replay") from error
    return Image, ImageDraw, ImageFont


def _temporary_output(target: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}-", suffix=suffix, dir=target.parent)
    os.close(descriptor)
    return Path(name)


def _validate_fps(fps: int) -> None:
    if not isinstance(fps, int) or fps < 1:
        raise ValueError("fps must be a positive integer")


def _validate_dimensions(width: int, height: int) -> None:
    if not isinstance(width, int) or not isinstance(height, int) or width < 2 or height < 2:
        raise ValueError("width and height must be positive integers")
    if width % 2 or height % 2:
        raise ValueError("width and height must be even for yuv420p output")


__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "ReplayExport",
    "ReplayFrame",
    "ReplayManifest",
    "build_replay_frames",
    "export_replay",
    "render_frame",
]
