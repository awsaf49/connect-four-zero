import { ChevronDown } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { analysisMetric, formatQ, ROWS, COLS } from "../lib/game";
import { discDropDuration } from "../lib/sound";
import type { Board, Move, OverlayMode, Player, PositionAnalysis } from "../types";

interface ConnectFourBoardProps {
  board: Board;
  currentPlayer: Player;
  analysis?: PositionAnalysis;
  overlay?: OverlayMode;
  disabled?: boolean;
  lastMove?: Move | null;
  winningCells?: Array<[number, number]>;
  onColumnSelect?: (column: number) => void;
  compact?: boolean;
  label?: string;
}

function valueLabel(analysis: PositionAnalysis | undefined, overlay: OverlayMode, column: number): string {
  if (!analysis) return "--";
  if (overlay === "policy") return `${Math.round(analysisMetric(analysis, overlay, column) * 100)}%`;
  if (overlay === "visits") return `${Math.round(analysisMetric(analysis, overlay, column))}`;
  return formatQ(analysisMetric(analysis, overlay, column));
}

function valueTone(analysis: PositionAnalysis | undefined, overlay: OverlayMode, column: number): string {
  if (!analysis) return "neutral";
  const value = analysisMetric(analysis, overlay, column);
  if (overlay === "q") return value >= 0 ? "positive" : "negative";
  return value > (overlay === "policy" ? 0.18 : 40) ? "positive" : "neutral";
}

export function ConnectFourBoard({
  board,
  currentPlayer,
  analysis,
  overlay = "policy",
  disabled = false,
  lastMove,
  winningCells = [],
  onColumnSelect,
  compact = false,
  label = "Connect Four board",
}: ConnectFourBoardProps) {
  const [hoveredColumn, setHoveredColumn] = useState<number | null>(null);
  const moveKey = lastMove ? `${lastMove.row}-${lastMove.column}-${lastMove.player}` : null;
  const previousMoveKey = useRef(moveKey);
  const [droppingMoveKey, setDroppingMoveKey] = useState<string | null>(null);
  const winningSet = new Set(winningCells.map(([row, column]) => `${row}-${column}`));
  const currentCanPlay = !disabled && Boolean(onColumnSelect);

  useEffect(() => {
    if (!moveKey) {
      previousMoveKey.current = null;
      setDroppingMoveKey(null);
      return;
    }
    if (moveKey === previousMoveKey.current) return;
    previousMoveKey.current = moveKey;
    setDroppingMoveKey(moveKey);
    const timer = window.setTimeout(
      () => setDroppingMoveKey((current) => current === moveKey ? null : current),
      discDropDuration(lastMove?.row ?? 0) + 80,
    );
    return () => window.clearTimeout(timer);
  }, [lastMove?.row, moveKey]);

  return (
    <div className={`board-frame ${compact ? "board-frame-compact" : ""}`}>
      <div className="column-targets" aria-label="Choose a column">
        {Array.from({ length: COLS }, (_, column) => {
          const isHovered = hoveredColumn === column;
          return (
            <button
              className={`column-target ${isHovered ? "is-hovered" : ""} ${!currentCanPlay ? "is-disabled" : ""}`}
              disabled={!currentCanPlay}
              key={column}
              onClick={() => onColumnSelect?.(column)}
              onFocus={() => setHoveredColumn(column)}
              onMouseEnter={() => setHoveredColumn(column)}
              onMouseLeave={() => setHoveredColumn(null)}
              type="button"
              aria-label={`Play column ${column + 1}. ${valueLabel(analysis, overlay, column)} ${overlay}.`}
            >
              <span className="column-target-number">{column + 1}</span>
              <ChevronDown size={15} strokeWidth={2.2} aria-hidden="true" />
              <span className={`column-target-value tone-${valueTone(analysis, overlay, column)}`}>
                {valueLabel(analysis, overlay, column)}
              </span>
            </button>
          );
        })}
      </div>
      <div className="board-grid" role="grid" aria-label={label}>
        {board.map((row, rowIndex) =>
          row.map((cell, columnIndex) => {
            const isLastMove = lastMove?.row === rowIndex && lastMove.column === columnIndex;
            const isDropping = isLastMove && moveKey === droppingMoveKey;
            const isWinningCell = winningSet.has(`${rowIndex}-${columnIndex}`);
            return (
              <div
                className={`board-cell cell-${cell} ${isLastMove ? "is-last-move" : ""} ${isWinningCell ? "is-winning" : ""}`}
                key={`${rowIndex}-${columnIndex}`}
                role="gridcell"
                aria-label={`${cell === 0 ? "Empty" : cell === currentPlayer ? "Your piece" : "Engine piece"}, row ${ROWS - rowIndex}, column ${columnIndex + 1}`}
              >
                <span
                  className={`piece piece-${cell} ${isDropping && cell ? "piece-dropping" : ""}`}
                  style={isDropping && cell ? {
                    "--drop-rows": rowIndex + 1,
                    "--drop-duration": `${discDropDuration(rowIndex)}ms`,
                  } as CSSProperties : undefined}
                  aria-hidden="true"
                />
                {isLastMove ? <span className="last-move-marker" aria-hidden="true" /> : null}
              </div>
            );
          }),
        )}
      </div>
      <div className="board-underbar">
        <span className="board-player-key"><span className="key-dot key-human" /> You</span>
        <span className="board-player-key"><span className="key-dot key-engine" /> Engine</span>
        <span className="board-grid-note">{overlay === "q" ? "-1 to +1" : overlay === "visits" ? "simulations" : "root policy"}</span>
      </div>
    </div>
  );
}
