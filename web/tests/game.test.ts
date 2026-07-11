import { describe, expect, it } from "vitest";
import {
  boardFromMoves,
  chooseAiMove,
  createAnalysis,
  dropPiece,
  emptyBoard,
  gameStateFromMoves,
  hasWinner,
  legalColumns,
  winningCells,
} from "../src/lib/game";

describe("Connect Four demo engine", () => {
  it("keeps gravity and alternating turns intact", () => {
    const state = gameStateFromMoves([3, 3, 2]);

    expect(state.board[5][3]).toBe(1);
    expect(state.board[4][3]).toBe(2);
    expect(state.board[5][2]).toBe(1);
    expect(state.currentPlayer).toBe(2);
    expect(state.lastMove).toMatchObject({ column: 2, row: 5, player: 1 });
  });

  it("finds horizontal, vertical, and diagonal wins", () => {
    const horizontal = boardFromMoves([0, 0, 1, 1, 2, 2, 3]);
    const vertical = boardFromMoves([0, 1, 0, 1, 0, 1, 2, 1]);
    const diagonal = boardFromMoves([0, 1, 1, 2, 5, 2, 2, 3, 5, 3, 6, 3, 3]);

    expect(hasWinner(horizontal, 1)).toBe(true);
    expect(hasWinner(vertical, 2)).toBe(true);
    expect(winningCells(horizontal, 1)).toEqual([[5, 0], [5, 1], [5, 2], [5, 3]]);
    expect(winningCells(vertical, 2)).toEqual([[2, 1], [3, 1], [4, 1], [5, 1]]);
    expect(winningCells(diagonal, 1)).toHaveLength(4);
  });

  it("rejects a full column without mutating the original board", () => {
    const board = emptyBoard();
    for (let turn = 0; turn < 6; turn += 1) {
      const result = dropPiece(board, 0, turn % 2 === 0 ? 1 : 2);
      expect(result).not.toBeNull();
      if (result) board.splice(0, board.length, ...result.board);
    }

    expect(legalColumns(board)).not.toContain(0);
    expect(dropPiece(board, 0, 1)).toBeNull();
  });

  it("returns a normalized search result with seeded snapshots", () => {
    const board = boardFromMoves([3, 2, 3, 4]);
    const analysis = createAnalysis(board, 1, "balanced", 256);

    expect(analysis.policy).toHaveLength(7);
    expect(analysis.visits.reduce((sum, value) => sum + value, 0)).toBe(256);
    expect(analysis.policy.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1);
    expect(analysis.snapshots.map((snapshot) => snapshot.simulation)).toEqual([1, 2, 4, 8, 16, 32, 64, 128, 256]);
    expect(createAnalysis(board, 1, "balanced", 256)).toEqual(analysis);
  });

  it("prioritizes a winning move and blocks an immediate threat", () => {
    const winningBoard = boardFromMoves([0, 6, 1, 6, 2, 5]);
    const blockingBoard = boardFromMoves([0, 6, 1, 6, 2]);

    expect(chooseAiMove(winningBoard, 1, "balanced")).toBe(3);
    expect(chooseAiMove(blockingBoard, 2, "balanced")).toBe(3);
  });
});
