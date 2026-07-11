import type {
  Board,
  Cell,
  DifficultyId,
  Move,
  Player,
  PositionAnalysis,
  SearchSnapshot,
} from "../types";

export const ROWS = 6;
export const COLS = 7;

export function emptyBoard(): Board {
  return Array.from({ length: ROWS }, () => Array<Cell>(COLS).fill(0));
}

export function cloneBoard(board: Board): Board {
  return board.map((row) => [...row]);
}

export function boardFromMoves(moves: number[]): Board {
  const board = emptyBoard();
  moves.forEach((column, index) => {
    const player: Player = index % 2 === 0 ? 1 : 2;
    const row = landingRow(board, column);
    if (row >= 0) {
      board[row][column] = player;
    }
  });
  return board;
}

export function gameStateFromMoves(moves: number[]) {
  const board = emptyBoard();
  const played: Move[] = [];
  let winner: Player | 0 = 0;

  moves.forEach((column, index) => {
    if (winner || !isColumnLegal(board, column)) return;
    const player: Player = index % 2 === 0 ? 1 : 2;
    const row = landingRow(board, column);
    board[row][column] = player;
    const move = { column, player, row } satisfies Move;
    played.push(move);
    if (hasWinner(board, player)) winner = player;
  });

  const nextPlayer: Player = played.length % 2 === 0 ? 1 : 2;
  return {
    board,
    currentPlayer: nextPlayer,
    winner,
    moves: played,
    lastMove: played.at(-1) ?? null,
  };
}

export function legalColumns(board: Board): number[] {
  return board[0].flatMap((cell, column) => (cell === 0 ? [column] : []));
}

export function isColumnLegal(board: Board, column: number): boolean {
  return column >= 0 && column < COLS && board[0][column] === 0;
}

export function landingRow(board: Board, column: number): number {
  for (let row = ROWS - 1; row >= 0; row -= 1) {
    if (board[row][column] === 0) return row;
  }
  return -1;
}

export function dropPiece(
  board: Board,
  column: number,
  player: Player,
): { board: Board; move: Move; winner: Player | 0 } | null {
  if (!isColumnLegal(board, column)) return null;
  const next = cloneBoard(board);
  const row = landingRow(next, column);
  next[row][column] = player;
  return {
    board: next,
    move: { column, player, row },
    winner: hasWinner(next, player) ? player : 0,
  };
}

export function isBoardFull(board: Board): boolean {
  return legalColumns(board).length === 0;
}

export function hasWinner(board: Board, player: Player): boolean {
  return winningCells(board, player).length > 0;
}

export function winningCells(board: Board, player: Player): Array<[number, number]> {
  const directions: Array<[number, number]> = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ];

  for (let row = 0; row < ROWS; row += 1) {
    for (let column = 0; column < COLS; column += 1) {
      if (board[row][column] !== player) continue;
      for (const [rowStep, columnStep] of directions) {
        const cells: Array<[number, number]> = [];
        for (let offset = 0; offset < 4; offset += 1) {
          const nextRow = row + rowStep * offset;
          const nextColumn = column + columnStep * offset;
          if (
            nextRow < 0 ||
            nextRow >= ROWS ||
            nextColumn < 0 ||
            nextColumn >= COLS ||
            board[nextRow][nextColumn] !== player
          ) {
            cells.length = 0;
            break;
          }
          cells.push([nextRow, nextColumn]);
        }
        if (cells.length === 4) return cells;
      }
    }
  }
  return [];
}

export function boardKey(board: Board, player: Player): string {
  return `${board.flat().join("")}:${player}`;
}

function hashString(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function unitNoise(seed: number): number {
  const value = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return value - Math.floor(value);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function immediateWinningColumns(board: Board, player: Player): number[] {
  return legalColumns(board).filter((column) => {
    const result = dropPiece(board, column, player);
    return result !== null && result.winner === player;
  });
}

function localThreatScore(board: Board, column: number, player: Player): number {
  const result = dropPiece(board, column, player);
  if (!result) return -1;
  const opponent: Player = player === 1 ? 2 : 1;
  const ownFollowUps = immediateWinningColumns(result.board, player).length;
  const opponentFollowUps = immediateWinningColumns(result.board, opponent).length;
  return ownFollowUps * 0.32 - opponentFollowUps * 0.22;
}

function normalized(values: number[], legal: number[]): number[] {
  const output = Array<number>(COLS).fill(0);
  const total = legal.reduce((sum, column) => sum + Math.max(0, values[column]), 0);
  if (!total) {
    legal.forEach((column) => {
      output[column] = 1 / legal.length;
    });
    return output;
  }
  legal.forEach((column) => {
    output[column] = Math.max(0, values[column]) / total;
  });
  return output;
}

function snapshotFor(
  analysis: Omit<PositionAnalysis, "snapshots">,
  simulation: number,
  seed: number,
): SearchSnapshot {
  const visits = Array<number>(COLS).fill(0);
  const legal = analysis.policy
    .map((value, column) => (value > 0 ? column : -1))
    .filter((column) => column >= 0);
  const raw = analysis.policy.map((probability, column) => {
    if (probability <= 0) return 0;
    const wobble = 1 + (unitNoise(seed + column * 19) - 0.5) * 0.24;
    return probability * wobble;
  });
  const policy = normalized(raw, legal);
  let assigned = 0;
  legal.forEach((column, index) => {
    const value = index === legal.length - 1 ? simulation - assigned : Math.round(policy[column] * simulation);
    visits[column] = Math.max(0, value);
    assigned += visits[column];
  });

  return {
    simulation,
    visits,
    qValues: analysis.qValues.map((value, column) => {
      if (!analysis.policy[column]) return 0;
      return clamp(value + (unitNoise(seed + column * 7) - 0.5) * 0.12, -1, 1);
    }),
    priors: analysis.priors.map((value) => value),
    rootValue: clamp(analysis.rootValue + (unitNoise(seed + 5) - 0.5) * 0.08, -1, 1),
  };
}

export function createAnalysis(
  board: Board,
  player: Player,
  difficulty: DifficultyId = "hard",
  simulations = 256,
): PositionAnalysis {
  const legal = legalColumns(board);
  const opponent: Player = player === 1 ? 2 : 1;
  const winning = immediateWinningColumns(board, player);
  const blocks = immediateWinningColumns(board, opponent);
  const key = boardKey(board, player);
  const seed = hashString(`${key}:${difficulty}:${simulations}`);
  const raw = Array<number>(COLS).fill(0);
  const qValues = Array<number>(COLS).fill(0);
  const priors = Array<number>(COLS).fill(0);
  const centerBias = [0.52, 0.74, 0.9, 1, 0.9, 0.74, 0.52];

  legal.forEach((column) => {
    const threat = localThreatScore(board, column, player);
    const center = centerBias[column];
    const immediate = winning.includes(column) ? 2.8 : blocks.includes(column) ? 1.7 : 0;
    const difficultyBoost = difficulty === "champion" ? 0.22 : difficulty === "beginner" ? -0.08 : 0;
    const noise = (unitNoise(seed + column * 31) - 0.5) * (difficulty === "beginner" ? 0.58 : difficulty === "intermediate" ? 0.24 : 0.1);
    raw[column] = Math.max(0.03, center + threat + immediate + difficultyBoost + noise);
    qValues[column] = clamp(
      (center - 0.62) * 0.5 + threat + immediate * 0.18 + noise * 0.35,
      -0.86,
      0.94,
    );
  });

  const policy = normalized(raw, legal);
  legal.forEach((column) => {
    priors[column] = clamp(policy[column] * (0.96 + unitNoise(seed + column * 11) * 0.08), 0, 1);
  });
  const priorTotal = priors.reduce((sum, value) => sum + value, 0);
  legal.forEach((column) => {
    priors[column] /= priorTotal || 1;
  });

  const visits = Array<number>(COLS).fill(0);
  let assigned = 0;
  legal.forEach((column, index) => {
    const value = index === legal.length - 1 ? simulations - assigned : Math.round(policy[column] * simulations);
    visits[column] = Math.max(0, value);
    assigned += visits[column];
  });
  if (legal.length && assigned !== simulations) {
    visits[legal[0]] += simulations - assigned;
  }

  const base = {
    policy,
    visits,
    qValues,
    priors,
    rootValue: clamp(qValues.reduce((sum, value) => sum + value, 0) / Math.max(1, legal.length), -1, 1),
    simulations,
    treeSize: 148 + Math.round(simulations * 0.72),
    principalVariations: principalVariations(board, player, policy),
  };
  const snapshotSteps = [...new Set([1, 2, 4, 8, 16, 32, 64, 128, simulations])].filter(
    (step) => step <= simulations,
  );
  const snapshots = snapshotSteps.map((step) => snapshotFor(base, step, seed + step * 43));
  return { ...base, snapshots };
}

function principalVariations(board: Board, player: Player, policy: number[]): number[][] {
  const ordered = legalColumns(board).sort((left, right) => policy[right] - policy[left]);
  return ordered.slice(0, 3).map((first, index) => {
    const line = [first];
    let current = dropPiece(board, first, player)?.board ?? board;
    let nextPlayer: Player = player === 1 ? 2 : 1;
    for (let depth = 0; depth < 4; depth += 1) {
      const nextAnalysis = createShallowPolicy(current, nextPlayer, index + depth + 1);
      const next = nextAnalysis[0];
      if (next < 0) break;
      line.push(next);
      const result = dropPiece(current, next, nextPlayer);
      if (!result) break;
      current = result.board;
      nextPlayer = nextPlayer === 1 ? 2 : 1;
    }
    return line;
  });
}

function createShallowPolicy(board: Board, player: Player, salt: number): [number, number[]] {
  const legal = legalColumns(board);
  if (!legal.length) return [-1, []];
  const opponent: Player = player === 1 ? 2 : 1;
  const winning = immediateWinningColumns(board, player);
  const blocks = immediateWinningColumns(board, opponent);
  const ranked = legal
    .map((column) => {
      const result = dropPiece(board, column, player);
      const threat = result ? localThreatScore(board, column, player) : -1;
      const score = (winning.includes(column) ? 100 : blocks.includes(column) ? 60 : 0) +
        (3 - Math.abs(3 - column)) * 2 + threat * 10 + unitNoise(salt * 17 + column) * 0.3;
      return { column, score };
    })
    .sort((left, right) => right.score - left.score);
  return [ranked[0].column, ranked.map((item) => item.column)];
}

export function chooseAiMove(
  board: Board,
  player: Player,
  difficulty: DifficultyId = "hard",
): number {
  const legal = legalColumns(board);
  if (!legal.length) return -1;
  const winning = immediateWinningColumns(board, player);
  if (winning.length) return winning[0];
  const opponent: Player = player === 1 ? 2 : 1;
  const blocks = immediateWinningColumns(board, opponent);
  if (blocks.length && difficulty !== "beginner") return blocks[0];
  const analysis = createAnalysis(board, player, difficulty, DIFFICULTY_SIMULATIONS[difficulty]);
  const ranked = legal.slice().sort((left, right) => analysis.policy[right] - analysis.policy[left]);
  if (difficulty === "beginner") return ranked[Math.min(1, ranked.length - 1)];
  if (difficulty === "intermediate" && ranked.length > 1 && analysis.policy[ranked[1]] > analysis.policy[ranked[0]] * 0.9) {
    return ranked[1];
  }
  return ranked[0];
}

const DIFFICULTY_SIMULATIONS: Record<DifficultyId, number> = {
  beginner: 32,
  intermediate: 96,
  hard: 256,
  champion: 800,
};

export function formatQ(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

export function analysisMetric(analysis: PositionAnalysis, mode: "policy" | "visits" | "q", column: number): number {
  if (mode === "policy") return analysis.policy[column] ?? 0;
  if (mode === "visits") return analysis.visits[column] ?? 0;
  return analysis.qValues[column] ?? 0;
}

export function coachInsight(analysis: PositionAnalysis, player: Player): { title: string; body: string; tag: string } {
  const best = analysis.policy.indexOf(Math.max(...analysis.policy));
  const centerDistance = Math.abs(3 - best);
  if (analysis.qValues[best] > 0.5) {
    return {
      title: `Column ${best + 1} keeps the initiative`,
      body: `${player === 1 ? "Your" : "The engine's"} leading line has a positive Q value and the widest visit margin. It is a good moment to play with purpose, not speed.`,
      tag: "positive position",
    };
  }
  if (centerDistance <= 1) {
    return {
      title: "The center is doing useful work",
      body: "The center columns keep more future connections available. A patient move here gives the search tree more room to breathe.",
      tag: "shape of the board",
    };
  }
  return {
    title: "Look for the quiet defense",
    body: "The top line is close to the next best move. Compare the Q values before committing; this is a position where one tempo matters.",
    tag: "close decision",
  };
}
