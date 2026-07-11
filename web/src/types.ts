export type Cell = 0 | 1 | 2;
export type Player = 1 | 2;
export type Board = Cell[][];

export type ViewId = "play" | "lab" | "evolution" | "replay" | "runs";
export type OverlayMode = "policy" | "visits" | "q";
export type DifficultyId = "beginner" | "intermediate" | "hard" | "champion";
export type DataSource = "checking" | "api" | "demo";

export interface Move {
  column: number;
  player: Player;
  row: number;
}

export interface SearchSnapshot {
  simulation: number;
  visits: number[];
  qValues: number[];
  priors: number[];
  rootValue: number;
}

export interface PositionAnalysis {
  policy: number[];
  visits: number[];
  qValues: number[];
  priors: number[];
  rootValue: number;
  simulations: number;
  treeSize: number;
  principalVariations: number[][];
  snapshots: SearchSnapshot[];
}

export interface RunSummary {
  id: string;
  name: string;
  status: "running" | "complete" | "queued" | "paused" | "stopped" | "interrupted" | "failed";
  generation: number;
  games: number;
  winRate: number;
  device: string;
  updatedAt: string;
  checkpoint: string;
}

export interface EvolutionPoint {
  generation: number;
  policyLoss: number;
  valueLoss: number;
  arenaWinRate: number;
  elo: number;
  games: number;
}

export interface ReplaySummary {
  id: string;
  runId: string;
  label: string;
  result: "win" | "loss" | "draw";
  winner: Player | 0;
  moves: number[];
  simulations: number;
  createdAt: string;
  opening: string;
}

export interface ReplayPayload extends ReplaySummary {
  notes: string;
}

export interface ApiResult<T> {
  data: T;
  source: Exclude<DataSource, "checking">;
}

export interface GameState {
  board: Board;
  currentPlayer: Player;
  winner: Player | 0;
  moves: Move[];
  lastMove: Move | null;
}

export const DIFFICULTIES: Record<
  DifficultyId,
  { label: string; detail: string; simulations: number; temperature: number }
> = {
  beginner: {
    label: "Beginner",
    detail: "Leaves room to explore",
    simulations: 32,
    temperature: 0.55,
  },
  intermediate: {
    label: "Intermediate",
    detail: "An earlier model with more search",
    simulations: 96,
    temperature: 0.3,
  },
  hard: {
    label: "Hard",
    detail: "The latest model at full training search",
    simulations: 256,
    temperature: 0.12,
  },
  champion: {
    label: "Champion",
    detail: "The final checkpoint thinking deeply",
    simulations: 800,
    temperature: 0.08,
  },
};

export const OVERLAY_LABELS: Record<OverlayMode, string> = {
  policy: "Policy",
  visits: "Visits",
  q: "Q value",
};
