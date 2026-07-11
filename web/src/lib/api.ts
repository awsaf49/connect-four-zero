import type {
  ApiResult,
  Board,
  DifficultyId,
  EvolutionPoint,
  PositionAnalysis,
  ReplayPayload,
  RunSummary,
} from "../types";

const REQUEST_TIMEOUT_MS = 8_000;

export const DEMO_RUNS: RunSummary[] = [
  {
    id: "demo-bundled",
    name: "Five-minute self-trained champion",
    status: "complete",
    generation: 1,
    games: 516,
    winRate: 1,
    device: "Apple M5 Max / MPS",
    updatedAt: "2026-07-11T10:08:10.000Z",
    checkpoint: "demo-champion",
  },
];

const REAL_METRICS = [
  [0, 1.951, 0.856, 0.25, 1382],
  [0, 1.935, 0.777, 0.5, 2856],
  [0, 1.903, 0.671, 0.5, 4586],
  [0, 1.85, 0.577, 0.25, 6397],
  [1, 1.814, 0.551, 1, 8134],
  [1, 1.783, 0.513, 0.5, 9762],
  [1, 1.775, 0.497, 0.5, 11162],
] as const;

export const DEMO_EVOLUTION: EvolutionPoint[] = REAL_METRICS.map((item, index) => ({
  generation: index,
  policyLoss: item[1],
  valueLoss: item[2],
  arenaWinRate: item[3],
  elo: 1000 + item[0] * 80 + index * 8,
  games: item[4],
}));

export const DEMO_REPLAYS: ReplayPayload[] = [
  {
    id: "demo-game-1",
    runId: "demo-bundled",
    label: "generation 1 self-play",
    result: "win",
    winner: 1,
    moves: [3, 2, 3, 4, 2, 4, 1, 5, 1, 3, 0],
    simulations: 128,
    createdAt: "2026-07-11T10:08:00.000Z",
    opening: "Central pressure",
    notes: "A real representative line from the bundled zero-knowledge run.",
  },
];

interface BackendRun {
  id: string;
  name: string;
  status: "created" | "running" | "paused" | "stopping" | "completed" | "stopped" | "interrupted" | "failed";
  updated_at: string;
  generation: number;
  config: { simulations: number };
}

interface BackendMetric {
  step: number;
  wall_time: number;
  values: Record<string, number | string | boolean>;
}

interface BackendCheckpoint {
  id: string;
  kind: string;
  generation: number;
  training_step: number;
}

interface BackendSearchSnapshot {
  simulation: number;
  visits: number[];
  q_values: number[];
  priors: number[];
  root_value: number;
}

export interface BackendSearch {
  simulations: number;
  tree_size?: number;
  visits: number[];
  q_values: number[];
  priors: number[];
  root_value: number;
  principal_variations: number[][];
  snapshots: BackendSearchSnapshot[];
  solver?: {
    complete: boolean;
    outcome: string;
    best_moves: number[];
    nodes: number;
    aborted_reason: string | null;
    grades: Record<string, { outcome: string; value: number | null; exact: boolean }>;
  };
}

export interface BackendMoveTrace {
  action: number;
  player: 1 | 2;
  board_after: Board;
  solver_grade: string | null;
  search: BackendSearch | null;
}

interface BackendReplay {
  id: string;
  run_id: string;
  source: string;
  player_one: string;
  player_two: string;
  winner: 0 | 1 | 2;
  created_at: string;
  moves: BackendMoveTrace[];
  metadata: Record<string, unknown>;
}

export interface RemoteGame {
  id: string;
  board: Board;
  current_player: 1 | 2;
  human_player: 1 | 2;
  winner: 0 | 1 | 2;
  terminal: boolean;
  difficulty: string;
  model_label: string;
  human_trace: BackendMoveTrace | null;
  ai_trace: BackendMoveTrace | null;
  moves: BackendMoveTrace[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T | null> {
  if (typeof fetch !== "function") return null;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      signal: controller.signal,
    });
    if (!response.ok) return null;
    return (await response.json()) as T;
  } catch {
    return null;
  } finally {
    window.clearTimeout(timeout);
  }
}

function valueNumber(values: BackendMetric["values"], key: string, fallback = 0): number {
  const value = values[key];
  return typeof value === "number" ? value : fallback;
}

function runStatus(status: BackendRun["status"]): RunSummary["status"] {
  if (status === "completed") return "complete";
  if (status === "created" || status === "stopping") return "queued";
  if (status === "failed" || status === "stopped" || status === "interrupted") return "paused";
  return status;
}

export function analysisFromBackend(search: BackendSearch): PositionAnalysis {
  const total = search.visits.reduce((sum, value) => sum + value, 0);
  return {
    policy: search.visits.map((value) => (total ? value / total : 0)),
    visits: search.visits,
    qValues: search.q_values,
    priors: search.priors,
    rootValue: search.root_value,
    simulations: search.simulations,
    treeSize: search.tree_size ?? Math.max(search.simulations, 1),
    principalVariations: search.principal_variations,
    snapshots: search.snapshots.map((snapshot) => ({
      simulation: snapshot.simulation,
      visits: snapshot.visits,
      qValues: snapshot.q_values,
      priors: snapshot.priors,
      rootValue: snapshot.root_value,
    })),
  };
}

function replayFromBackend(game: BackendReplay, simulations: number): ReplayPayload {
  const result = game.winner === 0 ? "draw" : game.winner === 1 ? "win" : "loss";
  return {
    id: game.id,
    runId: game.run_id,
    label: `${game.source.replace("_", " ")} / ${game.id.slice(-6)}`,
    result,
    winner: game.winner,
    moves: game.moves.map((move) => move.action),
    simulations: game.moves.at(0)?.search?.simulations ?? simulations,
    createdAt: game.created_at,
    opening: game.moves.at(0)?.action === 3 ? "Center opening" : "Off-center opening",
    notes: `${game.player_one} versus ${game.player_two}. ${game.moves.length} recorded moves with search traces.`,
  };
}

export const apiClient = {
  async getRuns(): Promise<ApiResult<RunSummary[]>> {
    const remote = await request<BackendRun[]>("/api/runs");
    if (!remote?.length) return { data: DEMO_RUNS, source: "demo" };
    const summaries = await Promise.all(remote.map(async (run) => {
      const [metrics, checkpoints, games] = await Promise.all([
        request<BackendMetric[]>(`/api/runs/${run.id}/metrics?limit=1`),
        request<BackendCheckpoint[]>(`/api/runs/${run.id}/checkpoints`),
        request<BackendReplay[]>(`/api/runs/${run.id}/games?limit=200`),
      ]);
      const latest = metrics?.at(-1)?.values ?? {};
      const checkpoint = checkpoints?.at(-1)?.id ?? "initializing";
      return {
        id: run.id,
        name: run.name,
        status: runStatus(run.status),
        generation: run.generation,
        games: run.id === "demo-bundled" ? 516 : games?.length ?? 0,
        winRate: valueNumber(latest, "evaluation/score"),
        device: "Apple M5 Max / MPS",
        updatedAt: run.updated_at,
        checkpoint,
      } satisfies RunSummary;
    }));
    return { data: summaries, source: "api" };
  },

  async getEvolution(runId: string): Promise<ApiResult<EvolutionPoint[]>> {
    const metrics = await request<BackendMetric[]>(`/api/runs/${runId}/metrics`);
    if (!metrics?.length) return { data: DEMO_EVOLUTION, source: "demo" };
    const points = metrics.map((metric, index) => ({
      generation: index,
      policyLoss: valueNumber(metric.values, "loss/policy"),
      valueLoss: valueNumber(metric.values, "loss/value"),
      arenaWinRate: valueNumber(metric.values, "evaluation/score"),
      elo: 1000 + valueNumber(metric.values, "run/generation") * 80 + index * 8,
      games: valueNumber(metric.values, "run/replay_positions"),
    }));
    return { data: points, source: "api" };
  },

  async getReplays(): Promise<ApiResult<ReplayPayload[]>> {
    const runs = await request<BackendRun[]>("/api/runs");
    if (!runs?.length) return { data: DEMO_REPLAYS, source: "demo" };
    const groups = await Promise.all(runs.slice(0, 4).map(async (run) => {
      const games = await request<BackendReplay[]>(`/api/runs/${run.id}/games?limit=8`);
      return (games ?? []).map((game) => replayFromBackend(game, run.config.simulations));
    }));
    const replays = groups.flat();
    return { data: replays.length ? replays : DEMO_REPLAYS, source: replays.length ? "api" : "demo" };
  },

  async createGame(difficulty: DifficultyId): Promise<RemoteGame | null> {
    return request<RemoteGame>("/api/games", {
      method: "POST",
      body: JSON.stringify({ difficulty, human_player: 1 }),
    });
  },

  async move(gameId: string, action: number): Promise<RemoteGame | null> {
    return request<RemoteGame>(`/api/games/${gameId}/moves`, {
      method: "POST",
      body: JSON.stringify({ action }),
    });
  },

  async analyzeGame(gameId: string): Promise<(BackendSearch & { terminal?: boolean }) | null> {
    return request<BackendSearch & { terminal?: boolean }>(`/api/games/${gameId}/analysis`);
  },

  async searchPosition(board: Board, currentPlayer: 1 | 2, simulations: number): Promise<BackendSearch | null> {
    return request<BackendSearch>("/api/search", {
      method: "POST",
      body: JSON.stringify({ board, current_player: currentPlayer, simulations }),
    });
  },

  async createRun(minutes: number | null): Promise<BackendRun | null> {
    return request<BackendRun>("/api/runs", {
      method: "POST",
      body: JSON.stringify({ config: { minutes } }),
    });
  },

  async runAction(runId: string, action: "pause" | "resume" | "stop"): Promise<BackendRun | null> {
    return request<BackendRun>(`/api/runs/${runId}/action`, {
      method: "POST",
      body: JSON.stringify({ action }),
    });
  },

  async exportGame(gameId: string): Promise<{ mp4_url: string; json_url: string } | null> {
    return request(`/api/games/${gameId}/export`, { method: "POST" });
  },

  async exportReplay(runId: string, gameId: string): Promise<{ mp4_url: string; json_url: string } | null> {
    return request(`/api/runs/${runId}/games/${gameId}/export`, { method: "POST" });
  },

  subscribeRun(runId: string, onEvent: (event: Record<string, unknown>) => void): () => void {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws/runs/${runId}`);
    socket.onmessage = (message) => {
      try {
        onEvent(JSON.parse(message.data as string) as Record<string, unknown>);
      } catch {
        // Ignore malformed telemetry without disturbing the run.
      }
    };
    return () => socket.close();
  },
};
