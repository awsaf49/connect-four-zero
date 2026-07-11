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
    name: "One-hour self-trained champion",
    status: "complete",
    generation: 19,
    games: 1905,
    winRate: 1,
    device: "Apple M5 Max / MPS",
    updatedAt: "2026-07-11T10:08:10.000Z",
    checkpoint: "g019-s005600-final",
  },
];

const REAL_METRICS = [
  [1, 1.718, 0.3, 1, 1571], [2, 1.505, 0.249, 1, 3035], [3, 1.448, 0.245, 1, 4785],
  [4, 1.335, 0.282, 0.75, 8419], [5, 1.202, 0.293, 1, 14049], [6, 1.198, 0.308, 1, 16035],
  [7, 1.173, 0.32, 1, 20498], [8, 1.152, 0.304, 0.75, 24485], [9, 1.15, 0.322, 0.75, 26682],
  [10, 1.15, 0.326, 1, 28825], [11, 1.107, 0.294, 0.75, 31071], [12, 1.084, 0.288, 1, 33372],
  [13, 1.076, 0.291, 0.75, 35641], [14, 1.063, 0.291, 1, 39864], [15, 1.046, 0.301, 0.75, 43929],
  [16, 1.055, 0.305, 1, 48292], [17, 1.066, 0.335, 0.75, 52825], [18, 1.065, 0.345, 0.75, 55005],
  [19, 1.069, 0.35, 1, 57264],
] as const;

export const DEMO_EVOLUTION: EvolutionPoint[] = REAL_METRICS.map((item) => ({
  generation: item[0],
  policyLoss: item[1],
  valueLoss: item[2],
  arenaWinRate: item[3],
  elo: 1000 + item[0] * 24,
  games: item[4],
}));

export const DEMO_REPLAYS: ReplayPayload[] = [
  {
    id: "demo-game-1",
    runId: "demo-bundled",
    label: "generation 19 arena",
    result: "loss",
    winner: 2,
    moves: [3, 3, 3, 3, 3, 5, 5, 5, 1, 2, 5, 2, 1, 2, 2, 2, 3, 4, 4, 5, 4, 5, 2, 6, 6, 6, 6, 4, 4, 4, 0, 0, 0, 0, 0, 0, 1, 1],
    simulations: 256,
    createdAt: "2026-07-11T11:12:00.000Z",
    opening: "Center opening",
    notes: "Champion 18 versus the generation 19 candidate in the final paired arena.",
  },
];

interface BackendRun {
  id: string;
  name: string;
  status: "created" | "running" | "paused" | "stopping" | "completed" | "stopped" | "interrupted" | "failed";
  updated_at: string;
  generation: number;
  game_count?: number;
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
        games: run.game_count ?? (run.id === "demo-bundled" ? 516 : games?.length ?? 0),
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
    const samples = metrics.map((metric, index) => ({
      generation: valueNumber(metric.values, "run/generation", index),
      policyLoss: valueNumber(metric.values, "loss/policy"),
      valueLoss: valueNumber(metric.values, "loss/value"),
      arenaWinRate: valueNumber(metric.values, "evaluation/score"),
      elo: 1000 + valueNumber(metric.values, "run/generation") * 80 + index * 8,
      games: valueNumber(metric.values, "run/replay_positions"),
    }));
    const points = Array.from(
      samples.reduce((latest, sample) => latest.set(sample.generation, sample), new Map<number, EvolutionPoint>()).values(),
    );
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
