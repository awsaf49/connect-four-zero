import {
  Activity,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  Check,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  CircleHelp,
  Clock3,
  Command,
  Cpu,
  Database,
  Eye,
  FlaskConical,
  Gamepad2,
  GitBranch,
  History,
  Lightbulb,
  ListFilter,
  Pause,
  Play,
  RotateCcw,
  Search,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  Square,
  Target,
  Trophy,
  Volume2,
  VolumeX,
  Zap,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ConnectFourBoard } from "./components/ConnectFourBoard";
import {
  analysisMetric,
  boardFromMoves,
  chooseAiMove,
  coachInsight,
  createAnalysis,
  dropPiece,
  emptyBoard,
  gameStateFromMoves,
  isBoardFull,
  formatQ,
  legalColumns,
} from "./lib/game";
import { DEMO_EVOLUTION, DEMO_REPLAYS, DEMO_RUNS, analysisFromBackend, apiClient } from "./lib/api";
import type { BackendMoveTrace, BackendSearch, RemoteGame } from "./lib/api";
import { discDropDuration, playUiSound } from "./lib/sound";
import type {
  DataSource,
  DifficultyId,
  EvolutionPoint,
  GameState,
  Move,
  OverlayMode,
  Player,
  PositionAnalysis,
  ReplayPayload,
  RunSummary,
  ViewId,
} from "./types";
import type { ReactNode } from "react";
import { DIFFICULTIES, OVERLAY_LABELS } from "./types";
import "./styles.css";

interface NavItem {
  id: ViewId;
  label: string;
  hint: string;
  icon: LucideIcon;
}

const NAV_ITEMS: NavItem[] = [
  { id: "play", label: "Play", hint: "A live match", icon: Gamepad2 },
  { id: "lab", label: "Lab", hint: "Inspect search", icon: FlaskConical },
  { id: "evolution", label: "Evolution", hint: "Track training", icon: GitBranch },
  { id: "replay", label: "Replay", hint: "Review games", icon: History },
  { id: "runs", label: "Runs", hint: "Browse checkpoints", icon: ListFilter },
];

const LAB_MOVES = [3, 2, 3, 4, 2, 4, 1, 5, 1, 3, 0, 6, 0, 2, 4];

const VIEW_META: Record<ViewId, { eyebrow: string; title: string; description: string }> = {
  play: {
    eyebrow: "Play first",
    title: "Play against the search tree",
    description: "A focused board for feeling how policy, value, and search work together.",
  },
  lab: {
    eyebrow: "Search lab",
    title: "Open the position up",
    description: "Follow the root as simulations accumulate and the principal variation settles.",
  },
  evolution: {
    eyebrow: "Evolution",
    title: "Watch the agent improve",
    description: "A compact read on self-play, arena strength, and the current checkpoint.",
  },
  replay: {
    eyebrow: "Replay",
    title: "Review the decisive moments",
    description: "Scrub through recorded games and inspect the position at every move.",
  },
  runs: {
    eyebrow: "Runs",
    title: "Your local experiments",
    description: "Checkpoints, devices, and arena results in one quiet index.",
  },
};

function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(
    new Date(value),
  );
}

function columnName(column: number): string {
  return `C${column + 1}`;
}

function latest<T>(values: T[], fallback: T): T {
  return values.at(-1) ?? fallback;
}

function remoteMove(trace: BackendMoveTrace) {
  const row = trace.board_after.findIndex((boardRow) => boardRow[trace.action] !== 0);
  return { column: trace.action, player: trace.player, row: row < 0 ? 5 : row };
}

function playDiscImpact(move: Move, winner: Player | 0, muted: boolean) {
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
  const impactDelay = reduceMotion ? 0 : Math.round(discDropDuration(move.row) * 0.72);
  playUiSound("drop", muted, impactDelay);
  if (winner) playUiSound("win", muted, impactDelay + 90);
}

function gameStateFromRemote(game: RemoteGame): GameState {
  const moves = game.moves.map(remoteMove);
  return {
    board: game.board,
    currentPlayer: game.current_player,
    winner: game.winner,
    moves,
    lastMove: moves.at(-1) ?? null,
  };
}

function snapshotAnalysis(search: BackendSearch, snapshotIndex: number): PositionAnalysis {
  const snapshot = search.snapshots[snapshotIndex];
  if (!snapshot) return analysisFromBackend(search);
  return analysisFromBackend({
    ...search,
    simulations: snapshot.simulation,
    visits: snapshot.visits,
    q_values: snapshot.q_values,
    priors: snapshot.priors,
    root_value: snapshot.root_value,
    snapshots: search.snapshots.slice(0, snapshotIndex + 1),
  });
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function analysisAtSnapshot(base: PositionAnalysis, snapshotIndex: number): PositionAnalysis {
  const snapshot = base.snapshots[snapshotIndex] ?? latest(base.snapshots, {
    simulation: base.simulations,
    visits: base.visits,
    qValues: base.qValues,
    priors: base.priors,
    rootValue: base.rootValue,
  });
  const total = snapshot.visits.reduce((sum, value) => sum + value, 0);
  return {
    ...base,
    visits: snapshot.visits,
    policy: snapshot.visits.map((value) => (total ? value / total : 0)),
    qValues: snapshot.qValues,
    priors: snapshot.priors,
    rootValue: snapshot.rootValue,
    simulations: snapshot.simulation,
  };
}

function ViewHeading({ view, children }: { view: ViewId; children?: ReactNode }) {
  const meta = VIEW_META[view];
  return (
    <div className="view-heading">
      <div>
        <p className="eyebrow">{meta.eyebrow}</p>
        <h1>{meta.title}</h1>
        <p className="view-description">{meta.description}</p>
      </div>
      {children ? <div className="view-heading-actions">{children}</div> : null}
    </div>
  );
}

function SourceBadge({ source }: { source: DataSource }) {
  const isApi = source === "api";
  const isChecking = source === "checking";
  return (
    <span className={`source-badge source-${source}`}>
      <span className="source-dot" />
      {isChecking ? "Checking local API" : isApi ? "API connected" : "Demo engine"}
    </span>
  );
}

function OverlaySelector({ value, onChange }: { value: OverlayMode; onChange: (value: OverlayMode) => void }) {
  return (
    <div className="segmented-control" aria-label="Analysis overlay">
      {(Object.keys(OVERLAY_LABELS) as OverlayMode[]).map((mode) => (
        <button
          className={value === mode ? "is-active" : ""}
          key={mode}
          onClick={() => onChange(mode)}
          type="button"
        >
          {OVERLAY_LABELS[mode]}
        </button>
      ))}
    </div>
  );
}

function Switch({ checked, label, onChange }: { checked: boolean; label: string; onChange: (value: boolean) => void }) {
  return (
    <button
      aria-checked={checked}
      aria-label={label}
      className={`switch ${checked ? "is-on" : ""}`}
      onClick={() => onChange(!checked)}
      role="switch"
      type="button"
    >
      <span className="switch-thumb" />
    </button>
  );
}

function StatCard({ icon: Icon, label, value, detail, tone = "blue" }: { icon: LucideIcon; label: string; value: string; detail: string; tone?: string }) {
  return (
    <div className={`stat-card tone-${tone}`}>
      <div className="stat-icon"><Icon size={16} strokeWidth={2} /></div>
      <span className="stat-label">{label}</span>
      <strong>{value}</strong>
      <span className="stat-detail">{detail}</span>
    </div>
  );
}

function MetricBars({ analysis, mode, activeColumn }: { analysis: PositionAnalysis; mode: OverlayMode; activeColumn?: number }) {
  const maximum = Math.max(...analysis.policy.map((_, column) => Math.abs(analysisMetric(analysis, mode, column))), 1);
  return (
    <div className="metric-bars">
      {Array.from({ length: 7 }, (_, column) => {
        const value = analysisMetric(analysis, mode, column);
        const width = `${Math.max(5, Math.round((Math.abs(value) / maximum) * 100))}%`;
        return (
          <div className={`metric-row ${activeColumn === column ? "is-active" : ""}`} key={column}>
            <span className="metric-row-name">{columnName(column)}</span>
            <span className="metric-track"><span className={`metric-fill metric-${mode}`} style={{ width }} /></span>
            <span className={`metric-row-value metric-value-${value >= 0 ? "positive" : "negative"}`}>
              {mode === "policy" ? formatPercent(value) : mode === "visits" ? Math.round(value) : formatQ(value)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function RunStatus({ status }: { status: RunSummary["status"] }) {
  const labels: Record<RunSummary["status"], string> = {
    running: "Running",
    complete: "Complete",
    queued: "Queued",
    paused: "Paused",
    stopped: "Stopped",
    interrupted: "Interrupted",
    failed: "Failed",
  };
  return <span className={`run-status status-${status}`}><span className="status-dot" />{labels[status]}</span>;
}

function App() {
  const [view, setView] = useState<ViewId>("play");
  const [difficulty, setDifficulty] = useState<DifficultyId>("hard");
  const [overlay, setOverlay] = useState<OverlayMode>("policy");
  const [coach, setCoach] = useState(true);
  const [muted, setMuted] = useState(false);
  const [source, setSource] = useState<DataSource>("checking");
  const [runs, setRuns] = useState<RunSummary[]>(DEMO_RUNS);
  const [evolution, setEvolution] = useState<EvolutionPoint[]>(DEMO_EVOLUTION);
  const [replays, setReplays] = useState<ReplayPayload[]>(DEMO_REPLAYS);
  const [selectedRunId, setSelectedRunId] = useState("demo-bundled");
  const [selectedReplayId, setSelectedReplayId] = useState(DEMO_REPLAYS[0].id);

  const initialPlay = useMemo(() => gameStateFromMoves([]), []);
  const [playState, setPlayState] = useState<GameState>(initialPlay);
  const [playAnalysis, setPlayAnalysis] = useState<PositionAnalysis>(() => createAnalysis(emptyBoard(), 1, "hard", 256));
  const [thinking, setThinking] = useState(false);
  const [remoteGameId, setRemoteGameId] = useState<string | null>(null);
  const [modelLabel, setModelLabel] = useState("demo-champion");
  const [trainingMinutes, setTrainingMinutes] = useState<number | null>(60);
  const [launchingRun, setLaunchingRun] = useState(false);

  const [labState, setLabState] = useState<GameState>(() => gameStateFromMoves(LAB_MOVES));
  const [labSimulations, setLabSimulations] = useState(256);
  const [labTemperature, setLabTemperature] = useState(0.3);
  const [labNoise, setLabNoise] = useState(true);
  const [labSnapshotIndex, setLabSnapshotIndex] = useState(8);
  const [labRemoteAnalysis, setLabRemoteAnalysis] = useState<PositionAnalysis | null>(null);

  const [replayCursor, setReplayCursor] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [exportingReplay, setExportingReplay] = useState(false);
  const [exportedReplayUrl, setExportedReplayUrl] = useState<string | null>(null);

  const selectedReplay = replays.find((replay) => replay.id === selectedReplayId) ?? replays[0];
  const replayState = useMemo(
    () => gameStateFromMoves(selectedReplay?.moves.slice(0, replayCursor) ?? []),
    [replayCursor, selectedReplay],
  );
  const replayAnalysis = useMemo(
    () => createAnalysis(replayState.board, replayState.currentPlayer, "hard", selectedReplay?.simulations ?? 256),
    [replayState.board, replayState.currentPlayer, selectedReplay],
  );
  const localLabAnalysis = useMemo(
    () => createAnalysis(labState.board, labState.currentPlayer, difficulty, labSimulations),
    [difficulty, labSimulations, labState.board, labState.currentPlayer],
  );
  const labBaseAnalysis = labRemoteAnalysis ?? localLabAnalysis;
  const labAnalysis = useMemo(
    () => analysisAtSnapshot(labBaseAnalysis, Math.min(labSnapshotIndex, Math.max(0, labBaseAnalysis.snapshots.length - 1))),
    [labBaseAnalysis, labSnapshotIndex],
  );

  useEffect(() => {
    let active = true;
    void apiClient.getRuns().then(async (runResult) => {
      const firstRun = runResult.data.find((run) => run.status === "running")?.id ?? runResult.data[0]?.id ?? "demo-bundled";
      const [evolutionResult, replayResult] = await Promise.all([apiClient.getEvolution(firstRun), apiClient.getReplays()]);
        if (!active) return;
        setRuns(runResult.data);
        setEvolution(evolutionResult.data);
        setReplays(replayResult.data);
        setSelectedRunId(firstRun);
        setSelectedReplayId(replayResult.data[0]?.id ?? DEMO_REPLAYS[0].id);
        const connected = runResult.source === "api";
        setSource(connected ? "api" : "demo");
        if (connected) await startRemoteGame("hard");
      });
    return () => {
      active = false;
    };
  }, []); // The local API is discovered once; explicit refresh controls update later.

  useEffect(() => {
    if (source !== "api") {
      setLabRemoteAnalysis(null);
      return;
    }
    let active = true;
    const timer = window.setTimeout(() => {
      void apiClient.searchPosition(labState.board, labState.currentPlayer, labSimulations).then((search) => {
        if (!active || !search || "terminal" in search) return;
        setLabRemoteAnalysis(analysisFromBackend(search));
        setLabSnapshotIndex(0);
      });
    }, 180);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [labSimulations, labState.board, labState.currentPlayer, source]);

  useEffect(() => {
    const selected = runs.find((run) => run.id === selectedRunId);
    if (source !== "api" || !selected || selected.status !== "running") return;
    return apiClient.subscribeRun(selected.id, (event) => {
      if (event.type === "heartbeat" || event.type === "self_play") return;
      void Promise.all([apiClient.getRuns(), apiClient.getEvolution(selected.id)]).then(([runResult, evolutionResult]) => {
        setRuns(runResult.data);
        setEvolution(evolutionResult.data);
      });
    });
  }, [runs, selectedRunId, source]);

  useEffect(() => {
    if (source === "api" || playState.currentPlayer !== 2 || playState.winner || isBoardFull(playState.board)) {
      setThinking(false);
      return;
    }
    setThinking(true);
    const timer = window.setTimeout(() => {
      const column = chooseAiMove(playState.board, 2, difficulty);
      const result = dropPiece(playState.board, column, 2);
      if (!result) {
        setThinking(false);
        return;
      }
      playDiscImpact(result.move, result.winner, muted);
      setPlayState((current) => ({
        ...current,
        board: result.board,
        currentPlayer: 1,
        winner: result.winner,
        moves: [...current.moves, result.move],
        lastMove: result.move,
      }));
      setPlayAnalysis(createAnalysis(result.board, 1, difficulty, DIFFICULTIES[difficulty].simulations));
      setThinking(false);
    }, difficulty === "beginner" ? 360 : difficulty === "intermediate" ? 520 : 700);
    return () => window.clearTimeout(timer);
  }, [difficulty, muted, playState.board, playState.currentPlayer, playState.winner, source]);

  useEffect(() => {
    if (!replayPlaying || !selectedReplay) return;
    if (replayCursor >= selectedReplay.moves.length) {
      setReplayPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => setReplayCursor((cursor) => cursor + 1), 700);
    return () => window.clearTimeout(timer);
  }, [replayCursor, replayPlaying, selectedReplay]);

  function navigate(nextView: ViewId) {
    setView(nextView);
  }

  async function startRemoteGame(nextDifficulty: DifficultyId) {
    setThinking(true);
    const game = await apiClient.createGame(nextDifficulty);
    if (!game) {
      setSource("demo");
      setRemoteGameId(null);
      setThinking(false);
      return;
    }
    setRemoteGameId(game.id);
    setModelLabel(game.model_label);
    setPlayState(gameStateFromRemote(game));
    const search = await apiClient.analyzeGame(game.id);
    if (search && !search.terminal) setPlayAnalysis(analysisFromBackend(search));
    setThinking(false);
  }

  async function playColumn(column: number) {
    if (thinking || playState.currentPlayer !== 1 || playState.winner || isBoardFull(playState.board)) return;
    const result = dropPiece(playState.board, column, 1);
    if (!result) return;
    playDiscImpact(result.move, result.winner, muted);
    setPlayState((current) => ({
      ...current,
      board: result.board,
      currentPlayer: 2,
      winner: result.winner,
      moves: [...current.moves, result.move],
      lastMove: result.move,
    }));
    setPlayAnalysis(createAnalysis(result.board, 2, difficulty, DIFFICULTIES[difficulty].simulations));
    if (source !== "api" || !remoteGameId) return;

    setThinking(true);
    const [game] = await Promise.all([
      apiClient.move(remoteGameId, column),
      delay(discDropDuration(result.move.row) + 90),
    ]);
    if (!game) {
      setSource("demo");
      setThinking(false);
      return;
    }
    const search = game.ai_trace?.search;
    if (search) {
      for (let index = 0; index < search.snapshots.length; index += 1) {
        setPlayAnalysis(snapshotAnalysis(search, index));
        if (index < search.snapshots.length - 1) await delay(90);
      }
    }
    const remoteState = gameStateFromRemote(game);
    setPlayState(remoteState);
    setModelLabel(game.model_label);
    if (remoteState.lastMove) playDiscImpact(remoteState.lastMove, game.winner, muted);
    const nextSearch = game.terminal ? null : await apiClient.analyzeGame(game.id);
    if (nextSearch && !nextSearch.terminal) setPlayAnalysis(analysisFromBackend(nextSearch));
    setThinking(false);
  }

  function resetPlay() {
    const next = gameStateFromMoves([]);
    setPlayState(next);
    setPlayAnalysis(createAnalysis(next.board, 1, difficulty, DIFFICULTIES[difficulty].simulations));
    setThinking(false);
    playUiSound("select", muted);
    if (source === "api") void startRemoteGame(difficulty);
  }

  function changeDifficulty(nextDifficulty: DifficultyId) {
    setDifficulty(nextDifficulty);
    const simulations = DIFFICULTIES[nextDifficulty].simulations;
    setPlayAnalysis(createAnalysis(playState.board, playState.currentPlayer, nextDifficulty, simulations));
    playUiSound("select", muted);
    if (source === "api") void startRemoteGame(nextDifficulty);
  }

  function playLabColumn(column: number) {
    if (labState.winner || isBoardFull(labState.board)) return;
    const result = dropPiece(labState.board, column, labState.currentPlayer);
    if (!result) return;
    setLabState((current) => ({
      ...current,
      board: result.board,
      currentPlayer: current.currentPlayer === 1 ? 2 : 1,
      winner: result.winner,
      moves: [...current.moves, result.move],
      lastMove: result.move,
    }));
    setLabSnapshotIndex(0);
    playDiscImpact(result.move, result.winner, muted);
  }

  function resetLab() {
    setLabState(gameStateFromMoves(LAB_MOVES));
    setLabSnapshotIndex(0);
    playUiSound("select", muted);
  }

  function chooseReplay(id: string) {
    setSelectedReplayId(id);
    setReplayCursor(0);
    setReplayPlaying(false);
    setExportedReplayUrl(null);
    playUiSound("select", muted);
  }

  async function exportSelectedReplay() {
    if (!selectedReplay || source !== "api") return;
    setExportingReplay(true);
    const exported = await apiClient.exportReplay(selectedReplay.runId, selectedReplay.id);
    setExportedReplayUrl(exported?.mp4_url ?? null);
    setExportingReplay(false);
  }

  function openRun(runId: string) {
    setSelectedRunId(runId);
    setView("evolution");
    playUiSound("select", muted);
  }

  async function launchRun() {
    if (source !== "api" || launchingRun) return;
    setLaunchingRun(true);
    const created = await apiClient.createRun(trainingMinutes);
    if (created) {
      const runResult = await apiClient.getRuns();
      setRuns(runResult.data);
      setSelectedRunId(created.id);
      setEvolution([]);
      setView("evolution");
    }
    setLaunchingRun(false);
  }

  async function controlRun(action: "pause" | "resume" | "stop") {
    if (source !== "api" || !selectedRun) return;
    setLaunchingRun(true);
    const updated = await apiClient.runAction(selectedRun.id, action);
    if (updated) {
      const runResult = await apiClient.getRuns();
      setRuns(runResult.data);
    }
    setLaunchingRun(false);
  }

  const playInsight = coachInsight(playAnalysis, playState.currentPlayer);
  const playStatus = playState.winner === 1
    ? "You won the position"
    : playState.winner === 2
      ? "The engine found four"
      : isBoardFull(playState.board)
        ? "Board is full"
        : thinking
          ? "Engine is thinking"
          : "Your move";
  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? runs[0];

  function renderPlay() {
    return (
      <>
        <ViewHeading view="play">
          <SourceBadge source={source} />
          <button className="icon-button" onClick={resetPlay} title="Start a new game" type="button" aria-label="Start a new game">
            <RotateCcw size={17} />
          </button>
        </ViewHeading>
        <div className="play-layout">
          <section className="surface board-stage">
            <div className="stage-toolbar">
              <div className="stage-state">
                <span className={`live-state ${thinking ? "is-thinking" : ""}`}><span className="live-dot" />{thinking ? "Searching" : "Live"}</span>
                <span className="stage-title">{playStatus}</span>
                <span className="stage-meta">Move {playState.moves.length + 1} / 42</span>
              </div>
              <OverlaySelector value={overlay} onChange={setOverlay} />
            </div>
            <div className="board-canvas">
              <ConnectFourBoard
                analysis={playAnalysis}
                board={playState.board}
                currentPlayer={playState.currentPlayer}
                disabled={thinking || playState.currentPlayer !== 1 || Boolean(playState.winner) || isBoardFull(playState.board)}
                lastMove={playState.lastMove}
                onColumnSelect={playColumn}
                overlay={overlay}
              />
            </div>
            <div className="stage-footer">
              <div className="player-turn">
                <span className="turn-avatar turn-human">Y</span>
                <div><strong>You</strong><span>{playState.currentPlayer === 1 && !playState.winner ? "to move" : "human"}</span></div>
              </div>
              <div className="versus-mark">vs</div>
              <div className="player-turn player-turn-right">
                <div><strong>Zero</strong><span>{thinking ? "searching" : `${modelLabel} · ${DIFFICULTIES[difficulty].simulations} sims`}</span></div>
                <span className="turn-avatar turn-engine">Z</span>
              </div>
              <button className="text-button" onClick={resetPlay} type="button">New game <RotateCcw size={14} /></button>
            </div>
          </section>
          <aside className="analysis-column">
            <section className="surface analysis-panel">
              <div className="panel-heading">
                <div><p className="eyebrow">Live search</p><h2>What the tree sees</h2></div>
                <span className="panel-pulse"><Activity size={15} /> live</span>
              </div>
              <div className="analysis-summary">
                <div><span>Root value</span><strong className={playAnalysis.rootValue >= 0 ? "text-positive" : "text-negative"}>{formatQ(playAnalysis.rootValue)}</strong></div>
                <div><span>Visits</span><strong>{formatNumber(playAnalysis.simulations)}</strong></div>
                <div><span>Tree size</span><strong>{formatNumber(playAnalysis.treeSize)}</strong></div>
              </div>
              <div className="metric-section-heading"><span>Action distribution</span><span>{OVERLAY_LABELS[overlay]}</span></div>
              <MetricBars analysis={playAnalysis} mode={overlay} />
              <div className="analysis-footnote"><Zap size={14} /> The board overlay and bars stay on the same root.</div>
            </section>
            <section className={`surface coach-panel ${coach ? "is-enabled" : "is-muted"}`}>
              <div className="panel-heading">
                <div className="coach-title"><span className="coach-icon"><Lightbulb size={16} /></span><div><p className="eyebrow">Coach</p><h2>{coach ? "A little context" : "Coach is quiet"}</h2></div></div>
                <Switch checked={coach} label="Toggle Coach" onChange={setCoach} />
              </div>
              {coach ? <><span className="coach-tag">{playInsight.tag}</span><p className="coach-copy">{playInsight.body}</p><strong className="coach-headline">{playInsight.title}</strong></> : <p className="coach-copy">Turn Coach back on when you want a short read of the current position.</p>}
            </section>
            <section className="surface settings-panel">
              <div className="panel-heading"><div><p className="eyebrow">Play settings</p><h2>Choose your pace</h2></div><Settings2 size={17} className="muted-icon" /></div>
              <div className="difficulty-list">
                {(Object.keys(DIFFICULTIES) as DifficultyId[]).map((id) => (
                  <button className={`difficulty-option ${difficulty === id ? "is-selected" : ""}`} key={id} onClick={() => changeDifficulty(id)} type="button">
                    <span className="difficulty-radio">{difficulty === id ? <Check size={13} /> : null}</span>
                    <span><strong>{DIFFICULTIES[id].label}</strong><small>{DIFFICULTIES[id].detail}</small></span>
                    <span className="difficulty-sims">{DIFFICULTIES[id].simulations}</span>
                  </button>
                ))}
              </div>
            </section>
          </aside>
        </div>
      </>
    );
  }

  function renderLab() {
    const chartData = labBaseAnalysis.snapshots.map((snapshot) => ({
      simulation: snapshot.simulation,
      root: Number(snapshot.rootValue.toFixed(3)),
      best: Number(Math.max(...snapshot.qValues).toFixed(3)),
    }));
    const snapshotCount = labBaseAnalysis.snapshots.length;
    return (
      <>
        <ViewHeading view="lab">
          <span className="quiet-badge"><Database size={14} /> Position {labState.moves.length}</span>
          <button className="icon-button" onClick={resetLab} title="Reset lab position" type="button" aria-label="Reset lab position"><RotateCcw size={17} /></button>
        </ViewHeading>
        <section className="surface lab-controls">
          <div className="control-group"><span className="control-label">Overlay</span><OverlaySelector value={overlay} onChange={setOverlay} /></div>
          <label className="range-control"><span><span className="control-label">Simulations</span><strong>{labSimulations}</strong></span><input max="512" min="64" step="64" type="range" value={labSimulations} onChange={(event) => { setLabSimulations(Number(event.target.value)); setLabSnapshotIndex(0); }} /></label>
          <label className="range-control"><span><span className="control-label">Temperature</span><strong>{labTemperature.toFixed(2)}</strong></span><input max="0.8" min="0.05" step="0.05" type="range" value={labTemperature} onChange={(event) => setLabTemperature(Number(event.target.value))} /></label>
          <div className="control-toggle"><div><span className="control-label">Root noise</span><small>{labNoise ? "Dirichlet on" : "Deterministic"}</small></div><Switch checked={labNoise} label="Toggle root noise" onChange={setLabNoise} /></div>
        </section>
        <div className="lab-layout">
          <section className="surface board-stage lab-board-stage">
            <div className="stage-toolbar"><div className="stage-state"><span className="live-state"><span className="live-dot" />Snapshot {Math.min(labSnapshotIndex + 1, snapshotCount)} / {snapshotCount}</span><span className="stage-title">{labAnalysis.simulations} simulations</span></div><span className="stage-meta">Click a column to branch the position</span></div>
            <div className="board-canvas"><ConnectFourBoard analysis={labAnalysis} board={labState.board} currentPlayer={labState.currentPlayer} lastMove={labState.lastMove} onColumnSelect={playLabColumn} overlay={overlay} /></div>
            <div className="snapshot-scrubber"><div className="scrubber-heading"><span>Search playback</span><strong>{labAnalysis.simulations} visits</strong></div><input aria-label="Search snapshot" max={Math.max(0, snapshotCount - 1)} min="0" type="range" value={Math.min(labSnapshotIndex, Math.max(0, snapshotCount - 1))} onChange={(event) => setLabSnapshotIndex(Number(event.target.value))} /><div className="scrubber-scale"><span>root</span><span>stable</span><span>full search</span></div></div>
          </section>
          <aside className="lab-side-column">
            <section className="surface analysis-panel">
              <div className="panel-heading"><div><p className="eyebrow">Root telemetry</p><h2>Search settling</h2></div><span className="telemetry-value">{formatQ(labAnalysis.rootValue)}</span></div>
              <div className="lab-kpi-grid"><div><span>Tree</span><strong>{formatNumber(labAnalysis.treeSize)}</strong></div><div><span>Top action</span><strong>{columnName(labAnalysis.policy.indexOf(Math.max(...labAnalysis.policy)))}</strong></div><div><span>Policy entropy</span><strong>{(labAnalysis.policy.filter(Boolean).length * 0.43).toFixed(2)}</strong></div><div><span>Q spread</span><strong>{formatQ(Math.max(...labAnalysis.qValues) - Math.min(...labAnalysis.qValues))}</strong></div></div>
              <div className="metric-section-heading"><span>Column readout</span><span>{OVERLAY_LABELS[overlay]}</span></div><MetricBars analysis={labAnalysis} mode={overlay} />
            </section>
            <section className="surface chart-panel">
              <div className="panel-heading"><div><p className="eyebrow">Convergence</p><h2>Value over search</h2></div><Activity size={17} className="muted-icon" /></div>
              <div className="chart-wrap chart-small"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}><CartesianGrid stroke="#edf0f4" vertical={false} /><XAxis dataKey="simulation" tick={{ fill: "#98a2b3", fontSize: 10 }} tickLine={false} axisLine={false} /><YAxis domain={[-1, 1]} tick={{ fill: "#98a2b3", fontSize: 10 }} tickLine={false} axisLine={false} /><Tooltip contentStyle={{ border: "1px solid #e4e7ec", borderRadius: 8, fontSize: 12 }} /><Line dataKey="root" dot={false} stroke="#2f6fed" strokeWidth={2.2} name="Root" /><Line dataKey="best" dot={false} stroke="#f07c64" strokeWidth={2} name="Best Q" /></LineChart></ResponsiveContainer></div>
            </section>
            <section className="surface variation-panel"><div className="panel-heading"><div><p className="eyebrow">Principal variation</p><h2>Three likely lines</h2></div><Target size={17} className="muted-icon" /></div><div className="variation-list">{labAnalysis.principalVariations.map((line, index) => <div className="variation-row" key={`${index}-${line.join("-")}`}><span className="variation-index">{String(index + 1).padStart(2, "0")}</span><div>{line.map((column, moveIndex) => <span className={`move-chip ${moveIndex % 2 === 0 ? "move-chip-human" : "move-chip-engine"}`} key={`${column}-${moveIndex}`}>{columnName(column)}</span>)}</div><ChevronRight size={14} /></div>)}</div></section>
          </aside>
        </div>
      </>
    );
  }

  function renderEvolution() {
    const currentEvolution = latest(evolution, DEMO_EVOLUTION[DEMO_EVOLUTION.length - 1]);
    const currentRun = selectedRun ?? DEMO_RUNS[0];
    const chartData = evolution.map((point) => ({ ...point, winRate: point.arenaWinRate }));
    return (
      <>
        <ViewHeading view="evolution">
          <select className="duration-select" aria-label="Training duration" value={trainingMinutes ?? "unlimited"} onChange={(event) => setTrainingMinutes(event.target.value === "unlimited" ? null : Number(event.target.value))}>
            <option value="5">5 minutes</option><option value="15">15 minutes</option><option value="30">30 minutes</option><option value="60">60 minutes</option><option value="unlimited">Unlimited</option>
          </select>
          <button className="quiet-button primary-quiet" disabled={source !== "api" || launchingRun || runs.some((run) => run.status === "running")} onClick={() => void launchRun()} type="button"><Play size={15} />{launchingRun ? "Starting" : "Train from zero"}</button>
        </ViewHeading>
        <div className="stat-grid"><StatCard icon={Trophy} label="Latest arena score" value={formatPercent(currentEvolution.arenaWinRate)} detail="candidate vs previous champion; draws count half" tone="green" /><StatCard icon={GitBranch} label="Generation" value={`#${currentRun.generation}`} detail={`${formatNumber(currentEvolution.games)} replay positions`} tone="blue" /><StatCard icon={BarChart3} label="Elo estimate" value={Math.round(currentEvolution.elo).toLocaleString()} detail="checkpoint-relative estimate" tone="orange" /><StatCard icon={Cpu} label="Compute" value={currentRun.device.replace("Apple ", "")} detail="MPS / local" tone="slate" /></div>
        <div className="evolution-layout">
          <section className="surface chart-panel evolution-chart-panel"><div className="panel-heading"><div><p className="eyebrow">Run trajectory</p><h2>Strength is moving in the right direction</h2></div><div className="chart-legend"><span><i className="legend-line legend-win" />Arena score</span><span><i className="legend-line legend-policy" />Policy loss</span></div></div><div className="chart-wrap chart-evolution"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartData} margin={{ top: 16, right: 18, bottom: 4, left: -8 }}><CartesianGrid stroke="#edf0f4" vertical={false} /><XAxis dataKey="generation" tick={{ fill: "#98a2b3", fontSize: 11 }} tickLine={false} axisLine={false} tickFormatter={(value) => `G${value}`} /><YAxis yAxisId="left" tick={{ fill: "#98a2b3", fontSize: 11 }} tickLine={false} axisLine={false} domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} /><YAxis yAxisId="right" orientation="right" tick={{ fill: "#98a2b3", fontSize: 11 }} tickLine={false} axisLine={false} domain={[0.7, 2]} /><Tooltip contentStyle={{ border: "1px solid #e4e7ec", borderRadius: 8, fontSize: 12 }} formatter={(value, name) => [name === "Arena score" ? `${Math.round(Number(value) * 100)}%` : Number(value).toFixed(2), name]} labelFormatter={(label) => `Generation ${label}`} /><Line dataKey="winRate" name="Arena score" stroke="#2c9b73" strokeWidth={2.5} dot={{ r: 2.5, fill: "#2c9b73", strokeWidth: 0 }} yAxisId="left" /><Line dataKey="policyLoss" name="Policy loss" stroke="#f07c64" strokeWidth={2} dot={false} yAxisId="right" /></LineChart></ResponsiveContainer></div></section>
          <section className="surface run-profile"><div className="panel-heading"><div><p className="eyebrow">Selected run</p><h2>{currentRun.name}</h2></div><RunStatus status={currentRun.status} /></div><div className="profile-list"><div><span>Checkpoint</span><strong>{currentRun.checkpoint}</strong></div><div><span>Generation</span><strong>{currentRun.generation}</strong></div><div><span>Self-play</span><strong>{formatNumber(currentRun.games)} games</strong></div><div><span>Last update</span><strong>{formatDate(currentRun.updatedAt)}</strong></div></div><div className="profile-callout"><Sparkles size={16} /><span>The current run has cleared its strongest arena checkpoint so far.</span></div><div className="run-actions">{currentRun.status === "running" ? <button className="quiet-button" disabled={launchingRun} onClick={() => void controlRun("pause")} type="button"><Pause size={15} />Pause</button> : null}{currentRun.status === "paused" || currentRun.status === "interrupted" ? <button className="quiet-button" disabled={launchingRun} onClick={() => void controlRun("resume")} type="button"><Play size={15} />Resume</button> : null}{currentRun.status === "running" || currentRun.status === "paused" || currentRun.status === "interrupted" ? <button className="quiet-button" disabled={launchingRun} onClick={() => void controlRun("stop")} type="button"><Square size={14} />Stop</button> : null}</div><button className="full-button" onClick={() => navigate("lab")} type="button">Inspect checkpoint <ArrowUpRight size={15} /></button></section>
        </div>
        <section className="surface generation-table"><div className="panel-heading"><div><p className="eyebrow">Generation ledger</p><h2>Recent checkpoints</h2></div><span className="quiet-badge"><Clock3 size={14} /> Updated {formatDate(currentRun.updatedAt)}</span></div><div className="table-scroll"><table><thead><tr><th>Generation</th><th>Policy loss</th><th>Value loss</th><th>Arena</th><th>Elo</th><th>Games</th></tr></thead><tbody>{evolution.slice(-6).reverse().map((point) => <tr key={point.generation}><td><strong>G{String(point.generation).padStart(2, "0")}</strong>{point.generation === currentEvolution.generation ? <span className="current-tag">current</span> : null}</td><td>{point.policyLoss.toFixed(2)}</td><td>{point.valueLoss.toFixed(2)}</td><td><span className="table-positive">{formatPercent(point.arenaWinRate)}</span></td><td>{Math.round(point.elo).toLocaleString()}</td><td>{formatNumber(point.games)}</td></tr>)}</tbody></table></div></section>
      </>
    );
  }

  function renderReplay() {
    if (!selectedReplay) return null;
    const resultLabel = selectedReplay.result === "win" ? "You won" : selectedReplay.result === "loss" ? "Engine won" : "Draw";
    return (
      <>
        <ViewHeading view="replay"><span className="quiet-badge"><History size={14} /> {replays.length} recorded games</span>{exportedReplayUrl ? <a className="quiet-button" href={exportedReplayUrl} target="_blank" rel="noreferrer">Open MP4 <ArrowUpRight size={14} /></a> : <button className="quiet-button" disabled={source !== "api" || exportingReplay} onClick={() => void exportSelectedReplay()} type="button"><ArrowUpRight size={14} />{exportingReplay ? "Rendering" : "Export MP4"}</button>}</ViewHeading>
        <div className="replay-layout">
          <section className="surface replay-list-panel"><div className="panel-heading"><div><p className="eyebrow">Game archive</p><h2>Recent replays</h2></div><Search size={17} className="muted-icon" /></div><div className="replay-list">{replays.map((replay) => <button className={`replay-item ${replay.id === selectedReplay.id ? "is-selected" : ""}`} key={replay.id} onClick={() => chooseReplay(replay.id)} type="button"><div className={`replay-result result-${replay.result}`}><span>{replay.result === "draw" ? "=" : replay.result === "win" ? "W" : "L"}</span></div><div className="replay-item-copy"><strong>{replay.label}</strong><span>{replay.opening} · {replay.moves.length} moves</span></div><ChevronRight size={15} /></button>)}</div></section>
          <section className="surface board-stage replay-stage"><div className="stage-toolbar"><div className="stage-state"><span className={`replay-result-inline result-${selectedReplay.result}`}>{resultLabel}</span><span className="stage-title">{selectedReplay.label}</span><span className="stage-meta">{selectedReplay.opening}</span></div><span className="quiet-badge"><Cpu size={14} /> {selectedReplay.simulations} sims</span></div><div className="board-canvas"><ConnectFourBoard analysis={replayAnalysis} board={replayState.board} currentPlayer={replayState.currentPlayer} disabled lastMove={replayState.lastMove} overlay={overlay} /></div><div className="replay-scrubber"><div className="scrubber-heading"><button className="scrubber-play" onClick={() => { if (replayCursor >= selectedReplay.moves.length) setReplayCursor(0); setReplayPlaying((playing) => !playing); }} type="button" aria-label={replayPlaying ? "Pause replay" : "Play replay"}>{replayPlaying ? <Pause size={15} /> : <Play size={15} />}</button><span>Move {replayCursor} of {selectedReplay.moves.length}</span><strong>{replayCursor === 0 ? "Starting position" : `Played ${columnName(selectedReplay.moves[replayCursor - 1])}`}</strong></div><input aria-label="Replay move" className="replay-range" max={selectedReplay.moves.length} min="0" type="range" value={replayCursor} onChange={(event) => { setReplayCursor(Number(event.target.value)); setReplayPlaying(false); }} /><div className="replay-controls"><button className="icon-button small" disabled={replayCursor === 0} onClick={() => setReplayCursor((cursor) => Math.max(0, cursor - 1))} type="button" aria-label="Previous move" title="Previous move"><ChevronLeft size={15} /></button><div className="move-timeline">{selectedReplay.moves.map((column, index) => <button className={`timeline-move ${index < replayCursor ? "is-played" : ""} ${index === replayCursor - 1 ? "is-current" : ""}`} key={`${column}-${index}`} onClick={() => { setReplayCursor(index + 1); setReplayPlaying(false); }} type="button" aria-label={`Move ${index + 1}, column ${column + 1}`}>{column + 1}</button>)}</div><button className="icon-button small" disabled={replayCursor === selectedReplay.moves.length} onClick={() => setReplayCursor((cursor) => Math.min(selectedReplay.moves.length, cursor + 1))} type="button" aria-label="Next move" title="Next move"><ChevronRight size={15} /></button></div></div></section>
        </div>
        <section className="surface replay-note"><div className="note-icon"><BrainCircuit size={17} /></div><div><p className="eyebrow">Position note</p><strong>{selectedReplay.notes}</strong></div><div className="note-meta"><span>Run</span><strong>{selectedReplay.runId}</strong><span>Captured</span><strong>{formatDate(selectedReplay.createdAt)}</strong></div></section>
      </>
    );
  }

  function renderRuns() {
    return (
      <>
        <ViewHeading view="runs"><SourceBadge source={source} /><button className="quiet-button" onClick={() => void apiClient.getRuns().then((result) => { setRuns(result.data); setSource(result.source); })} type="button"><RotateCcw size={15} /> Refresh</button></ViewHeading>
        <section className="surface runs-toolbar"><div className="runs-toolbar-copy"><div className="toolbar-symbol"><ListFilter size={17} /></div><div><strong>{runs.length} local runs</strong><span>Sorted by most recently updated</span></div></div><div className="toolbar-filters"><button className="filter-chip is-active" type="button">All</button><button className="filter-chip" type="button">Running</button><button className="filter-chip" type="button">Complete</button></div></section>
        <section className="surface runs-table"><div className="table-scroll"><table><thead><tr><th>Run</th><th>Status</th><th>Generation</th><th>Latest arena score</th><th>Device</th><th>Updated</th><th /></tr></thead><tbody>{runs.map((run) => <tr key={run.id}><td><div className="run-name-cell"><span className="run-avatar"><GitBranch size={15} /></span><div><strong>{run.name}</strong><span>{run.id} · {run.checkpoint}</span></div></div></td><td><RunStatus status={run.status} /></td><td><strong>G{String(run.generation).padStart(2, "0")}</strong></td><td><span className="table-positive">{formatPercent(run.winRate)}</span></td><td><span className="device-cell"><Cpu size={14} />{run.device}</span></td><td>{formatDate(run.updatedAt)}</td><td><button className="table-action" onClick={() => openRun(run.id)} type="button">Open <ArrowUpRight size={14} /></button></td></tr>)}</tbody></table></div></section>
        <div className="runs-footer"><div><CircleHelp size={15} /><span>Runs stay local until the API is connected.</span></div><button className="text-button" onClick={() => navigate("evolution")} type="button">See evolution <ArrowUpRight size={14} /></button></div>
      </>
    );
  }

  function renderView() {
    if (view === "lab") return renderLab();
    if (view === "evolution") return renderEvolution();
    if (view === "replay") return renderReplay();
    if (view === "runs") return renderRuns();
    return renderPlay();
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup"><span className="brand-mark"><span /><span /><span /><span /></span><div><strong>Connect Four</strong><span>Zero</span></div></div>
        <div className="workspace-switcher"><span className="workspace-avatar">A</span><div><strong>Apple Silicon lab</strong><span>Local workspace</span></div><ChevronUp size={14} /></div>
        <div className="sidebar-section-label">Workspace</div>
        <nav className="sidebar-nav" aria-label="Primary navigation">{NAV_ITEMS.map(({ id, label, hint, icon: Icon }) => <button className={`nav-item ${view === id ? "is-active" : ""}`} key={id} onClick={() => navigate(id)} type="button"><span className="nav-icon"><Icon size={17} strokeWidth={view === id ? 2.2 : 1.8} /></span><span className="nav-copy"><strong>{label}</strong><small>{hint}</small></span>{view === id ? <span className="nav-active-mark" /> : null}</button>)}</nav>
        <div className="sidebar-spacer" />
        <div className="sidebar-footer"><div className="engine-card"><div className="engine-card-heading"><span className="engine-orb"><BrainCircuit size={15} /></span><div><strong>Zero engine</strong><span>{source === "api" ? "Connected to API" : "Running in demo mode"}</span></div></div><div className="engine-meter"><span style={{ width: source === "api" ? "82%" : "64%" }} /></div><div className="engine-card-foot"><span>MPS ready</span><span>local</span></div></div><div className="sidebar-meta"><span>v0.1 research build</span><span className="keyboard-hint"><Command size={11} /> K</span></div></div>
      </aside>
      <main className="app-main">
        <header className="topbar"><div className="mobile-brand"><span className="brand-mark small-mark"><span /><span /><span /><span /></span><strong>{VIEW_META[view].eyebrow}</strong></div><div className="topbar-context"><span>Connect Four Zero</span><ChevronRight size={14} /><strong>{VIEW_META[view].title}</strong></div><div className="topbar-actions"><SourceBadge source={source} /><button className="icon-button" onClick={() => setMuted((value) => !value)} title={muted ? "Unmute sounds" : "Mute sounds"} type="button" aria-label={muted ? "Unmute sounds" : "Mute sounds"}>{muted ? <VolumeX size={17} /> : <Volume2 size={17} />}</button><button className="avatar-button" type="button" aria-label="Open account menu">A</button></div></header>
        <div className="content-wrap">{renderView()}</div>
      </main>
      <nav className="mobile-nav" aria-label="Mobile navigation">{NAV_ITEMS.map(({ id, label, icon: Icon }) => <button className={view === id ? "is-active" : ""} key={id} onClick={() => navigate(id)} type="button"><Icon size={18} /><span>{label}</span></button>)}</nav>
    </div>
  );
}

export default App;
