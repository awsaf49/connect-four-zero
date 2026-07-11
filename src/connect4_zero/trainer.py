from __future__ import annotations

import copy
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from .game import ConnectFour
from .mcts import MCTS, Node, SearchResult
from .network import PolicyValueEvaluator, PolicyValueNet, resolve_device
from .schemas import (
    CheckpointRecord,
    GameReplay,
    MetricSample,
    MoveTrace,
    RunConfig,
    RunRecord,
    RunStatus,
    SearchTrace,
)
from .store import RunStore
from .tracking import WandbTracker

EventCallback = Callable[[dict[str, object]], None]


@dataclass(slots=True)
class TrainingExample:
    encoded: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.examples: deque[TrainingExample] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def extend(self, examples: Sequence[TrainingExample]) -> None:
        self.examples.extend(examples)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.examples:
            raise ValueError("cannot sample an empty replay buffer")
        indices = self.rng.integers(0, len(self.examples), size=batch_size)
        encoded, policies, values = [], [], []
        for index in indices:
            example = self.examples[int(index)]
            board = example.encoded.copy()
            policy = example.policy.copy()
            if self.rng.random() < 0.5:
                board = np.flip(board, axis=2).copy()
                policy = policy[::-1].copy()
            if self.rng.random() < 0.5:
                board[[0, 1]] = board[[1, 0]]
                board[2] = 1.0 - board[2]
            encoded.append(board)
            policies.append(policy)
            values.append(example.value)
        return (
            np.stack(encoded).astype(np.float32),
            np.stack(policies).astype(np.float32),
            np.asarray(values, dtype=np.float32),
        )

    def save(self, path: Path) -> None:
        if not self.examples:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            encoded=np.stack([item.encoded for item in self.examples]),
            policies=np.stack([item.policy for item in self.examples]),
            values=np.asarray([item.value for item in self.examples], dtype=np.float32),
        )
        temporary.replace(path)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        payload = np.load(path)
        self.examples.clear()
        for encoded, policy, value in zip(payload["encoded"], payload["policies"], payload["values"]):
            self.examples.append(TrainingExample(encoded, policy, float(value)))


@dataclass(slots=True)
class ActiveGame:
    index: int
    state: ConnectFour = field(default_factory=ConnectFour.new)
    root: Node | None = None
    history: list[tuple[np.ndarray, np.ndarray, int]] = field(default_factory=list)
    moves: list[MoveTrace] = field(default_factory=list)


class TrainingController:
    def __init__(
        self,
        store: RunStore,
        config: RunConfig,
        callback: EventCallback | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.callback = callback or (lambda event: None)
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started_at = 0.0
        self.step = 0
        self.generation = 0
        self.device = resolve_device(config.allow_cpu)
        self.replay = ReplayBuffer(config.replay_capacity, config.seed)
        self.replay.load(store.replay_path)
        self.training_model = PolicyValueNet().to(self.device)
        self.champion_model = copy.deepcopy(self.training_model).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.training_model.parameters(),
            lr=0.05,
            momentum=0.9,
            weight_decay=1e-4,
        )
        self.tracker: WandbTracker | None = None
        self._restore_if_available()

    def _restore_if_available(self) -> None:
        trainer_state = self.store.root / "trainer_state.pt"
        if trainer_state.exists():
            payload = torch.load(trainer_state, map_location="cpu", weights_only=True)
            self.training_model.load_state_dict(payload["training_model"])
            self.champion_model.load_state_dict(payload["champion_model"])
            self.optimizer.load_state_dict(payload["optimizer"])
            self.step = int(payload["step"])
            self.generation = int(payload["generation"])
            return
        checkpoints = [item for item in self.store.list_checkpoints() if Path(item.path).exists()]
        if not checkpoints:
            return
        latest_training = max(checkpoints, key=lambda item: (item.training_step, item.created_at))
        champion_candidates = [
            item for item in checkpoints if item.kind in {"initial", "champion", "final", "demo"}
        ]
        latest_champion = max(
            champion_candidates or checkpoints,
            key=lambda item: (item.generation, item.training_step, item.created_at),
        )
        training_payload = torch.load(latest_training.path, map_location="cpu", weights_only=True)
        champion_payload = torch.load(latest_champion.path, map_location="cpu", weights_only=True)
        self.training_model.load_state_dict(training_payload["model"])
        self.champion_model.load_state_dict(champion_payload["model"])
        self.step = latest_training.training_step
        self.generation = latest_champion.generation

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, name=f"training-{self.store.root.name}", daemon=True)
        self.thread.start()

    def pause(self) -> None:
        self.pause_event.set()
        self.store.update_run(status=RunStatus.PAUSED, phase="paused")
        self._emit("run_status", status="paused")

    def resume(self) -> None:
        self.pause_event.clear()
        self.store.update_run(status=RunStatus.RUNNING)
        self._emit("run_status", status="running")

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        self.store.update_run(status=RunStatus.STOPPING, phase="stopping")

    def join(self, timeout: float | None = None) -> None:
        if self.thread:
            self.thread.join(timeout)

    def _wait_if_paused(self) -> None:
        while self.pause_event.is_set() and not self.stop_event.is_set():
            time.sleep(0.2)

    def _budget_exhausted(self) -> bool:
        if self.stop_event.is_set():
            return True
        return self.config.minutes is not None and time.monotonic() - self.started_at >= self.config.minutes * 60

    def _run(self) -> None:
        self.started_at = time.monotonic()
        run = self.store.get_run()
        if run is None:
            raise RuntimeError("store must contain a run before training starts")
        self.store.update_run(status=RunStatus.RUNNING, phase="initializing")
        self.tracker = WandbTracker(run.id, self.config.model_dump(), self.config.wandb_mode, self.store.root)
        try:
            self._save_checkpoint("initial", score=None)
            while not self._budget_exhausted():
                self._wait_if_paused()
                if self._budget_exhausted():
                    break
                self.store.update_run(phase="self_play", generation=self.generation)
                self._emit("phase", phase="self_play", generation=self.generation)
                examples, games = self._self_play(self.config.self_play_games)
                if not examples:
                    break
                self.replay.extend(examples)
                for game in games:
                    self.store.add_game(game)
                self.replay.save(self.store.replay_path)

                self.store.update_run(phase="training")
                losses = self._train_candidate()
                candidate = self._save_checkpoint("candidate", score=None)

                self.store.update_run(phase="evaluation")
                score = self._evaluate_candidate()
                promoted = score > self.config.promotion_threshold
                if promoted:
                    self.generation += 1
                    self.champion_model.load_state_dict(copy.deepcopy(self.training_model.state_dict()))
                    candidate = self._save_checkpoint("champion", score=score)
                else:
                    candidate = self._save_checkpoint("rejected", score=score)

                elapsed = time.monotonic() - self.started_at
                values: dict[str, float | int | str | bool] = {
                    **losses,
                    "evaluation/score": score,
                    "evaluation/promoted": promoted,
                    "run/generation": self.generation,
                    "run/replay_positions": len(self.replay),
                    "run/elapsed_seconds": elapsed,
                    "system/mps_allocated_bytes": float(torch.mps.current_allocated_memory()) if self.device.type == "mps" else 0.0,
                }
                metric = MetricSample(step=self.step, wall_time=elapsed, values=values)
                self.store.add_metric(metric)
                if self.tracker:
                    self.tracker.log(values, step=self.step)
                    self.tracker.log_checkpoint(Path(candidate.path), [candidate.kind, f"generation-{candidate.generation}"], candidate.metadata)
                self.store.update_run(elapsed_seconds=elapsed, generation=self.generation)
                self._emit("iteration", **values)

            final = self._save_checkpoint("final", score=None)
            elapsed = time.monotonic() - self.started_at
            status = RunStatus.COMPLETED
            self.store.update_run(status=status, phase="complete", elapsed_seconds=elapsed, generation=self.generation)
            self.replay.save(self.store.replay_path)
            self._emit("run_status", status="completed", checkpoint=final.model_dump(mode="json"))
            if self.tracker:
                self.tracker.finish()
        except Exception as exc:
            self.store.update_run(status=RunStatus.FAILED, phase="failed", error=str(exc))
            self._emit("run_status", status="failed", error=str(exc))
            if self.tracker:
                self.tracker.finish(exit_code=1)

    def _learning_rate(self) -> float:
        if self.config.minutes is None:
            return 0.05 if self.step < 2_000 else 0.01 if self.step < 10_000 else 0.001
        progress = min(1.0, (time.monotonic() - self.started_at) / (self.config.minutes * 60))
        return 0.05 if progress < 0.5 else 0.01 if progress < 0.8 else 0.001

    def _train_candidate(self) -> dict[str, float]:
        self.training_model.train()
        totals = np.zeros(3, dtype=np.float64)
        steps = self.config.training_steps
        started = time.perf_counter()
        for _ in range(steps):
            if self._budget_exhausted():
                break
            self._wait_if_paused()
            inputs, target_policy, target_value = self.replay.sample(self.config.batch_size)
            x = torch.from_numpy(inputs).to(self.device)
            pi = torch.from_numpy(target_policy).to(self.device)
            z = torch.from_numpy(target_value).to(self.device)
            for group in self.optimizer.param_groups:
                group["lr"] = self._learning_rate()
            logits, values = self.training_model(x)
            policy_loss = -(pi * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(values, z)
            loss = policy_loss + value_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            totals += [float(loss.detach()), float(policy_loss.detach()), float(value_loss.detach())]
            self.step += 1
        if self.device.type == "mps":
            torch.mps.synchronize()
        elapsed = max(time.perf_counter() - started, 1e-6)
        divisor = max(1, steps)
        return {
            "loss/total": totals[0] / divisor,
            "loss/policy": totals[1] / divisor,
            "loss/value": totals[2] / divisor,
            "training/steps": float(steps),
            "training/positions_per_second": steps * self.config.batch_size / elapsed,
            "training/learning_rate": self._learning_rate(),
        }

    def _self_play(self, count: int) -> tuple[list[TrainingExample], list[GameReplay]]:
        evaluator = PolicyValueEvaluator(self.champion_model, device=self.device, allow_cpu=self.config.allow_cpu)
        search = MCTS(
            evaluator,
            simulations=self.config.simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            seed=self.config.seed + self.step,
        )
        active = [ActiveGame(index=index) for index in range(count)]
        completed: list[ActiveGame] = []
        started = time.perf_counter()
        while active and not self._budget_exhausted():
            self._wait_if_paused()
            results, roots = search.search_many(
                [game.state for game in active],
                add_noise=True,
                roots=[game.root for game in active],
                record_snapshots=len(completed) == 0,
            )
            survivors: list[ActiveGame] = []
            for game, result, root in zip(active, results, roots):
                before = game.state
                game.history.append((before.encode(), result.policy.copy(), before.current_player))
                temperature = 1.0 if before.move_count < self.config.temperature_moves else 0.0
                action = search.select_action(result, temperature)
                after = before.play(action)
                game.moves.append(self._move_trace(before, after, action, result))
                game.state = after
                game.root = MCTS.advance_root(root, action)
                if after.is_terminal:
                    completed.append(game)
                else:
                    survivors.append(game)
            active = survivors
            if completed:
                self._emit(
                    "self_play",
                    completed=len(completed),
                    total=count,
                    board=completed[-1].state.as_list(),
                    winner=completed[-1].state.winner,
                )

        examples: list[TrainingExample] = []
        replays: list[GameReplay] = []
        for game in completed:
            winner = game.state.winner
            for encoded, policy, player in game.history:
                value = 0.0 if winner == 0 else 1.0 if player == winner else -1.0
                examples.append(TrainingExample(encoded, policy, value))
            replays.append(
                GameReplay(
                    id=f"sp-{uuid.uuid4().hex[:12]}",
                    run_id=self.store.root.name,
                    source="self_play",
                    player_one=f"champion-{self.generation}",
                    player_two=f"champion-{self.generation}",
                    winner=winner,
                    moves=game.moves,
                    metadata={"generation": self.generation},
                )
            )
        elapsed = max(time.perf_counter() - started, 1e-6)
        self._emit("self_play_complete", games=len(completed), positions=len(examples), seconds=elapsed)
        return examples, replays

    def _evaluate_candidate(self) -> float:
        candidate_eval = PolicyValueEvaluator(self.training_model, device=self.device, allow_cpu=self.config.allow_cpu)
        champion_eval = PolicyValueEvaluator(self.champion_model, device=self.device, allow_cpu=self.config.allow_cpu)
        candidate_search = MCTS(candidate_eval, simulations=self.config.evaluation_simulations, seed=self.config.seed + self.step)
        champion_search = MCTS(champion_eval, simulations=self.config.evaluation_simulations, seed=self.config.seed + self.step + 1)
        games = [ActiveGame(index=index) for index in range(self.config.evaluation_games)]
        candidate_players = [1 if index % 2 == 0 else 2 for index in range(len(games))]
        active = list(range(len(games)))
        while active and not self._budget_exhausted():
            candidate_turns = [index for index in active if games[index].state.current_player == candidate_players[index]]
            champion_turns = [index for index in active if index not in candidate_turns]
            for indices, search in ((candidate_turns, candidate_search), (champion_turns, champion_search)):
                if not indices:
                    continue
                results, roots = search.search_many([games[index].state for index in indices])
                for index, result, root in zip(indices, results, roots):
                    game = games[index]
                    action = search.select_action(result, 0.0)
                    before = game.state
                    game.state = before.play(action)
                    game.moves.append(self._move_trace(before, game.state, action, result))
            active = [index for index in active if not games[index].state.is_terminal]
        if active:
            return 0.0
        score = 0.0
        for index, game in enumerate(games):
            if game.state.winner == 0:
                score += 0.5
            elif game.state.winner == candidate_players[index]:
                score += 1.0
            if index < 4:
                self.store.add_game(
                    GameReplay(
                        id=f"ev-{uuid.uuid4().hex[:12]}",
                        run_id=self.store.root.name,
                        source="evaluation",
                        player_one="candidate" if candidate_players[index] == 1 else f"champion-{self.generation}",
                        player_two="candidate" if candidate_players[index] == 2 else f"champion-{self.generation}",
                        winner=game.state.winner,
                        moves=game.moves,
                        metadata={"candidate_player": candidate_players[index]},
                    )
                )
        return score / len(games)

    def _save_checkpoint(self, kind: str, score: float | None) -> CheckpointRecord:
        checkpoint_id = f"g{self.generation:03d}-s{self.step:06d}-{kind}"
        path = self.store.checkpoint_dir / f"{checkpoint_id}.pt"
        model = self.champion_model if kind in {"initial", "champion", "final"} else self.training_model
        evaluator = PolicyValueEvaluator(model, device=self.device, allow_cpu=self.config.allow_cpu)
        metadata = {
            "generation": self.generation,
            "training_step": self.step,
            "kind": kind,
            "score": score,
            "parameter_count": model.parameter_count,
            "device": str(self.device),
        }
        evaluator.save(path, metadata)
        record = CheckpointRecord(
            id=checkpoint_id,
            run_id=self.store.root.name,
            generation=self.generation,
            training_step=self.step,
            kind=kind,  # type: ignore[arg-type]
            path=str(path.resolve()),
            score=score,
            metadata=metadata,
        )
        self.store.add_checkpoint(record)
        self._save_trainer_state()
        return record

    def _save_trainer_state(self) -> None:
        path = self.store.root / "trainer_state.pt"
        temporary = path.with_suffix(".tmp.pt")
        torch.save(
            {
                "version": 1,
                "training_model": self.training_model.state_dict(),
                "champion_model": self.champion_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "generation": self.generation,
            },
            temporary,
        )
        temporary.replace(path)

    @staticmethod
    def _move_trace(before: ConnectFour, after: ConnectFour, action: int, result: SearchResult) -> MoveTrace:
        return MoveTrace(
            ply=before.move_count,
            player=before.current_player,
            action=action,
            board_before=before.as_list(),
            board_after=after.as_list(),
            search=SearchTrace(
                simulations=result.simulations,
                priors=[float(value) for value in result.priors],
                visits=result.visits.tolist(),
                q_values=[float(value) for value in result.q_values],
                root_value=result.root_value,
                principal_variations=result.principal_variations,
                snapshots=[
                    {
                        "simulation": snapshot.simulation,
                        "visits": snapshot.visits,
                        "q_values": snapshot.q_values,
                        "priors": snapshot.priors,
                        "root_value": snapshot.root_value,
                    }
                    for snapshot in result.snapshots
                ],
            ),
        )

    def _emit(self, event_type: str, **payload: object) -> None:
        event = {"version": 1, "type": event_type, "run_id": self.store.root.name, **payload}
        try:
            self.callback(event)
        except Exception:
            pass
