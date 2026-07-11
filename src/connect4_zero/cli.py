from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import torch
import uvicorn

from .network import PolicyValueNet
from .network import PolicyValueEvaluator
from .evaluation import play_vs_random
from .schemas import RunConfig, RunRecord, RunStatus
from .store import RunStore
from .trainer import TrainingController


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="connect4-zero")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="start the local API and built web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    train = commands.add_parser("train", help="run a zero-knowledge experiment")
    train.add_argument("--minutes", type=float, default=60.0)
    train.add_argument("--simulations", type=int, default=256)
    train.add_argument("--games", type=int, default=64)
    train.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=os.getenv("WANDB_MODE", "online"))
    train.add_argument("--allow-cpu", action="store_true")

    commands.add_parser("system", help="show accelerator readiness")
    evaluate = commands.add_parser("evaluate", help="benchmark a checkpoint against random play")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--games", type=int, default=200)
    evaluate.add_argument("--simulations", type=int, default=64)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "serve":
        uvicorn.run("connect4_zero.api:app", host=args.host, port=args.port, reload=args.reload)
        return
    if args.command == "system":
        print(
            json.dumps(
                {
                    "mps_built": torch.backends.mps.is_built(),
                    "mps_available": torch.backends.mps.is_available(),
                    "torch": torch.__version__,
                    "model_parameters": PolicyValueNet().parameter_count,
                    "mps_recommended_memory": torch.mps.recommended_max_memory() if torch.backends.mps.is_available() else None,
                },
                indent=2,
            )
        )
        return
    if args.command == "evaluate":
        evaluator, metadata = PolicyValueEvaluator.load(args.checkpoint)
        result = play_vs_random(evaluator, games=args.games, simulations=args.simulations)
        print(json.dumps({"checkpoint": str(args.checkpoint), "metadata": metadata, **result.as_dict()}, indent=2))
        return

    run_id = f"run-{uuid.uuid4().hex[:12]}"
    config = RunConfig(
        minutes=args.minutes,
        simulations=args.simulations,
        evaluation_simulations=args.simulations,
        self_play_games=args.games,
        allow_cpu=args.allow_cpu,
        wandb_mode=args.wandb_mode,
    )
    store = RunStore(Path("runs"), run_id)
    store.create_run(RunRecord(id=run_id, name=f"CLI run {run_id[-4:]}", status=RunStatus.CREATED, config=config))

    def report(event: dict[str, object]) -> None:
        if event.get("type") in {"phase", "iteration", "run_status"}:
            print(json.dumps(event, default=str), flush=True)

    controller = TrainingController(store, config, report)
    controller.start()
    try:
        while controller.thread and controller.thread.is_alive():
            controller.join(1)
    except KeyboardInterrupt:
        controller.stop()
        controller.join(30)


if __name__ == "__main__":
    main()
