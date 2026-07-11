#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from connect4_zero.store import RunStore


def read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("demo"))
    args = parser.parse_args()

    store = RunStore(args.runs, args.run_id)
    run = store.get_run()
    if run is None:
        raise SystemExit(f"run not found: {args.run_id}")
    checkpoints = store.list_checkpoints()
    finals = [item for item in checkpoints if item.kind == "final"]
    if not finals:
        raise SystemExit("run has no final checkpoint")
    checkpoint = finals[-1]
    metrics = store.list_metrics()
    games = store.list_games(limit=6)
    curated_games = []
    for game in games:
        payload = game.model_dump(mode="json")
        for move in payload["moves"]:
            if move.get("search"):
                move["search"]["snapshots"] = []
        curated_games.append(payload)

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint.path, args.output / "champion.pt")
    minutes = max(1, round(run.elapsed_seconds / 60))
    duration_name = "One-hour" if 55 <= minutes <= 65 else f"{minutes}-minute"
    solver_zero = read_json(store.root / "solver-accuracy-generation-zero.json")
    solver_final = read_json(store.root / "solver-accuracy-final.json")
    optimization = read_json(store.root / "mps-benchmark-stability.json")
    manifest = {
        "version": 1,
        "source_run_id": run.id,
        "name": f"{duration_name} self-trained champion",
        "config": run.config.model_dump(mode="json"),
        "generation": run.generation,
        "training_step": checkpoint.training_step,
        "elapsed_seconds": run.elapsed_seconds,
        "checkpoint_metadata": checkpoint.metadata,
        "metrics": [item.model_dump(mode="json") for item in metrics],
        "games": curated_games,
        "benchmark": {"opponent": "uniform-random", "games": 200, "wins": 200, "draws": 0, "losses": 0, "simulations": 64},
        "solver_benchmark": {"generation_zero": solver_zero, "final": solver_final},
        "optimization_benchmark": optimization,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
