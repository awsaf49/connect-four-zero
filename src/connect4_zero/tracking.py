from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class WandbTracker:
    """Best-effort W&B mirror. Local storage is always authoritative."""

    def __init__(self, run_id: str, config: dict[str, Any], mode: str, directory: Path) -> None:
        self.run = None
        self.error: str | None = None
        if mode == "disabled":
            return
        try:
            import wandb

            self.run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "connect-four-zero"),
                entity=os.getenv("WANDB_ENTITY") or None,
                id=run_id,
                name=f"Connect Four Zero {run_id[-6:]}",
                resume="allow",
                mode=mode,
                config=config,
                dir=str(directory),
                settings=wandb.Settings(start_method="thread"),
            )
        except Exception as exc:  # W&B must never interrupt training.
            self.error = str(exc)

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self.run.log(values, step=step)
        except Exception as exc:
            self.error = str(exc)

    def log_checkpoint(self, path: Path, aliases: list[str], metadata: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            import wandb

            artifact = wandb.Artifact(f"model-{self.run.id}", type="model", metadata=metadata)
            artifact.add_file(str(path))
            self.run.log_artifact(artifact, aliases=aliases)
        except Exception as exc:
            self.error = str(exc)

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception as exc:
                self.error = str(exc)

