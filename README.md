# Connect Four Zero

A local, inspectable AlphaGo Zero system for standard 7x6 Connect Four. It trains from random weights with self-play, PUCT search, and a single residual policy/value network accelerated by Apple Metal.

Profile faithful model, batch-size, and mixed-precision variants with `connect4-zero benchmark-mps`. The benchmark measures both optimization and inference throughput so faster training does not hide slower self-play.

The measured M5 Max default is batch size 1024 with MPS FP16 autocast. Checkpoints retain FP32 master weights, and CPU debug runs remain FP32.

## Setup

```bash
conda env create -f environment.yml
npm --prefix web install
conda activate connect4-zero
wandb login
```

## Run

```bash
make api
make web
```

Open `http://localhost:5173`. The API runs at `http://localhost:8000`.

Training refuses to run without PyTorch MPS unless `--allow-cpu` is passed explicitly. Generated runs, checkpoints, databases, W&B state, and exports are local and ignored by Git.

```bash
make test
make train-60
```

Compare any checkpoint against a deterministic suite of exact late-game probes:

```bash
conda run -n connect4-zero connect4-zero audit-accuracy runs/<run-id>/checkpoints/<checkpoint>.pt
```
