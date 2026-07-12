# Connect Four Zero

A local, inspectable AlphaGo Zero-style laboratory for standard 7x6 Connect Four. The agent starts from random weights, creates its own training data through self-play, evaluates new candidates against previous champions, and exposes its policy, value, and MCTS search in a playable interface.

![Connect Four Zero play interface](docs/connect-four-zero.jpg)

## What You Can Do

- Play against historical checkpoints at four search strengths.
- Watch policy priors, visit counts, Q-values, root value, and principal variations.
- Train a new model from zero for 5, 15, 30, or 60 minutes, or indefinitely.
- Scrub through self-play, evaluation, and human games.
- Compare generations and promotion results.
- Export recorded games as MP4 and portable replay JSON.
- Grade moves with an isolated exact Connect Four solver in Coach mode.

The solver is an auditor only. It is not imported by self-play or training.

## Bundled Champion

The included checkpoint was trained by this repository for one hour on an Apple M5 Max:

| Measurement | Result |
| --- | ---: |
| Network | 64 channels, 6 residual blocks |
| Parameters | 449,370 |
| Promoted generations | 19 |
| Recorded self-play games | 1,905 |
| Replay positions | 57,264 |
| Against uniform random, 64 MCTS simulations | 200 wins / 200 games |
| Exact tactical probes, 800 MCTS simulations | 21 / 24 |

This is a self-trained research checkpoint, not a perfect Connect Four solver. A human can beat it. New training runs always initialize randomly and cannot consume the bundled weights, games, or replay buffer.

## Run It

Requirements:

- Apple Silicon Mac with PyTorch MPS for production training
- Conda
- Node.js and npm
- ffmpeg for MP4 replay export

```bash
conda env create -f environment.yml
npm --prefix web install
conda activate connect4-zero
npm --prefix web run build
make start
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

For frontend and API development with hot reload:

```bash
make dev
```

## Train From Zero

```bash
make train-60
```

Production training refuses to start unless `torch.backends.mps.is_available()` is true. CPU execution is reserved for tests and explicit debugging.

The default learning system uses:

- Three 6x7 input planes
- A residual policy/value network
- PUCT Monte Carlo tree search
- 64 concurrent self-play games
- A 100,000-position replay window
- Candidate-versus-champion promotion gates
- Batch size 1,024 with MPS FP16 autocast
- Horizontal reflection and color-swap augmentation

Profile model, precision, and batch-size variants on Apple Metal:

```bash
connect4-zero benchmark-mps
```

Audit a checkpoint against a deterministic exact-solver suite:

```bash
connect4-zero audit-accuracy \
  runs/<run-id>/checkpoints/<checkpoint>.pt \
  --positions 24 --simulations 800
```

## Verification

```bash
make test
npm --prefix web run build
```

The test suite covers game rules, bitboard equivalence, PUCT behavior, perspective changes, augmentation, checkpoint gating, persistence, MPS execution, solver isolation, API behavior, and frontend data contracts.

## Experiment Tracking

Local SQLite databases and checkpoints remain the source of truth. W&B logging is optional and never interrupts training.

```bash
wandb login
export WANDB_PROJECT=connect-four-zero
make train-60
```

Runs, checkpoints, exports, W&B state, environments, credentials, and frontend build output are excluded from Git.

## Deployment

The React frontend can be deployed as a static showcase. In that mode it uses the bundled evolution timeline and replay data, but it does **not** run the neural checkpoint.

Real neural play requires the FastAPI service because every move performs MCTS backed by the policy/value network. Training should remain on an Apple Silicon machine; inference can be adapted to a small CPU or GPU service. The clean public architecture is therefore:

```text
Static frontend  ->  FastAPI inference service  ->  bundled .pt checkpoint
                                      |
                                      +-> optional local training runs
```

Do not expose unrestricted training endpoints on a public deployment. A hosted version should serve read-only runs and rate-limited human games while keeping training local.
