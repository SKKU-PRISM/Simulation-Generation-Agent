# Docker Release Guide

The public Docker environment targets `headless IsaacLab + data collection + e2e batch`. `Isaac Sim + MCP` visual verification is outside the scope of this container.

## Target Runtime

- Host OS: Ubuntu 22.04 recommended
- GPU: NVIDIA GPU required
- Container runtime: NVIDIA Container Toolkit required
- Base image: `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`
- Python runtime: `3.11`
- Isaac Sim runtime: pip package `isaacsim[all,extscache]==5.1.0`
- Isaac Lab source checkout: `/workspace/IsaacLab` pinned to `v2.3.2`

## Release Layout

- `Dockerfile`
- `.dockerignore`
- `run_agent.sh`
- `configs/docker/e2e_batch_release.yaml`
- `configs/docker/e2e_batch_representative.yaml`
- `configs/docker/e2e_batch_smoke.yaml`
- `configs/docker/data_collection_release.yaml`
- `configs/docker/isaaclab_agent_release.yaml`
- `scripts/audit_release_repo.py`
- `scripts/validate_docker_release.py`

Runtime artifacts are written to `/workspace/artifacts` instead of the source tree `outputs/`.

## Required Environment Variables (one of)

Provide either OpenAI platform or Azure OpenAI credentials.

**OpenAI platform (recommended):**
- `OPENAI_API_KEY`

**Azure OpenAI:**
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_BASE_URL`

## Optional Environment Variables

- `OPENAI_BASE_URL` (custom endpoint)
- `HF_TOKEN`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `ISAACLAB_PATH`
- `ADC_PATH`

No API keys or `.env` files are baked into the image.

## Host Setup

The host must have Docker Engine and the NVIDIA Container Toolkit installed.

- Docker Engine: `https://docs.docker.com/engine/install/ubuntu/`
- NVIDIA Container Toolkit: `https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`

After installation, open a new login session (or run `newgrp docker`) and verify:

```bash
docker info
```

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 nvidia-smi
```

## Run (Simplest Method)

Provide a natural language task description and the full pipeline runs automatically.

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="sk-..." \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

Results are saved to `./artifacts/` on the host (mapped to `/workspace/artifacts` inside the container).

## Quick Run (JSON Input)

Build the image and run with a JSON input file.

```bash
# 1. Build the image
docker build -t simgen-agent .

# 2. Run the container
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="sk-..." \
  simgen-agent \
  data/input_sample.json results/output.json
```

Input JSON format:
```json
{
  "tasks": [
    {"task_description": "Stack the blocks inside the tray on the table", "robot": "franka"}
  ],
  "config": {"target_success": 1, "max_attempts": 3}
}
```

Results are written to `results/output.json` as structured JSON.

## `run_agent.sh` Modes

`run_agent.sh` supports two interfaces.

**Default mode** (positional args):
```bash
run_agent.sh <input_file.json> [output_file.json]
```

**Advanced pipeline mode** (`--mode` flag):
- `e2e-batch`: Runs `scripts/run_e2e_batch.py` for release or representative batch execution.
- `isaac-lab`: Runs `scripts/run_isaac_lab.py` for single task code generation and runtime validation.
- `data-collection`: Runs `scripts/run_data_collection.py` for single task data collection.

## Build

Build a new image. Submodules must be initialized first.

```bash
cd /path/to/Simulation-Generation-Agent

# Initialize submodules (first time only)
git submodule update --init --recursive

# Build
docker build -t simgen-agent .
```

## Help / Healthcheck

Verify the entrypoint contract and Python runtime.

```bash
docker run --rm simgen-agent --help
```

```bash
docker run --rm --gpus all \
  --entrypoint python \
  simgen-agent \
  -c "import isaaclab, isaacsim; print(isaaclab.__file__); print(isaacsim.__file__)"
```

Run a small smoke batch to verify artifact isolation and basic batch flow:

```bash
docker run --rm --gpus all \
  -e OPENAI_API_KEY \
  -v $(pwd)/artifacts:/workspace/artifacts \
  simgen-agent \
  --mode e2e-batch \
  --config /workspace/Simulation-Generation-Agent/configs/docker/e2e_batch_smoke.yaml
```

Run a static security/hygiene audit:

```bash
python3 scripts/audit_release_repo.py
```

## Representative Validation

Representative validation covers build, entrypoint, image audit, single task execution, representative batch, and resume probe.

```bash
python3 scripts/validate_docker_release.py
```

To run only the representative batch directly:

```bash
docker run --rm --gpus all \
  -e OPENAI_API_KEY \
  -v $(pwd)/artifacts:/workspace/artifacts \
  simgen-agent \
  --mode e2e-batch \
  --config /workspace/Simulation-Generation-Agent/configs/docker/e2e_batch_representative.yaml
```

## Full Validation

Full validation includes the complete release batch and full soak test.

```bash
python3 scripts/validate_docker_release.py --run-full-soak
```

To run the full batch directly:

```bash
docker run --rm --gpus all \
  -e OPENAI_API_KEY \
  -e HF_TOKEN \
  -v $(pwd)/artifacts:/workspace/artifacts \
  simgen-agent \
  --mode e2e-batch \
  --config /workspace/Simulation-Generation-Agent/configs/docker/e2e_batch_release.yaml \
  --resume
```

## Operating Notes

- `run_agent.sh` is the single public release entry point.
- Default mode is `e2e-batch`.
- Default artifact root is `/workspace/artifacts`.
- `FrankaPickPlaceMug` and `FrankaCabinetBlocks` remain `enabled: false` in the Docker release config.
- `hf.upload` is `false` by default in release configs. Enable via config override only.
