# Docker Guide

[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](README.docker.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](docs/ko/docker.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](docs/zh/docker.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](docs/ja/docker.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](docs/de/docker.md)

This guide walks you through building and running the Simulation-Generation-Agent inside a Docker container, step by step. No local IsaacLab installation is needed.

---

## Prerequisites

Before you begin, make sure you have:

- [ ] **NVIDIA GPU** with recent drivers installed
- [ ] **Docker Engine** (v26.0+)
- [ ] **NVIDIA Container Toolkit**
- [ ] **OpenAI API Key** (or Azure OpenAI credentials)

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU Memory | 8 GB | 16+ GB |
| Disk Space | 50 GB free | 100+ GB free |
| RAM | 16 GB | 32+ GB |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |

### Install Docker Engine

Follow the official guide: [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

After installation, verify:
```bash
docker --version
docker info
```

### Install NVIDIA Container Toolkit

Follow the official guide: [NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

After installation, verify GPU access from Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 nvidia-smi
```

You should see your GPU listed. If you get a permission error, try `newgrp docker` or log out and back in.

---

## Step 1: Get the Source Code

```bash
git clone --recurse-submodules <repo-url>
cd Simulation-Generation-Agent
```

> If you already cloned without `--recurse-submodules`, run:
> ```bash
> git submodule update --init --recursive
> ```

---

## Step 2: Configure API Keys

```bash
cp .env.example .env
```

Open `.env` and set your API key (for local runs):
```
OPENAI_API_KEY=sk-your-key-here
```

> **Docker users**: You don't need to create `.env` for Docker runs. API keys are passed directly via `-e` flags at runtime. The `.env` file is only needed for local (non-Docker) execution.

---

## Step 3: Build the Docker Image

This repository includes a `Dockerfile` that sets up the complete environment automatically.

```bash
docker build -t simgen-agent .
```

> **Disk space**: ~50GB required. **First build**: ~30-60 minutes. Subsequent builds are much faster due to Docker layer caching.

### What the Dockerfile does

The `Dockerfile` in the repository root builds a self-contained image with:

| Layer | What's installed |
|-------|-----------------|
| Base | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |
| Python | 3.11 (venv at `/opt/isaaclab-env`) |
| Isaac Sim | 5.1.0 (pip from `pypi.nvidia.com`) |
| IsaacLab | v2.3.2 (source build from GitHub) |
| PyTorch | 2.7.0 (CUDA 12.8) |
| Project | `requirements.txt` + all source code |
| Entry point | `ENTRYPOINT ["./run_agent.sh"]` |

The container's entry point is `run_agent.sh`, so when you run:
```bash
docker run simgen-agent "Stack the blocks..."
```
it automatically executes `run_agent.sh "Stack the blocks..."` inside the container.

Verify the image was built:
```bash
docker images | grep simgen-agent
```

---

## Step 4: Run the Pipeline

### Option A: Natural Language Input (Simplest)

Just describe what the robot should do:

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/results:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="your-key-here" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

With options:
```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/results:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="your-key-here" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

### Option B: JSON Input

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  data/input_sample.json results/output.json
```

The default `data/input_sample.json` contains a sample FrankaStackTray task. You can create your own:
```json
{
  "tasks": [
    {"task_description": "Pick up the cube and place it on the target", "robot": "franka"}
  ],
  "config": {"target_success": 1, "max_attempts": 3}
}
```

### Option C: Individual Stages (Advanced)

```bash
# Stage 2 only: Generate IsaacLab environment code
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode isaac-lab --task tasks/franka/stack/franka_stack.yaml

# Stage 3 only: Data collection
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode data-collection --task tasks/franka/stack/franka_stack.yaml

# Batch execution (multiple tasks)
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode e2e-batch --config configs/docker/e2e_batch_smoke.yaml
```

### Using Azure OpenAI

To use Azure OpenAI instead of the OpenAI platform, pass Azure credentials via `-e` flags:

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e AZURE_OPENAI_API_KEY="your-azure-key" \
  -e AZURE_OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/v1/" \
  -e AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
  -e AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `AZURE_OPENAI_BASE_URL` | Yes | Azure endpoint with `/openai/v1/` suffix |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure resource endpoint (used by Stage 1) |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Yes | Deployed model name in Azure (used by Stage 1) |

> You need either `OPENAI_API_KEY` or the Azure set above. Not both.

### Choosing a Model

The default model is `gpt-5-mini`. To override, pass the `OPENAI_MODEL` environment variable:

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="your-key" \
  -e OPENAI_MODEL="gpt-4o" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

### Show Help

```bash
docker run --rm simgen-agent --help
```

### What You'll See

When the pipeline runs, you'll see clean progress updates in your terminal:

```
🚀 RAPIDS Pipeline — "Stack the blocks inside the tray on the table"
   Robot: franka | Target: 5 episodes

  ✅ Stage 1: NL → YAML                              1m 12s
  ✅ Stage 2: YAML → IsaacLab                         9m 44s
  ✅ Stage 3: Data Collection (5/5 episodes)           7m 30s

──────────────────────────────────────────────────────
  📊 Result: ✅ completed
  ⏱️  Total: 18m 26s
  🔤 Tokens: 123,008 (10 API calls)
  💰 Cost: ~$0.15
  📄 Output: results/output.json
  📁 Logs: outputs/challenge_run_20260402_151823/
──────────────────────────────────────────────────────
```

A spinning indicator (⏳) appears while each stage is running. Detailed logs are saved to files — only clean status lines appear in the terminal.

---

## Step 5: View Results

Results are saved to `./artifacts/` on your host machine (mapped from `/workspace/artifacts` inside the container).

```bash
ls artifacts/
```

Typical output structure:
```
artifacts/
├── isaaclab/              # Stage 2: generated environment code
│   └── 20260402_*/        # timestamped run directory
│       ├── env_cfg.py     # environment configuration
│       ├── run_env.py     # environment runner
│       ├── result.json    # success/failure status
│       └── debug/         # screenshots (front, top, wrist)
└── data_collection/       # Stage 3: collected episodes
    └── TaskName_*/
        ├── collection_results.json
        ├── raw_dataset/   # episode data
        └── videos/        # recorded videos
```

For JSON input mode, results are also written to `results/output.json`.

---

## Troubleshooting

### GPU not detected

```
Error: could not select device driver "nvidia"
```

**Fix**: Install or reinstall the NVIDIA Container Toolkit:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Build fails at IsaacSim install

The IsaacSim pip package is ~15GB. If it fails due to network issues:
```bash
# Retry with no-cache
docker build --no-cache -t simgen-agent .
```

### Out of memory (OOM)

IsaacLab simulation requires at least **8GB GPU memory** (Stage 2 only) or **16GB+** (full pipeline with Stage 3). If you get OOM errors:
- Close other GPU-intensive applications
- Reduce `--episodes` to 1
- Use `--mode isaac-lab` to test Stage 2 alone first

### API key not working

```
Either OPENAI_API_KEY or AZURE_OPENAI_API_KEY must be set.
```

**Fix**: Make sure the key is correctly passed with `-e`:
```bash
# Option 1: Inline
-e OPENAI_API_KEY="sk-your-key"

# Option 2: From .env file
-e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)"

# Option 3: Export first
export OPENAI_API_KEY="sk-your-key"
docker run ... -e OPENAI_API_KEY ...
```

---

## Reference

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (or Azure) | OpenAI platform API key |
| `AZURE_OPENAI_API_KEY` | Yes (or OpenAI) | Azure OpenAI API key |
| `AZURE_OPENAI_BASE_URL` | With Azure | Azure endpoint URL |
| `OPENAI_BASE_URL` | No | Custom OpenAI-compatible endpoint |
| `HF_TOKEN` | No | HuggingFace token (for dataset upload) |
| `OPENAI_MODEL` | No | Override default model (default: gpt-5-mini) |

### `run_agent.sh` Modes

| Mode | Description |
|------|-------------|
| *(positional args)* | Full pipeline: NL or JSON input |
| `--mode e2e-batch` | Batch execution of multiple tasks |
| `--mode isaac-lab` | Stage 2 only: environment code generation |
| `--mode data-collection` | Stage 3 only: data collection |

### Image Specifications

| Component | Version |
|-----------|---------|
| Base image | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |
| Python | 3.11 |
| Isaac Sim | 5.1.0 (pip) |
| IsaacLab | v2.3.2 (source) |
| PyTorch | 2.7.0 (CUDA 12.8) |

### Validation Scripts

```bash
# Static audit (secrets, paths, structure)
python3 scripts/audit_release_repo.py

# Full Docker validation (build + run + test)
python3 scripts/validate_docker_release.py

# Full soak test (all tasks)
python3 scripts/validate_docker_release.py --run-full-soak
```
