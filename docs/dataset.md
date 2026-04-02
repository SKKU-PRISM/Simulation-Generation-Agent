# Dataset Workflow

[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](dataset.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](ko/dataset.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](zh/dataset.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](ja/dataset.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](de/dataset.md)

This guide covers the complete workflow from running the pipeline to uploading a training-ready dataset to HuggingFace.

---

## Overview

```
run_agent.sh "task description"
    |
    v
outputs/data_collection/<run>/raw_dataset/     <-- Stage 3 output
    |
    v  scripts/export_dataset.py
outputs/exported_datasets/<name>/              <-- Normalized training schema
    |
    v  scripts/preprocess_dataset.py
outputs/preprocessed_datasets/<name>/          <-- train.jsonl / val.jsonl
    |
    v  scripts/convert_lerobot_dataset.py
outputs/lerobot_datasets/<repo_id>/            <-- LeRobot v3.0 format (Parquet)
    |
    v  scripts/publish_lerobot_dataset.py
https://huggingface.co/datasets/<org>/<name>   <-- HuggingFace Hub
```

---

## Step 1: Run the Pipeline

```bash
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

When Stage 3 completes, results are saved to:
- **Local**: `outputs/data_collection/<TaskName>_<timestamp>/`
- **Docker**: `/workspace/artifacts/data_collection/` (mounted to `./artifacts/`)

### Check the output

```bash
ls outputs/data_collection/
# FrankaStackTray_20260402_151823/

ls outputs/data_collection/FrankaStackTray_20260402_151823/
# collection_results.json   raw_dataset/   videos/

cat outputs/data_collection/FrankaStackTray_20260402_151823/collection_results.json
```

The `collection_results.json` file contains:
- `pipeline_completed`: whether the collection finished
- `geometry_successful_episodes`: episodes that passed geometric verification
- `total_episodes`: total episodes attempted

### Raw dataset structure

```
raw_dataset/
├── metadata.json              # Robot config, DOF, FPS, cameras
├── episodes/
│   ├── episode_000000/
│   │   ├── states.npy         # (T, N_dof) joint positions
│   │   ├── actions.npy        # (T, N_dof) commanded positions
│   │   ├── tcp_world_xyzrpy.npy  # (T, 6) TCP in world frame
│   │   ├── tcp_robot_xyzrpy.npy  # (T, 6) TCP in robot base frame
│   │   ├── gripper_state.npy  # (T, 1) gripper state
│   │   ├── goal_robot_xyzrpy.npy # (T, 6) goal pose per frame
│   │   ├── skills.json        # Per-frame skill metadata
│   │   └── images/
│   │       ├── top/           # 000000.png, 000001.png, ...
│   │       └── wrist/
│   └── episode_000001/
│       └── ...
```

---

## Step 2: Export (Optional)

Normalizes raw data into a training-friendly schema with consistent joint normalization.

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets
```

Options:
- `--schema adc_compatible` (default): Normalizes joints to [-100, 100] range, ADC-compatible field names
- `--schema canonical_training`: Clean schema with explicit TCP current/goal poses
- `--no-link-images`: Copy images instead of symlinking

---

## Step 3: Preprocess (Optional)

Creates per-frame training manifests with train/val splits.

```bash
python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets \
  --success-only \
  --train-ratio 0.9
```

Output:
- `train.jsonl` / `val.jsonl` — per-frame samples with file paths
- `manifest.json` — schema and statistics

---

## Step 4: Convert to LeRobot Format

Converts raw data to [LeRobot v3.0](https://github.com/huggingface/lerobot) format (Apache Parquet).

### Prerequisites

```bash
pip install "lerobot>=0.4.0,<0.5.0"
```

### Convert

```bash
python3 scripts/convert_lerobot_dataset.py \
  outputs/data_collection/<run>/raw_dataset \
  --repo-id local/franka_stack_sim \
  --output-root outputs/lerobot_datasets
```

Output structure:
```
outputs/lerobot_datasets/local/franka_stack_sim/
├── meta/
│   └── info.json           # LeRobot metadata, features, frame count
├── data/
│   └── chunk-0000/
│       └── file-00000.parquet
└── videos/                 # (if video recording enabled)
```

### Validate

```bash
python3 scripts/check_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack_sim \
  --repo-id local/franka_stack_sim
```

The validation checks:
- Required features: `observation.state`, `observation.gripper_state`, `observation.tcp.robot_xyzrpy`, `action`, `skill.goal_position.robot_xyzrpy`
- Parquet shard integrity
- Frame count > 0
- `meta/info.json` presence

---

## Step 5: Upload to HuggingFace

### Prerequisites

```bash
pip install huggingface_hub
```

Set your HuggingFace token:
```bash
export HF_TOKEN="hf_your_token_here"
# Or: huggingface-cli login
```

### Publish

```bash
python3 scripts/publish_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack_sim \
  --repo-id your-org/franka-stack-sim \
  --local-repo-id local/franka_stack_sim \
  --private
```

Options:
- `--private`: Create as a private dataset (default: public)
- `--token-env HF_TOKEN`: Environment variable for authentication (default)
- `--token <token>`: Pass token directly

The script will:
1. Validate the local dataset
2. Create the HuggingFace dataset repo (if it doesn't exist)
3. Upload all files using `upload_large_folder()`
4. Print a JSON report with the repo URL

---

## Quick Reference

### All-in-one (from raw to HuggingFace)

```bash
# 1. Run pipeline
./run_agent.sh "Stack the blocks" --robot franka --episodes 5

# 2. Find the output
RUN_DIR=$(ls -td outputs/data_collection/*/ | head -1)

# 3. Convert to LeRobot
python3 scripts/convert_lerobot_dataset.py \
  ${RUN_DIR}/raw_dataset \
  --repo-id local/franka_stack \
  --output-root outputs/lerobot_datasets

# 4. Validate
python3 scripts/check_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack \
  --repo-id local/franka_stack

# 5. Upload
python3 scripts/publish_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack \
  --repo-id your-org/franka-stack --private
```

### Batch pipeline (automated)

The `e2e-batch` mode can automate the full flow including LeRobot conversion and HuggingFace upload:

```bash
./run_agent.sh --mode e2e-batch --config configs/docker/e2e_batch_release.yaml
```

Configure upload in the batch config YAML:
```yaml
hf:
  upload: true
  namespace: your-org
  dataset_name: franka-sim-dataset
  private: true
  token_env: HF_TOKEN
```

---

## LeRobot Dataset Features

Each frame in the converted dataset contains:

| Feature | Shape | Description |
|---------|-------|-------------|
| `observation.state` | (N_dof,) | Joint positions |
| `observation.gripper_state` | (1,) | Gripper state |
| `observation.tcp.world_xyzrpy` | (6,) | TCP pose in world frame |
| `observation.tcp.robot_xyzrpy` | (6,) | TCP pose in robot base frame |
| `action` | (N_dof,) | Commanded joint positions |
| `skill.natural_language` | (1,) | Skill description |
| `skill.type` | (1,) | Skill type (pick, place, etc.) |
| `skill.progress` | (1,) | Skill progress [0, 1] |
| `skill.goal_position.joint` | (N_dof,) | Target joint positions |
| `skill.goal_position.world_xyzrpy` | (6,) | Target pose in world frame |
| `skill.goal_position.robot_xyzrpy` | (6,) | Target pose in robot base frame |
| `skill.goal_position.gripper` | (1,) | Target gripper state |
| `observation.images.{cam}` | (480, 640, 3) | Camera images (top, wrist) |

**N_dof by robot:** Franka=9, UR10e=12, OpenARM=9, SO-101=6

---

## Related

- Pipeline architecture: [architecture.md](architecture.md)
- CLI usage: [usage.md](usage.md)
- Installation: [getting_started.md](getting_started.md)
