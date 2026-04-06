# Usage
[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](usage.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](ko/usage.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](zh/usage.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](ja/usage.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](de/usage.md)

Audience: Users who actually run the CLI  
What this document covers: Representative commands, key options, output structure, interpreting results  
Installation and environment setup: `docs/getting_started.md`

This document is the **source-of-truth for how to run and interpret results**. Internal implementation details and schema background are separated into other documents.
Pipeline architecture: [docs/architecture.md](architecture.md)

## Interactive TUI Mode

Launch `run_agent.sh` without arguments to open the interactive terminal dashboard:

```bash
./run_agent.sh
```

The TUI provides:
- **Task input** — type a natural language task description and press Enter to run the full pipeline
- **Settings (F1)** — configure pipeline mode, LLM provider/model, robot, episodes, paths, and advanced options
- **Run History (F2)** — browse past runs with arrow keys, press Enter to view detailed results including environment scores and dataset info
- **Help (F3)** — keyboard shortcuts and slash commands

Slash commands: `/config`, `/history`, `/help`, `/quit`

In Docker, use `-it` for interactive mode:
```bash
docker run -it --rm --gpus all \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  --env-file .env \
  simgen-agent
```

---

## run_agent.sh — Full Pipeline Execution (CLI Mode)

Provide a natural language task description as argument for non-interactive execution:

```bash
# Natural language input
./run_agent.sh "Stack the blocks inside the tray on the table"
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5

# JSON input
./run_agent.sh data/input_sample.json results/output.json
```

### Options

| Option | Description |
| --- | --- |
| `--robot <type>` | Robot type: franka, ur10e, openarm, so101 (default: franka) |
| `--episodes <n>` | Target number of successful episodes (default: 1) |
| `--max-attempts <n>` | Maximum number of attempts (default: 3) |
| `--output-root <dir>` | Base directory for all pipeline outputs (default: `outputs/`) |

### Running in Docker

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --episodes 5
```

> **Output path**: Inside Docker, results are saved to `/workspace/artifacts`. Map it to the host using the `-v` option. For local execution, results are saved to `outputs/`.
>
> To change the output directory, use `--output-root`:
> ```bash
> ./run_agent.sh "Stack the blocks" --output-root /path/to/my/outputs
> ```
>
> After completion, the dataset is at `<output-root>/data_collection/<TaskName>_<timestamp>/raw_dataset/`.

### Results

Results are recorded as structured JSON in `results/output.json`:

```json
{
  "status": "completed",
  "tasks": [{
    "name": "StackTheBlocksInsideTheTrayOnThe",
    "steps": {
      "nl_to_yaml": {"success": true},
      "yaml_to_isaaclab": {"success": true},
      "data_collection": {"success": true, "success_episodes": 5}
    }
  }]
}
```

---

## 0. Task Spec Agent (NL → YAML)

Converts a natural language task description into a structured YAML task specification (Pipeline Stage 1).

> **Note**: Using `run_agent.sh "natural language task"` automatically runs the entire Stage 1→2→3 pipeline.
> Below is how to run Stage 1 individually.

### Representative Commands

```bash
# Basic usage
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube and place it on the target" \
  --robot franka --output task.yaml

# Template-based generation without RAG
python3 scripts/task_spec_agent/task_spec_agent.py "Stack blocks" --no-rag

# Use a different LLM provider
python3 scripts/task_spec_agent/task_spec_agent.py "Sort the colored blocks into matching colored bins" --provider huggingface

# Verbose logging
python3 scripts/task_spec_agent/task_spec_agent.py "Reach the goal" --verbose
```

### Key Options

| Option | Description |
| --- | --- |
| `--robot {franka,openarm,ur10,so101}` | Target robot (default: franka) |
| `--output <path>` | YAML save path (outputs to stdout if not specified) |
| `--provider {azure,huggingface,bedrock}` | LLM provider override (uses config default if not specified, default=openai) |
| `--no-rag` | Template-based YAML generation instead of RAG |
| `--verbose` | DEBUG level logging |

### Internal Processing Steps

```
Natural language input
  → NL Parser (extract actions, objects, locations)
  → Task Decomposer (decompose into atomic action sequences)
  → Feasibility Validator (verify physical feasibility for the robot)
  → RAG YAML Generator (FAISS vector search → nearest task YAML matching)
  → task.yaml
```

## 1. IsaacLab Pipeline (Stage 2)

> **Running via run_agent.sh**: `./run_agent.sh --mode isaac-lab --task <yaml>`
> Below is how to run the Python script directly.

### Representative Commands

```bash
# Generate + Run
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml

# Generate only
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run

# Generate + Run + Evaluate
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate

# Re-evaluate existing output
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml \
  --eval-only outputs/isaaclab/frankastack_20260219_160916

# Standalone evaluator
python3 scripts/evaluate.py \
  outputs/isaaclab/frankastack_20260219_160916 \
  tasks/franka/stack/franka_stack.yaml

# Batch
python3 scripts/run_isaac_lab.py --batch tasks/franka/
```

### Key Options

| Option | Description |
| --- | --- |
| `--dry-run` | Generate code only, skip IsaacLab execution |
| `--evaluate` | Run evaluator after success |
| `--eval-only <dir>` | Evaluate existing generated output only |
| `--batch <dir>` | Batch process all YAMLs in a directory |
| `--output-dir <dir>` | Use a different output root instead of default `outputs/isaaclab` |
| `--config <path>` | Agent config override |

### Output Structure

```text
outputs/isaaclab/<task_slug>_<timestamp>/
├── env_cfg.py
├── run_env.py
├── mdp/
├── .success_marker
├── error_attempt_*.txt
├── debug/                    # Environment screenshots (front/top/wrist)
├── result.json               # Execution result + includes scene_verification
└── eval_report.json          # When --evaluate is used
```

## 2. Isaac Sim Pipeline (Local Development Environment Only)

> **Note**: Isaac Sim MCP visual verification is only available when Isaac Sim Desktop is running locally. It is not supported in Docker environments.

### Representative Commands

```bash
# Auto build + screenshot + VLM evaluation
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --backend auto

# Build + capture once without VLM
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --skip-vlm

# Iteration/threshold override
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml \
  --backend azure --max-iterations 3 --threshold 85
```

### Key Options

| Option | Description |
| --- | --- |
| `--skip-vlm` | Perform a single build/capture without VLM evaluation |
| `--backend {auto,azure,claude,gemini,ollama,mock}` | VLM backend selection |
| `--max-iterations <n>` | Maximum iteration count override |
| `--threshold <n>` | Success threshold score override |
| `--output-dir <dir>` | Output root override |
| `--config <path>` | Alternative to `configs/pipeline_config.yaml` |

### Output Structure

```text
outputs/isaac_sim/<task_slug>_<timestamp>/
├── iter_01.png
├── iter_02.png
├── rgb_0000.png
├── metadata.txt
└── run_report.json
```

## 3. Data Collection (Stage 3)

> **Running via run_agent.sh**: `./run_agent.sh --mode data-collection --task <yaml>`
> Below is how to run the Python script directly.

### Representative Commands

```bash
# Auto-generate env then collect
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml

# Use existing env
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/frankastack_20260219_160916

# Success episode target
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --target-success 5 --max-attempts 25

# Geometric verification only, without VLM
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --no-vlm-judge

# Batch
python3 scripts/run_data_collection.py --batch tasks/franka/ --episodes 10
```

### Key Options

| Option | Description |
| --- | --- |
| `--env-dir <path>` | Use an existing IsaacLab output directory |
| `--episodes <n>` | Maximum episode count override |
| `--target-success <n>` | Target number of successful episodes |
| `--max-attempts <n>` | Upper limit on total attempts |
| `--repo-id <id>` | Dataset ID |
| `--fps <n>` | Recording FPS override |
| `--no-vlm-judge` | Disable VLM judgment |
| `--gui` | Run with GUI instead of headless |
| `--config <path>` | Data collection config override |
| `-v`, `--verbose` | Verbose logging |

### Output Structure

```text
outputs/data_collection/<TaskName>_<timestamp>/
├── collect_data.py
├── pipeline_config.json
├── cap_runs/
├── debug_initial_*.png
├── raw_dataset/
│   ├── episodes/
│   └── metadata.json
├── collection_results.json
├── COLLECTION_COMPLETE_MARKER
└── <repo_id>/                   # Optional LeRobot conversion output
```

### Interpreting Results

The first values to check in `collection_results.json` are the following.

| Key | Meaning |
| --- | --- |
| `pipeline_completed` | Whether the collection/cleanup pipeline completed to the end |
| `target_met` | Whether the target number of successful episodes was achieved |
| `successful_episodes` | Number of successful episodes |
| `total_episodes` | Number of episodes actually attempted/saved |
| `raw_dataset` | Raw dataset path |

Interpretation rules:

- `success` is currently an alias for `pipeline_completed`.
- `pipeline_completed=true` does not necessarily mean `target_met=true`.
- The default completion criterion is the creation of `raw_dataset/`. LeRobot conversion may be skipped depending on the environment.

## 4. Dataset Export / LeRobot Conversion / HuggingFace Upload

After Stage 3 completes, the raw dataset is at `outputs/data_collection/<run>/raw_dataset/`. From here you can export, convert to LeRobot format, and upload to HuggingFace.

For the **complete step-by-step guide** with output structure details, validation, and batch automation, see **[dataset.md](dataset.md)**.

### Quick reference

```bash
# Export to normalized schema
python3 scripts/export_dataset.py \
  outputs/data_collection/<run>/raw_dataset \
  --source-type sim_raw --output-dir outputs/exported_datasets

# Preprocess into train/val splits
python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets --success-only

# Convert to LeRobot v3.0 format (requires: pip install "lerobot>=0.4.0,<0.5.0")
python3 scripts/convert_lerobot_dataset.py \
  outputs/data_collection/<run>/raw_dataset \
  --repo-id local/my_dataset --output-root outputs/lerobot_datasets

# Validate
python3 scripts/check_lerobot_dataset.py \
  outputs/lerobot_datasets/local/my_dataset --repo-id local/my_dataset

# Upload to HuggingFace (requires: pip install huggingface_hub, export HF_TOKEN=...)
python3 scripts/publish_lerobot_dataset.py \
  outputs/lerobot_datasets/local/my_dataset \
  --repo-id your-org/my_dataset --private
```

## 5. Recommended Workflow Order

### NL → Dataset (Full Pipeline)

```bash
# 1. NL → YAML
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube and stack it" \
  --robot franka --output outputs/generated_task.yaml

# 2. YAML → IsaacLab environment code
python3 scripts/run_isaac_lab.py outputs/generated_task.yaml --evaluate

# 3. Data collection
python3 scripts/run_data_collection.py outputs/generated_task.yaml \
  --env-dir outputs/isaaclab/<run_dir> --target-success 10

# 4. Export / Preprocess
python3 scripts/export_dataset.py outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw --output-dir outputs/exported_datasets
python3 scripts/preprocess_dataset.py outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

### Existing YAML → Dataset

```bash
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --env-dir outputs/isaaclab/<run_dir>
python3 scripts/export_dataset.py outputs/data_collection/<run_dir>/raw_dataset --source-type sim_raw --output-dir outputs/exported_datasets
python3 scripts/preprocess_dataset.py outputs/exported_datasets/<export_dir> --output-dir outputs/preprocessed_datasets
```

Add `scripts/run_isaac_sim.py` only when visual verification is needed.

## 6. Full Pipeline (NL → Video)

`run_full_test.sh` is an integration script that sequentially runs Stage 1→2→3 for 13 Franka tasks.

```bash
bash scripts/run_full_test.sh
```

Internally, for each task:
1. `task_spec_agent.py` — Natural language → YAML generation
2. `run_isaac_lab.py` — YAML → IsaacLab environment code generation/verification
3. `run_data_collection.py` — CaP code generation → execution → success determination → data collection

### Output Structure

```text
outputs/test_run_<timestamp>/
├── summary.txt                  # Overall task result summary
├── token_usage.jsonl            # API token usage (when TOKEN_USAGE_FILE is set)
├── FrankaLift/
│   ├── step1_nl_to_yaml.log
│   ├── step2_isaaclab.log
│   ├── step3_cap_execution.log
│   └── task.yaml
├── FrankaStack/
│   └── ...
└── ...
```

### Token Usage Tracking

```bash
export TOKEN_USAGE_FILE=outputs/token_usage.jsonl
export TOKEN_USAGE_LOG=1  # Real-time console logging
bash scripts/run_full_test.sh
```

## 7. E2E Batch Pipeline

Batch processes environment generation + data collection + export + LeRobot conversion for multiple tasks based on a config file.

### Representative Commands

```bash
# Batch execution (Docker)
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml

# Resume after interruption
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml --resume

# Quiet logging
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml -q
```

### Docker / run_agent.sh

```bash
# Inside Docker (default mode: e2e-batch)
run_agent.sh --mode e2e-batch --resume

# Single task
run_agent.sh --mode isaac-lab --task tasks/franka/lift/franka_lift.yaml

# Data collection
run_agent.sh --mode data-collection --task tasks/franka/lift/franka_lift.yaml -- --episodes 10
```

### Config Structure

The E2E batch config YAML consists of 4 sections:

| Section | Role |
| --- | --- |
| `run` | output_root, resume, cleanup, export settings |
| `collection` | LLM/VLM model, max_attempts, timeout, VLM judge usage |
| `hf` | HuggingFace Hub upload settings (optional) |
| `tasks` | Task list (YAML path, target demo count, enabled flag) |

### Key Options

| Option | Description |
| --- | --- |
| `--resume` | Skip previously successful tasks and re-run only failed/incomplete tasks |
| `-q`, `--quiet` | INFO level logging (default: DEBUG) |

### Output Structure

```text
<output_root>/
├── config_snapshot.yaml         # Copy of the config used for execution
├── task_runs/                   # Per-task collection results
├── task_reports/                # Per-task JSON reports
├── successful_raw/              # Raw dataset of successful episodes only
├── exported/                    # Export results
├── preprocessed/                # Preprocess results
├── lerobot/                     # LeRobot conversion results
├── batch_report.md              # Markdown summary report
└── batch_report.json            # JSON report
```

## Related Documents

- Pipeline architecture: `docs/architecture.md`
- Installation: `docs/getting_started.md`
