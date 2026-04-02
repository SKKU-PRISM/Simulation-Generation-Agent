# Installation Guide

> **[한국어 (Korean)](ko/getting_started.md)**

Audience: Users running this repo for the first time
For detailed CLI options and result interpretation: `docs/usage.md`

## 1. Common Installation

```bash
git clone --recurse-submodules <repo-url>
cd Simulation-Generation-Agent

pip install -e .
cp .env.example .env
```

### Submodule (AutoDataCollector)

This project includes `external/AutoDataCollector` as a git submodule. It is referenced during Data Collection (Stage 3) for judge prompts and IK utilities.

Cloning with `--recurse-submodules` will fetch it automatically. If it was missed, initialize manually:

```bash
git submodule update --init --recursive
ls external/AutoDataCollector/  # Files should be present
```

Required `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1/
```

Optional settings:

```bash
# VLM backends (for environment verification + episode success judgment)
# ANTHROPIC_API_KEY=...
# GOOGLE_API_KEY=...

# Paths
# ISAACLAB_PATH=~/workspace/IsaacLab

# Token usage tracking
# TOKEN_USAGE_FILE=outputs/token_usage.jsonl
```

## 2. IsaacLab Installation (Primary Path)

```bash
cd ~/workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

How to point the project to your IsaacLab location:

```bash
export ISAACLAB_PATH=~/workspace/IsaacLab
```

Or set `isaaclab.path` directly in `configs/isaaclab_agent_config.yaml`.

Verify the connection:

```bash
# Local (conda environment)
conda run -n env_isaaclab --no-capture-output \
  python -c "import isaaclab; print('IsaacLab import OK')"

# In Docker, venv is used instead of conda, so verify as follows
# python -c "import isaaclab; print('IsaacLab import OK')"

python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run
```

## 3. Isaac Sim + MCP Installation (Optional)

Isaac Sim is only needed for the visual verification path.

```bash
cd ~/workspace
git clone https://github.com/isaac-sim/isaac-sim-mcp.git
```

```bash
cd ~/workspace/isaac-sim
./isaac-sim.streaming.sh \
  --ext-folder /home/$USER/workspace/isaac-sim-mcp \
  --enable isaac.sim.mcp_extension
```

On successful startup, a TCP socket will be listening on `localhost:8766`.

Verify the connection:

```bash
python3 tests/test_components.py connection
```

## 4. Data Collection Installation (Optional)

Data Collection runs on top of IsaacLab.

```bash
pip install -e ".[data-collection]"
pip install pin
```

Verify:

```bash
python3 -c "from src.agent.data_collection.adc_imports import is_adc_available; print(is_adc_available())"
python3 -c "import pinocchio; print('Pinocchio OK')"
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --help
```

## 5. Task Spec Agent Installation (NL -> YAML)

The Task Spec Agent is a Stage 1 pipeline that converts natural language input into YAML task specifications.

```bash
pip install langchain langchain-openai langchain-community faiss-cpu sentence-transformers
```

The vector store is automatically created at `data/vector_store/` on the first run (indexing existing YAMLs from the tasks/ directory).

Verify:

```bash
cd scripts/task_spec_agent
python3 -c "from rag_match_yaml_generator import RAGYAMLGenerator; print('RAG OK')"
cd ../..
```

Standalone execution:

```bash
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube" --robot franka --output outputs/test_task.yaml
```

## 6. Recommended Order for Your First Successful Run

### Simplest Execution -- A Single Line of Natural Language

```bash
./run_agent.sh "Stack the blocks inside the tray on the table"
```

Simply provide a natural language task description, and the entire NL->YAML->IsaacLab->DataCollection pipeline runs automatically.

Options:
```bash
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

### Full Pipeline (Batch of 13 Tasks)

```bash
bash scripts/run_full_test.sh
```

Runs all 3 stages sequentially for 13 Franka tasks:
1. **NL -> YAML**: `task_spec_agent` converts natural language to YAML
2. **YAML -> IsaacLab**: LLM generates environment Python code and verifies execution
3. **CaP -> Data**: LLM generates skill code, executes via IK, judges success, and collects data

### IsaacLab Only (Stage 2 Only)

```bash
./run_agent.sh --mode isaac-lab --task tasks/franka/stack/franka_stack.yaml
```

Success criteria:
- `outputs/isaaclab/<run_dir>/env_cfg.py`
- `outputs/isaaclab/<run_dir>/result.json`

### Data Collection Only (Stage 3 Only)

```bash
./run_agent.sh --mode data-collection --task tasks/franka/stack/franka_stack.yaml
```

Success criteria:
- `outputs/data_collection/<run_dir>/collection_results.json`
- `outputs/data_collection/<run_dir>/raw_dataset/`

### Training Preparation Path

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets

python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

## 7. Running with Docker

You can run immediately using a Docker container without any local environment setup.

Prerequisites: [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
# Initialize submodule + build image
git submodule update --init --recursive
docker build -t simgen-agent .

# Run the full pipeline with natural language input
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --robot franka --episodes 5

# JSON input
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  data/input_sample.json results/output.json

# Show usage
docker run --rm simgen-agent --help
```

> **Output path difference**: In Docker, results are saved to `/workspace/artifacts` (mounted to `./artifacts/` on the host). Locally, results are saved to `outputs/`. To use the same path as local inside Docker, add `-e SIMGEN_ARTIFACT_ROOT=/workspace/Simulation-Generation-Agent/outputs`.

Detailed Docker guide: [README.docker.md](../README.docker.md)

## 8. What to Read Next

- Pipeline architecture details: [docs/architecture.md](architecture.md)
- CLI options / output structure / result interpretation: [docs/usage.md](usage.md)
