# 安装指南

[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../getting_started.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](../ko/getting_started.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](getting_started.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](../ja/getting_started.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](../de/getting_started.md)

目标读者：首次运行本仓库的用户
详细 CLI 选项和结果解读请参阅：`docs/usage.md`

## 1. 通用安装

```bash
git clone --recurse-submodules <repo-url>
cd Simulation-Generation-Agent

pip install -e .
cp .env.example .env
```

### 子模块（AutoDataCollector）

本项目包含 `external/AutoDataCollector` 作为 git 子模块。它在数据收集（阶段 3）中用于评判提示词和 IK 工具。

使用 `--recurse-submodules` 克隆时会自动获取。如果遗漏，可手动初始化：

```bash
git submodule update --init --recursive
ls external/AutoDataCollector/  # 应能看到文件
```

必需的 `.env`：

```bash
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1/
```

可选设置：

```bash
# VLM 后端（用于环境验证 + 回合成功判定）
# ANTHROPIC_API_KEY=...
# GOOGLE_API_KEY=...

# 路径
# ISAACLAB_PATH=~/workspace/IsaacLab

# Token 用量追踪
# TOKEN_USAGE_FILE=outputs/token_usage.jsonl
```

## 2. IsaacLab 安装（主路径）

```bash
cd ~/workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

如何将项目指向您的 IsaacLab 位置：

```bash
export ISAACLAB_PATH=~/workspace/IsaacLab
```

或直接在 `configs/isaaclab_agent_config.yaml` 中设置 `isaaclab.path`。

验证连接：

```bash
# 本地（conda 环境）
conda run -n env_isaaclab --no-capture-output \
  python -c "import isaaclab; print('IsaacLab import OK')"

# 在 Docker 中使用 venv 而非 conda，因此验证方式如下
# python -c "import isaaclab; print('IsaacLab import OK')"

python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run
```

## 3. Isaac Sim + MCP 安装（可选）

Isaac Sim 仅用于视觉验证路径。

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

成功启动后，TCP socket 将在 `localhost:8766` 上监听。

验证连接：

```bash
python3 tests/test_components.py connection
```

## 4. 数据收集安装（可选）

数据收集运行在 IsaacLab 之上。

```bash
pip install -e ".[data-collection]"
pip install pin
```

验证：

```bash
python3 -c "from src.agent.data_collection.adc_imports import is_adc_available; print(is_adc_available())"
python3 -c "import pinocchio; print('Pinocchio OK')"
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --help
```

## 5. 任务规范代理安装（NL -> YAML）

任务规范代理是将自然语言输入转换为 YAML 任务规范的阶段 1 流水线。

```bash
pip install langchain langchain-openai langchain-community faiss-cpu sentence-transformers
```

向量存储会在首次运行时自动创建在 `data/vector_store/`（索引 tasks/ 目录中的现有 YAML）。

验证：

```bash
cd scripts/task_spec_agent
python3 -c "from rag_match_yaml_generator import RAGYAMLGenerator; print('RAG OK')"
cd ../..
```

独立执行：

```bash
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube" --robot franka --output outputs/test_task.yaml
```

## 6. 首次成功运行的推荐顺序

### 交互模式（推荐首次使用者）

```bash
./run_agent.sh
```

不带参数启动即可打开交互式 TUI 仪表板。使用 F1 配置设置，F2 浏览运行历史，直接输入任务描述即可运行。

### CLI 模式 -- 一行自然语言

```bash
./run_agent.sh "Stack the blocks inside the tray on the table"
```

提供自然语言任务描述作为参数，整个 NL→YAML→IsaacLab→DataCollection 流水线将自动运行。

运行时终端会显示进度信息：
```
🚀 RAPIDS Pipeline — "Stack the blocks inside the tray on the table"
   Robot: franka | Target: 1 episodes

  ✅ Stage 1: NL → YAML                              1m 12s
  ✅ Stage 2: YAML → IsaacLab                         9m 44s
  ✅ Stage 3: Data Collection (1/1 episodes)           7m 30s

──────────────────────────────────────────────────────
  📊 Result: ✅ completed
  ⏱️  Total: 18m 26s
  🔤 Tokens: 123,008 (10 API calls)
  💰 Cost: ~$0.15
  📄 Output: results/output.json
──────────────────────────────────────────────────────
```



带选项的执行示例：
选项：
```bash
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

### 完整流水线（13 个任务批量）

```bash
bash scripts/run_full_test.sh
```

对 13 个 Franka 任务依次运行全部 3 个阶段：
1. **NL -> YAML**：`task_spec_agent` 将自然语言转换为 YAML
2. **YAML -> IsaacLab**：LLM 生成环境 Python 代码并验证执行
3. **CaP -> 数据**：LLM 生成技能代码，通过 IK 执行，判定成功，收集数据

### 仅 IsaacLab（仅阶段 2）

```bash
./run_agent.sh --mode isaac-lab --task tasks/franka/stack/franka_stack.yaml
```

成功标准：
- `outputs/isaaclab/<run_dir>/env_cfg.py`
- `outputs/isaaclab/<run_dir>/result.json`

### 仅数据收集（仅阶段 3）

```bash
./run_agent.sh --mode data-collection --task tasks/franka/stack/franka_stack.yaml
```

成功标准：
- `outputs/data_collection/<run_dir>/collection_results.json`
- `outputs/data_collection/<run_dir>/raw_dataset/`

### 训练准备路径

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets

python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

## 7. 使用 Docker 运行

无需本地环境设置，可直接使用 Docker 容器运行。

前提条件：[Docker Engine](https://docs.docker.com/engine/install/ubuntu/) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
# 初始化子模块 + 构建镜像
git submodule update --init --recursive
docker build -t simgen-agent .

# 使用自然语言输入运行完整流水线
docker run --rm --gpus all \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --robot franka --episodes 5

# JSON 输入
docker run --rm --gpus all \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  data/input_sample.json results/output.json

# 显示用法
docker run --rm simgen-agent --help
```

> **输出路径差异**：在 Docker 中，结果保存到 `/workspace/artifacts`（挂载到主机的 `./artifacts/`）。本地运行时，结果保存到 `outputs/`。要在 Docker 内使用与本地相同的路径，请添加 `-e SIMGEN_ARTIFACT_ROOT=/workspace/Simulation-Generation-Agent/outputs`。

详细 Docker 指南：[README.docker.md](../../README.docker.md)

## 8. 下一步阅读

- 流水线架构详情：[docs/architecture.md](architecture.md)
- CLI 选项 / 输出结构 / 结果解读：[docs/usage.md](usage.md)
