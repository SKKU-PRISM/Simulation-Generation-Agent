# 使用方法
[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../usage.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](../ko/usage.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](usage.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](../ja/usage.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](../de/usage.md)

目标读者：实际运行 CLI 的用户
本文档涵盖：典型命令、关键选项、输出结构、结果解读
安装和环境设置请参阅：`docs/getting_started.md`

本文档是**如何运行和解读结果的权威参考**。内部实现细节和 schema 背景请参阅其他文档。
流水线架构：[docs/architecture.md](architecture.md)

## 交互式 TUI 模式

不带参数启动 `run_agent.sh` 即可打开交互式终端仪表板：

```bash
./run_agent.sh
```

TUI 提供以下功能：
- **任务输入** — 输入自然语言任务描述，按 Enter 运行完整流水线
- **设置 (F1)** — 配置流水线模式、LLM 提供商/模型、机器人、回合数、路径及高级选项
- **运行历史 (F2)** — 使用方向键浏览历史记录，按 Enter 查看详细结果（环境评分、数据集信息）
- **帮助 (F3)** — 快捷键和斜杠命令

斜杠命令：`/config`、`/history`、`/help`、`/quit`

在 Docker 中使用 `-it` 启动交互模式：
```bash
docker run -it --rm --gpus all \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  --env-file .env \
  simgen-agent
```

---

## run_agent.sh — 完整流水线执行（CLI 模式）

提供自然语言任务描述作为参数，进行非交互式执行：

```bash
# 自然语言输入
./run_agent.sh "Stack the blocks inside the tray on the table"
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5

# JSON 输入
./run_agent.sh data/input_sample.json results/output.json
```

### 选项

| 选项 | 描述 |
| --- | --- |
| `--robot <type>` | 机器人类型：franka、ur10e、openarm、so101（默认：franka） |
| `--episodes <n>` | 目标成功回合数（默认：1） |
| `--max-attempts <n>` | 最大尝试次数（默认：3） |

### 在 Docker 中运行

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --episodes 5
```

> **输出路径**：在 Docker 中，结果保存到 `/workspace/artifacts`。使用 `-v` 选项将其映射到主机。本地执行时，结果保存到 `outputs/`。

### 结果

结果以结构化 JSON 记录在 `results/output.json` 中：

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

## 0. 任务规范代理（NL → YAML）

将自然语言任务描述转换为结构化 YAML 任务规范（流水线阶段 1）。

> **注意**：使用 `run_agent.sh "自然语言任务"` 会自动运行整个阶段 1→2→3 流水线。
> 以下是单独运行阶段 1 的方法。

### 典型命令

```bash
# 基本用法
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube and place it on the target" \
  --robot franka --output task.yaml

# 不使用 RAG 的模板生成
python3 scripts/task_spec_agent/task_spec_agent.py "Stack blocks" --no-rag

# 使用其他 LLM 提供商
python3 scripts/task_spec_agent/task_spec_agent.py "Sort the colored blocks into matching colored bins" --provider huggingface

# 详细日志
python3 scripts/task_spec_agent/task_spec_agent.py "Reach the goal" --verbose
```

### 关键选项

| 选项 | 描述 |
| --- | --- |
| `--robot {franka,openarm,ur10,so101}` | 目标机器人（默认：franka） |
| `--output <path>` | YAML 保存路径（未指定时输出到 stdout） |
| `--provider {azure,huggingface,bedrock}` | LLM 提供商覆盖（未指定时使用配置默认值，默认=openai） |
| `--no-rag` | 使用模板生成 YAML 而非 RAG |
| `--verbose` | DEBUG 级别日志 |

### 内部处理步骤

```
自然语言输入
  → NL 解析器（提取动作、对象、位置）
  → 任务分解器（分解为原子动作序列）
  → 可行性验证器（验证机器人的物理可行性）
  → RAG YAML 生成器（FAISS 向量搜索 → 最近任务 YAML 匹配）
  → task.yaml
```

## 1. IsaacLab 流水线（阶段 2）

> **通过 run_agent.sh 运行**：`./run_agent.sh --mode isaac-lab --task <yaml>`
> 以下是直接运行 Python 脚本的方法。

### 典型命令

```bash
# 生成 + 运行
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml

# 仅生成
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run

# 生成 + 运行 + 评估
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate

# 重新评估现有输出
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml \
  --eval-only outputs/isaaclab/frankastack_20260219_160916

# 独立评估器
python3 scripts/evaluate.py \
  outputs/isaaclab/frankastack_20260219_160916 \
  tasks/franka/stack/franka_stack.yaml

# 批量
python3 scripts/run_isaac_lab.py --batch tasks/franka/
```

### 关键选项

| 选项 | 描述 |
| --- | --- |
| `--dry-run` | 仅生成代码，跳过 IsaacLab 执行 |
| `--evaluate` | 成功后运行评估器 |
| `--eval-only <dir>` | 仅评估现有生成的输出 |
| `--batch <dir>` | 批量处理目录中的所有 YAML |
| `--output-dir <dir>` | 使用其他输出根目录而非默认的 `outputs/isaaclab` |
| `--config <path>` | 代理配置覆盖 |

### 输出结构

```text
outputs/isaaclab/<task_slug>_<timestamp>/
├── env_cfg.py
├── run_env.py
├── mdp/
├── .success_marker
├── error_attempt_*.txt
├── debug/                    # 环境截图（前方/顶部/腕部）
├── result.json               # 执行结果 + 包含 scene_verification
└── eval_report.json          # 使用 --evaluate 时生成
```

## 2. Isaac Sim 流水线（仅本地开发环境）

> **注意**：Isaac Sim MCP 视觉验证仅在本地运行 Isaac Sim Desktop 时可用。不支持 Docker 环境。

### 典型命令

```bash
# 自动构建 + 截图 + VLM 评估
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --backend auto

# 构建 + 捕获一次，不使用 VLM
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --skip-vlm

# 迭代/阈值覆盖
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml \
  --backend azure --max-iterations 3 --threshold 85
```

### 关键选项

| 选项 | 描述 |
| --- | --- |
| `--skip-vlm` | 执行一次构建/捕获，不进行 VLM 评估 |
| `--backend {auto,azure,claude,gemini,ollama,mock}` | VLM 后端选择 |
| `--max-iterations <n>` | 最大迭代次数覆盖 |
| `--threshold <n>` | 成功阈值分数覆盖 |
| `--output-dir <dir>` | 输出根目录覆盖 |
| `--config <path>` | 替代 `configs/pipeline_config.yaml` |

### 输出结构

```text
outputs/isaac_sim/<task_slug>_<timestamp>/
├── iter_01.png
├── iter_02.png
├── rgb_0000.png
├── metadata.txt
└── run_report.json
```

## 3. 数据收集（阶段 3）

> **通过 run_agent.sh 运行**：`./run_agent.sh --mode data-collection --task <yaml>`
> 以下是直接运行 Python 脚本的方法。

### 典型命令

```bash
# 自动生成环境然后收集
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml

# 使用现有环境
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/frankastack_20260219_160916

# 成功回合目标
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --target-success 5 --max-attempts 25

# 仅几何验证，不使用 VLM
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --no-vlm-judge

# 批量
python3 scripts/run_data_collection.py --batch tasks/franka/ --episodes 10
```

### 关键选项

| 选项 | 描述 |
| --- | --- |
| `--env-dir <path>` | 使用现有 IsaacLab 输出目录 |
| `--episodes <n>` | 最大回合数覆盖 |
| `--target-success <n>` | 目标成功回合数 |
| `--max-attempts <n>` | 总尝试次数上限 |
| `--repo-id <id>` | 数据集 ID |
| `--fps <n>` | 录制 FPS 覆盖 |
| `--no-vlm-judge` | 禁用 VLM 判定 |
| `--gui` | 使用 GUI 运行而非无头模式 |
| `--config <path>` | 数据收集配置覆盖 |
| `-v`, `--verbose` | 详细日志 |

### 输出结构

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
└── <repo_id>/                   # 可选 LeRobot 转换输出
```

### 结果解读

`collection_results.json` 中首先需要检查的值如下。

| 键 | 含义 |
| --- | --- |
| `pipeline_completed` | 收集/清理流水线是否执行完毕 |
| `target_met` | 是否达到目标成功回合数 |
| `successful_episodes` | 成功回合数 |
| `total_episodes` | 实际尝试/保存的回合数 |
| `raw_dataset` | 原始数据集路径 |

解读规则：

- `success` 目前是 `pipeline_completed` 的别名。
- `pipeline_completed=true` 不一定意味着 `target_met=true`。
- 默认完成标准是创建 `raw_dataset/`。LeRobot 转换可能因环境不同而被跳过。

## 4. 数据集导出 / 预处理

### 导出

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets
```

默认 schema 为 `adc_compatible`。

### 预处理

```bash
python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

输出：

- `manifest.json`
- `samples.jsonl`
- `train.jsonl`
- `val.jsonl`

在默认的 `adc_compatible` 路径中，`observation.gripper_state`、`observation.tcp.robot_xyzrpy` 和 `skill.goal_position.robot_xyzrpy` 包含在训练输入 manifest 中。

默认导出使用 `adc_compatible` schema，`canonical_training` 用于扩展分析。

### 可选：本地 LeRobot 转换

```bash
python3 scripts/convert_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --repo-id local/franka_stack_sim
python3 scripts/check_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/local/franka_stack_sim \
  --repo-id local/franka_stack_sim
```

需要在主机 Python 环境中安装 `lerobot` 包。

### 可选：发布到 Hub

```bash
python3 scripts/publish_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/local/franka_stack_sim \
  --repo-id <org>/<dataset_name> \
  --local-repo-id local/franka_stack_sim \
  --private
```

默认认证使用 `HF_TOKEN` 环境变量或已有的 `huggingface-cli login` 会话。

## 5. 推荐工作流顺序

### NL → 数据集（完整流水线）

```bash
# 1. NL → YAML
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube and stack it" \
  --robot franka --output outputs/generated_task.yaml

# 2. YAML → IsaacLab 环境代码
python3 scripts/run_isaac_lab.py outputs/generated_task.yaml --evaluate

# 3. 数据收集
python3 scripts/run_data_collection.py outputs/generated_task.yaml \
  --env-dir outputs/isaaclab/<run_dir> --target-success 10

# 4. 导出 / 预处理
python3 scripts/export_dataset.py outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw --output-dir outputs/exported_datasets
python3 scripts/preprocess_dataset.py outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

### 现有 YAML → 数据集

```bash
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --env-dir outputs/isaaclab/<run_dir>
python3 scripts/export_dataset.py outputs/data_collection/<run_dir>/raw_dataset --source-type sim_raw --output-dir outputs/exported_datasets
python3 scripts/preprocess_dataset.py outputs/exported_datasets/<export_dir> --output-dir outputs/preprocessed_datasets
```

仅在需要视觉验证时添加 `scripts/run_isaac_sim.py`。

## 6. 完整流水线（NL → 视频）

`run_full_test.sh` 是一个集成脚本，对 13 个 Franka 任务依次运行阶段 1→2→3。

```bash
bash scripts/run_full_test.sh
```

内部对每个任务执行：
1. `task_spec_agent.py` — 自然语言 → YAML 生成
2. `run_isaac_lab.py` — YAML → IsaacLab 环境代码生成/验证
3. `run_data_collection.py` — CaP 代码生成 → 执行 → 成功判定 → 数据收集

### 输出结构

```text
outputs/test_run_<timestamp>/
├── summary.txt                  # 总体任务结果摘要
├── token_usage.jsonl            # API token 用量（设置 TOKEN_USAGE_FILE 时）
├── FrankaLift/
│   ├── step1_nl_to_yaml.log
│   ├── step2_isaaclab.log
│   ├── step3_cap_execution.log
│   └── task.yaml
├── FrankaStack/
│   └── ...
└── ...
```

### Token 用量追踪

```bash
export TOKEN_USAGE_FILE=outputs/token_usage.jsonl
export TOKEN_USAGE_LOG=1  # 实时控制台日志
bash scripts/run_full_test.sh
```

## 7. E2E 批量流水线

基于配置文件批量处理多个任务的环境生成 + 数据收集 + 导出 + LeRobot 转换。

### 典型命令

```bash
# 批量执行（Docker）
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml

# 中断后恢复
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml --resume

# 静默日志
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml -q
```

### Docker / run_agent.sh

```bash
# Docker 内部（默认模式：e2e-batch）
run_agent.sh --mode e2e-batch --resume

# 单个任务
run_agent.sh --mode isaac-lab --task tasks/franka/lift/franka_lift.yaml

# 数据收集
run_agent.sh --mode data-collection --task tasks/franka/lift/franka_lift.yaml -- --episodes 10
```

### 配置结构

E2E 批量配置 YAML 由 4 个部分组成：

| 部分 | 作用 |
| --- | --- |
| `run` | output_root、resume、cleanup、export 设置 |
| `collection` | LLM/VLM 模型、max_attempts、timeout、VLM judge 使用 |
| `hf` | HuggingFace Hub 上传设置（可选） |
| `tasks` | 任务列表（YAML 路径、目标演示数量、启用标志） |

### 关键选项

| 选项 | 描述 |
| --- | --- |
| `--resume` | 跳过之前成功的任务，仅重新运行失败/未完成的任务 |
| `-q`, `--quiet` | INFO 级别日志（默认：DEBUG） |

### 输出结构

```text
<output_root>/
├── config_snapshot.yaml         # 执行时使用的配置副本
├── task_runs/                   # 每个任务的收集结果
├── task_reports/                # 每个任务的 JSON 报告
├── successful_raw/              # 仅成功回合的原始数据集
├── exported/                    # 导出结果
├── preprocessed/                # 预处理结果
├── lerobot/                     # LeRobot 转换结果
├── batch_report.md              # Markdown 摘要报告
└── batch_report.json            # JSON 报告
```

## 相关文档

- 流水线架构：`docs/architecture.md`
- 安装：`docs/getting_started.md`
