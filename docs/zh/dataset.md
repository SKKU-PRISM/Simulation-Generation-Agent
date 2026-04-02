# 数据集工作流程

[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../dataset.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](../ko/dataset.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](dataset.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](../ja/dataset.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](../de/dataset.md)

本指南涵盖从运行流水线到将训练就绪的数据集上传至 HuggingFace 的完整工作流程。

---

## 概览

```
run_agent.sh "task description"
    |
    v
outputs/data_collection/<run>/raw_dataset/     <-- Stage 3 输出
    |
    v  scripts/export_dataset.py
outputs/exported_datasets/<name>/              <-- 标准化训练模式
    |
    v  scripts/preprocess_dataset.py
outputs/preprocessed_datasets/<name>/          <-- train.jsonl / val.jsonl
    |
    v  scripts/convert_lerobot_dataset.py
outputs/lerobot_datasets/<repo_id>/            <-- LeRobot v3.0 格式 (Parquet)
    |
    v  scripts/publish_lerobot_dataset.py
https://huggingface.co/datasets/<org>/<name>   <-- HuggingFace Hub
```

---

## 步骤 1：运行流水线

```bash
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

当 Stage 3 完成后，结果将保存至：
- **本地**：`outputs/data_collection/<TaskName>_<timestamp>/`
- **Docker**：`/workspace/artifacts/data_collection/`（挂载至 `./artifacts/`）

### 检查输出

```bash
ls outputs/data_collection/
# FrankaStackTray_20260402_151823/

ls outputs/data_collection/FrankaStackTray_20260402_151823/
# collection_results.json   raw_dataset/   videos/

cat outputs/data_collection/FrankaStackTray_20260402_151823/collection_results.json
```

`collection_results.json` 文件包含：
- `pipeline_completed`：数据采集是否完成
- `geometry_successful_episodes`：通过几何验证的回合数
- `total_episodes`：总尝试回合数

### 原始数据集结构

```
raw_dataset/
├── metadata.json              # 机器人配置、自由度、FPS、相机
├── episodes/
│   ├── episode_000000/
│   │   ├── states.npy         # (T, N_dof) 关节位置
│   │   ├── actions.npy        # (T, N_dof) 指令位置
│   │   ├── tcp_world_xyzrpy.npy  # (T, 6) 世界坐标系下的 TCP
│   │   ├── tcp_robot_xyzrpy.npy  # (T, 6) 机器人基坐标系下的 TCP
│   │   ├── gripper_state.npy  # (T, 1) 夹爪状态
│   │   ├── goal_robot_xyzrpy.npy # (T, 6) 每帧的目标姿态
│   │   ├── skills.json        # 每帧的技能元数据
│   │   └── images/
│   │       ├── top/           # 000000.png, 000001.png, ...
│   │       └── wrist/
│   └── episode_000001/
│       └── ...
```

---

## 步骤 2：导出（可选）

将原始数据标准化为训练友好的模式，包含一致的关节归一化处理。

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets
```

选项：
- `--schema adc_compatible`（默认）：将关节归一化至 [-100, 100] 范围，使用 ADC 兼容的字段名
- `--schema canonical_training`：简洁模式，包含显式的 TCP 当前/目标姿态
- `--no-link-images`：复制图像而非使用符号链接

---

## 步骤 3：预处理（可选）

创建逐帧训练清单，并进行训练/验证集划分。

```bash
python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets \
  --success-only \
  --train-ratio 0.9
```

输出：
- `train.jsonl` / `val.jsonl` — 包含文件路径的逐帧样本
- `manifest.json` — 模式和统计信息

---

## 步骤 4：转换为 LeRobot 格式

将原始数据转换为 [LeRobot v3.0](https://github.com/huggingface/lerobot) 格式（Apache Parquet）。

### 前置要求

```bash
pip install "lerobot>=0.4.0,<0.5.0"
```

### 转换

```bash
python3 scripts/convert_lerobot_dataset.py \
  outputs/data_collection/<run>/raw_dataset \
  --repo-id local/franka_stack_sim \
  --output-root outputs/lerobot_datasets
```

输出结构：
```
outputs/lerobot_datasets/local/franka_stack_sim/
├── meta/
│   └── info.json           # LeRobot 元数据、特征、帧数
├── data/
│   └── chunk-0000/
│       └── file-00000.parquet
└── videos/                 # （如果启用了视频录制）
```

### 验证

```bash
python3 scripts/check_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack_sim \
  --repo-id local/franka_stack_sim
```

验证检查项目：
- 必需特征：`observation.state`、`observation.gripper_state`、`observation.tcp.robot_xyzrpy`、`action`、`skill.goal_position.robot_xyzrpy`
- Parquet 分片完整性
- 帧数 > 0
- `meta/info.json` 存在性

---

## 步骤 5：上传至 HuggingFace

### 前置要求

```bash
pip install huggingface_hub
```

设置 HuggingFace 令牌：
```bash
export HF_TOKEN="hf_your_token_here"
# 或者：huggingface-cli login
```

### 发布

```bash
python3 scripts/publish_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack_sim \
  --repo-id your-org/franka-stack-sim \
  --local-repo-id local/franka_stack_sim \
  --private
```

选项：
- `--private`：创建为私有数据集（默认：公开）
- `--token-env HF_TOKEN`：用于认证的环境变量（默认）
- `--token <token>`：直接传入令牌

该脚本将：
1. 验证本地数据集
2. 创建 HuggingFace 数据集仓库（如果不存在）
3. 使用 `upload_large_folder()` 上传所有文件
4. 打印包含仓库 URL 的 JSON 报告

---

## 快速参考

### 一站式流程（从原始数据到 HuggingFace）

```bash
# 1. 运行流水线
./run_agent.sh "Stack the blocks" --robot franka --episodes 5

# 2. 找到输出目录
RUN_DIR=$(ls -td outputs/data_collection/*/ | head -1)

# 3. 转换为 LeRobot 格式
python3 scripts/convert_lerobot_dataset.py \
  ${RUN_DIR}/raw_dataset \
  --repo-id local/franka_stack \
  --output-root outputs/lerobot_datasets

# 4. 验证
python3 scripts/check_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack \
  --repo-id local/franka_stack

# 5. 上传
python3 scripts/publish_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack \
  --repo-id your-org/franka-stack --private
```

### 批量流水线（自动化）

`e2e-batch` 模式可以自动完成完整流程，包括 LeRobot 转换和 HuggingFace 上传：

```bash
./run_agent.sh --mode e2e-batch --config configs/docker/e2e_batch_release.yaml
```

在批量配置 YAML 中设置上传参数：
```yaml
hf:
  upload: true
  namespace: your-org
  dataset_name: franka-sim-dataset
  private: true
  token_env: HF_TOKEN
```

---

## LeRobot 数据集特征

转换后的数据集中每帧包含以下特征：

| 特征 | 形状 | 描述 |
|------|------|------|
| `observation.state` | (N_dof,) | 关节位置 |
| `observation.gripper_state` | (1,) | 夹爪状态 |
| `observation.tcp.world_xyzrpy` | (6,) | 世界坐标系下的 TCP 姿态 |
| `observation.tcp.robot_xyzrpy` | (6,) | 机器人基坐标系下的 TCP 姿态 |
| `action` | (N_dof,) | 指令关节位置 |
| `skill.natural_language` | (1,) | 技能描述 |
| `skill.type` | (1,) | 技能类型（抓取、放置等） |
| `skill.progress` | (1,) | 技能进度 [0, 1] |
| `skill.goal_position.joint` | (N_dof,) | 目标关节位置 |
| `skill.goal_position.world_xyzrpy` | (6,) | 世界坐标系下的目标姿态 |
| `skill.goal_position.robot_xyzrpy` | (6,) | 机器人基坐标系下的目标姿态 |
| `skill.goal_position.gripper` | (1,) | 目标夹爪状态 |
| `observation.images.{cam}` | (480, 640, 3) | 相机图像（top、wrist） |

**各机器人的 N_dof：** Franka=9、UR10e=12、OpenARM=9、SO-101=6

---

## 相关文档

- 流水线架构：[architecture.md](architecture.md)
- CLI 使用方法：[usage.md](usage.md)
- 安装指南：[getting_started.md](getting_started.md)
