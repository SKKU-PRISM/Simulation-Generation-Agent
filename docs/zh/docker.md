# Docker 指南

[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../../README.docker.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](../ko/docker.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](docker.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](../ja/docker.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](../de/docker.md)

本指南将逐步引导您在 Docker 容器中构建和运行 Simulation-Generation-Agent。无需在本地安装 IsaacLab。

---

## 前提条件

开始之前，请确保您已具备：

- [ ] 已安装最新驱动的 **NVIDIA GPU**
- [ ] **Docker Engine**（v26.0+）
- [ ] **NVIDIA Container Toolkit**
- [ ] **OpenAI API 密钥**（或 Azure OpenAI 凭据）

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU 显存 | 8 GB | 16+ GB |
| 磁盘空间 | 50 GB 可用 | 100+ GB 可用 |
| 内存 | 16 GB | 32+ GB |
| 操作系统 | Ubuntu 22.04 | Ubuntu 22.04 |

### 安装 Docker Engine

请参阅官方指南：[在 Ubuntu 上安装 Docker Engine](https://docs.docker.com/engine/install/ubuntu/)

安装后验证：
```bash
docker --version
docker info
```

### 安装 NVIDIA Container Toolkit

请参阅官方指南：[NVIDIA Container Toolkit 安装](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

安装后验证 Docker 中的 GPU 访问：
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 nvidia-smi
```

您应该能看到 GPU 列表。如果出现权限错误，请尝试 `newgrp docker` 或注销后重新登录。

---

## 第 1 步：获取源代码

```bash
git clone --recurse-submodules <repo-url>
cd Simulation-Generation-Agent
```

> 如果您已在未使用 `--recurse-submodules` 的情况下克隆，请运行：
> ```bash
> git submodule update --init --recursive
> ```

---

## 第 2 步：配置 API 密钥

```bash
cp .env.example .env
```

打开 `.env` 并设置您的 API 密钥（用于本地运行）：
```
OPENAI_API_KEY=sk-your-key-here
```

> **Docker 用户**：Docker 运行时无需创建 `.env` 文件。API 密钥在运行时通过 `-e` 标志直接传递。`.env` 文件仅在本地（非 Docker）执行时需要。

---

## 第 3 步：构建 Docker 镜像

本仓库包含一个 `Dockerfile`，可自动设置完整的运行环境。

```bash
docker build -t simgen-agent .
```

> **磁盘空间**：约需 50GB。**首次构建**：约 30-60 分钟。由于 Docker 层缓存，后续构建会快得多。

### Dockerfile 的作用

仓库根目录中的 `Dockerfile` 构建一个包含以下内容的独立镜像：

| Layer | What's installed |
|-------|-----------------|
| Base | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |
| Python | 3.11 (venv at `/opt/isaaclab-env`) |
| Isaac Sim | 5.1.0 (pip from `pypi.nvidia.com`) |
| IsaacLab | v2.3.2 (source build from GitHub) |
| PyTorch | 2.7.0 (CUDA 12.8) |
| Project | `requirements.txt` + all source code |
| Entry point | `ENTRYPOINT ["./run_agent.sh"]` |

容器的入口点是 `run_agent.sh`，因此当您运行：
```bash
docker run simgen-agent "Stack the blocks..."
```
它会在容器内自动执行 `run_agent.sh "Stack the blocks..."`。

验证镜像已构建：
```bash
docker images | grep simgen-agent
```

---

## 第 4 步：运行流水线

### 选项 A：自然语言输入（最简单）

只需描述机器人应执行的操作：

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -v $(pwd)/results:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="your-key-here" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

带选项：
```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -v $(pwd)/results:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="your-key-here" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

### 选项 B：JSON 输入

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  data/input_sample.json results/output.json
```

默认的 `data/input_sample.json` 包含一个示例 FrankaStackTray 任务。您可以创建自己的：
```json
{
  "tasks": [
    {"task_description": "Pick up the cube and place it on the target", "robot": "franka"}
  ],
  "config": {"target_success": 1, "max_attempts": 3}
}
```

### 选项 C：单独阶段（高级）

```bash
# 仅 Stage 2：生成 IsaacLab 环境代码
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode isaac-lab --task tasks/franka/stack/franka_stack.yaml

# 仅 Stage 3：数据收集
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode data-collection --task tasks/franka/stack/franka_stack.yaml

# 批量执行（多个任务）
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode e2e-batch --config configs/docker/e2e_batch_smoke.yaml
```

### 使用 Azure OpenAI

如需使用 Azure OpenAI 而非 OpenAI 平台，通过 `-e` 标志传递 Azure 凭据：

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e AZURE_OPENAI_API_KEY="your-azure-key" \
  -e AZURE_OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/v1/" \
  -e AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
  -e AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

| 变量 | 必需 | 说明 |
|------|------|------|
| `AZURE_OPENAI_API_KEY` | 是 | Azure OpenAI API 密钥 |
| `AZURE_OPENAI_BASE_URL` | 是 | Azure 端点（带 `/openai/v1/` 后缀） |
| `AZURE_OPENAI_ENDPOINT` | 是 | Azure 资源端点（Stage 1 使用） |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | 是 | Azure 中部署的模型名称（Stage 1 使用） |

> 只需 `OPENAI_API_KEY` 或上述 Azure 变量组其中之一。

### 选择模型

默认模型为 `gpt-5`。可通过 `OPENAI_MODEL` 环境变量覆盖：

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="your-key" \
  -e OPENAI_MODEL="gpt-4o" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

### 交互式 TUI 模式

在 Docker 内启动交互式终端仪表板。需要 `-it` 标志以访问 TTY：

```bash
docker run -it --rm --gpus all \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  --env-file .env \
  simgen-agent
```

使用 F1 配置设置，F2 浏览运行历史，或直接输入任务描述。

### 显示帮助

```bash
docker run --rm simgen-agent --help
```

### 运行时的显示内容

流水线运行时，终端会显示简洁的进度更新：

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

每个阶段运行时会显示旋转指示器。详细日志保存到文件中，终端只显示简洁的状态行。

---

## 第 5 步：查看结果

结果保存在宿主机的 `./artifacts/` 中（从容器内部的 `/workspace/artifacts` 映射）。

```bash
ls artifacts/
```

典型输出结构：
```
artifacts/
├── isaaclab/              # Stage 2：生成的环境代码
│   └── 20260402_*/        # 带时间戳的运行目录
│       ├── env_cfg.py     # 环境配置
│       ├── run_env.py     # 环境运行器
│       ├── result.json    # 成功/失败状态
│       └── debug/         # 截图（前方、顶部、手腕）
└── data_collection/       # Stage 3：收集的回合
    └── TaskName_*/
        ├── collection_results.json
        ├── raw_dataset/   # 回合数据
        └── videos/        # 录制的视频
```

对于 JSON 输入模式，结果也会写入 `results/output.json`。

### 自定义输出目录

默认情况下，管道输出保存到 `outputs/`。要更改路径，使用 `--output-root`：

```bash
# 本地
./run_agent.sh "Stack the blocks" --output-root /path/to/my/outputs

# Docker
docker run --rm --gpus all \
  -v /path/to/my/outputs:/workspace/custom_outputs \
  -v $(pwd)/results:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  "Stack the blocks" --output-root /workspace/custom_outputs
```

### 数据集位置

成功运行后，原始数据集保存在：
```
outputs/data_collection/<TaskName>_<timestamp>/raw_dataset/
├── episodes/
│   └── episode_000000/
│       ├── actions.npy          # 关节动作
│       ├── states.npy           # 关节状态
│       ├── gripper_state.npy    # 夹爪开合
│       ├── tcp_world_xyzrpy.npy # TCP 位姿（世界坐标系）
│       ├── tcp_robot_xyzrpy.npy # TCP 位姿（机器人坐标系）
│       ├── skills.json          # 技能序列日志
│       └── images/
│           └── front_cam/*.png  # 相机帧
└── metadata.json                # 机器人信息、自由度、关节名称
```

---

## 故障排除

### 未检测到 GPU

```
Error: could not select device driver "nvidia"
```

**解决方法**：安装或重新安装 NVIDIA Container Toolkit：
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### IsaacSim 安装时构建失败

IsaacSim pip 包约为 15GB。如果因网络问题失败：
```bash
# 不使用缓存重试
docker build --no-cache -t simgen-agent .
```

### 内存不足 (OOM)

IsaacLab 仿真至少需要 **8GB GPU 显存**（仅 Stage 2）或 **16GB+**（包含 Stage 3 的完整流水线）。如果出现 OOM 错误：
- 关闭其他 GPU 密集型应用程序
- 将 `--episodes` 减少为 1
- 使用 `--mode isaac-lab` 先单独测试 Stage 2

### API 密钥无效

```
Either OPENAI_API_KEY or AZURE_OPENAI_API_KEY must be set.
```

**解决方法**：确保密钥通过 `-e` 正确传递：
```bash
# 方法 1：内联
-e OPENAI_API_KEY="sk-your-key"

# 方法 2：从 .env 文件
-e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)"

# 方法 3：先 export
export OPENAI_API_KEY="sk-your-key"
docker run ... -e OPENAI_API_KEY ...
```

---

## 参考

### 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | 是（或 Azure） | OpenAI 平台 API 密钥 |
| `AZURE_OPENAI_API_KEY` | 是（或 OpenAI） | Azure OpenAI API 密钥 |
| `AZURE_OPENAI_BASE_URL` | 使用 Azure 时 | Azure 端点 URL |
| `OPENAI_BASE_URL` | 否 | 自定义 OpenAI 兼容端点 |
| `HF_TOKEN` | 否 | HuggingFace 令牌（用于数据集上传） |
| `OPENAI_MODEL` | 否 | 覆盖默认模型（默认：gpt-5） |

### `run_agent.sh` 模式

| 模式 | 说明 |
|------|------|
| *（位置参数）* | 完整流水线：自然语言或 JSON 输入 |
| `--mode e2e-batch` | 多任务批量执行 |
| `--mode isaac-lab` | 仅 Stage 2：环境代码生成 |
| `--mode data-collection` | 仅 Stage 3：数据收集 |

### 镜像规格

| 组件 | 版本 |
|------|------|
| 基础镜像 | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |
| Python | 3.11 |
| Isaac Sim | 5.1.0 (pip) |
| IsaacLab | v2.3.2 (source) |
| PyTorch | 2.7.0 (CUDA 12.8) |

### 验证脚本

```bash
# 静态审计（密钥、路径、结构）
python3 scripts/audit_release_repo.py

# 完整 Docker 验证（构建 + 运行 + 测试）
python3 scripts/validate_docker_release.py

# 完整浸泡测试（所有任务）
python3 scripts/validate_docker_release.py --run-full-soak
```
