# Docker Release Guide

이 저장소의 공개 Docker 환경은 `headless IsaacLab + data collection + e2e batch` 경로를 대상으로 한다. `Isaac Sim + MCP` 기반 시각 검증은 이 컨테이너 범위 밖이며, 지원하지 않는다.

## Target Runtime

- Host OS: Ubuntu 22.04 계열 권장
- GPU: NVIDIA GPU 필수
- Container runtime: NVIDIA Container Toolkit 필수
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
런타임 산출물은 소스 트리 내부 `outputs/` 대신 `/workspace/artifacts`에 기록되도록 분리되어 있다.

## Required Environment Variables (택1)

OpenAI platform 또는 Azure OpenAI 중 하나를 선택하여 환경변수를 주입한다.

**OpenAI platform (권장):**
- `OPENAI_API_KEY`

**Azure OpenAI:**
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_BASE_URL`

## Optional Environment Variables

- `OPENAI_BASE_URL` (커스텀 엔드포인트)
- `HF_TOKEN`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `ISAACLAB_PATH`
- `ADC_PATH`

이미지 안에 API key나 `.env` 파일은 포함하지 않는다.

## Host Setup

호스트에는 Docker Engine과 NVIDIA Container Toolkit이 준비되어 있어야 한다. 설치 자체는 공개 repo 바깥의 로컬 호스트 준비 단계로 보고, 공개 문서에서는 공식 문서 경로만 안내한다.

- Docker Engine: `https://docs.docker.com/engine/install/ubuntu/`
- NVIDIA Container Toolkit: `https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`

설치 후에는 새 로그인 세션을 열거나 `newgrp docker`를 실행한 뒤 아래 명령으로 daemon 접근과 GPU 컨테이너를 확인한다.

```bash
docker info
```

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 nvidia-smi
```

## 실행 (가장 간단한 방법)

자연어로 태스크를 설명하면 전체 파이프라인이 자동 실행된다.

```bash
docker run --rm --gpus all \
  -v $(pwd)/results:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="sk-..." \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

결과는 `results/output.json`에 기록된다.

## Challenge Submission (심사위원 실행)

심사위원은 아래 명령어로 이미지를 빌드하고 실행할 수 있다.

```bash
# 1. 이미지 빌드
docker build -t simgen-agent .

# 2. 컨테이너 실행 (심사위원 인터페이스)
docker run --rm --gpus all \
  -v $(pwd)/input_data:/workspace/Simulation-Generation-Agent/data \
  -v $(pwd)/output_data:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="sk-..." \
  simgen-agent \
  data/input_sample.json results/output.json
```

입력 JSON 형식:
```json
{
  "tasks": [
    {"task_description": "Stack the blocks inside the tray on the table", "robot": "franka"}
  ],
  "config": {"target_success": 1, "max_attempts": 3}
}
```

결과는 `results/output.json`에 구조화된 JSON으로 기록된다.

## `run_agent.sh` Modes

`run_agent.sh`는 두 가지 인터페이스를 지원한다.

**심사 제출 모드** (positional args):
```bash
run_agent.sh <input_file.json> [output_file.json]
```

**고급 파이프라인 모드** (`--mode` flag):
- `e2e-batch`: `scripts/run_e2e_batch.py`를 호출해 release 또는 representative batch를 실행한다.
- `isaac-lab`: `scripts/run_isaac_lab.py`를 호출해 단일 task의 코드 생성과 runtime validation을 수행한다.
- `data-collection`: `scripts/run_data_collection.py`를 호출해 단일 task collection 파이프라인을 실행한다.

## Build

이미지를 새로 빌드한다. submodule이 초기화되어 있어야 한다.

```bash
cd /path/to/Simulation-Generation-Agent

# submodule 초기화 (최초 1회)
git submodule update --init --recursive

# 빌드
docker build -t simgen-agent .
```

## Help / Healthcheck

Entrypoint 계약과 기본 Python runtime이 제대로 들어갔는지 빠르게 확인한다.

```bash
docker run --rm simgen-agent --help
```

```bash
docker run --rm --gpus all \
  --entrypoint python \
  simgen-agent \
  -c "import isaaclab, isaacsim; print(isaaclab.__file__); print(isaacsim.__file__)"
```

작은 smoke batch로 artifact 분리와 기본 배치 경로를 확인할 수도 있다.

```bash
docker run --rm --gpus all \
  -e OPENAI_API_KEY \
  -v $(pwd)/artifacts:/workspace/artifacts \
  simgen-agent \
  --mode e2e-batch \
  --config /workspace/Simulation-Generation-Agent/configs/docker/e2e_batch_smoke.yaml
```

정적 공개/보안 audit는 아래 명령으로 확인한다.

```bash
python3 scripts/audit_release_repo.py
```

## Representative Validation

Representative validation은 build, entrypoint, image audit, 단일 task 실행, 대표 batch, resume probe까지 자동으로 묶어 확인한다.

```bash
python3 scripts/validate_docker_release.py
```

대표 batch만 직접 실행하려면 아래 명령을 사용한다.

```bash
docker run --rm --gpus all \
  -e OPENAI_API_KEY \
  -v $(pwd)/artifacts:/workspace/artifacts \
  simgen-agent \
  --mode e2e-batch \
  --config /workspace/Simulation-Generation-Agent/configs/docker/e2e_batch_representative.yaml
```

## Full Validation

전체 release batch와 full soak까지 포함한 전수 검증을 수행한다.

```bash
python3 scripts/validate_docker_release.py --run-full-soak
```

전체 batch를 직접 실행하려면 아래 명령을 사용한다.

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

- `run_agent.sh`가 공개 release용 단일 진입점이다.
- 기본 모드는 `e2e-batch`다.
- 기본 artifact root는 `/workspace/artifacts`다.
- Docker release config는 현재 안정적으로 검증된 정책을 유지하며 `FrankaPickPlaceMug`와 `FrankaCabinetBlocks`는 `enabled: false`로 남긴다.
- `hf.upload`는 release config에서 기본적으로 `false`다. 필요하면 config override로만 활성화한다.
