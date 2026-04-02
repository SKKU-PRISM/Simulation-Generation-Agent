# Docker 가이드

[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../../README.docker.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](docker.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](../zh/docker.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](../ja/docker.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](../de/docker.md)

이 가이드는 Simulation-Generation-Agent를 Docker 컨테이너 내에서 빌드하고 실행하는 방법을 단계별로 안내합니다. 로컬에 IsaacLab을 설치할 필요가 없습니다.

---

## 사전 요구 사항

시작하기 전에 다음이 준비되어 있는지 확인하세요:

- [ ] 최신 드라이버가 설치된 **NVIDIA GPU**
- [ ] **Docker Engine** (v26.0+)
- [ ] **NVIDIA Container Toolkit**
- [ ] **OpenAI API 키** (또는 Azure OpenAI 자격 증명)

### 시스템 요구 사항

| 구성 요소 | 최소 | 권장 |
|-----------|------|------|
| GPU 메모리 | 8 GB | 16+ GB |
| 디스크 공간 | 50 GB 여유 | 100+ GB 여유 |
| RAM | 16 GB | 32+ GB |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |

### Docker Engine 설치

공식 가이드를 따르세요: [Ubuntu에 Docker Engine 설치](https://docs.docker.com/engine/install/ubuntu/)

설치 후 확인:
```bash
docker --version
docker info
```

### NVIDIA Container Toolkit 설치

공식 가이드를 따르세요: [NVIDIA Container Toolkit 설치](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

설치 후 Docker에서 GPU 접근을 확인하세요:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 nvidia-smi
```

GPU 목록이 표시되어야 합니다. 권한 오류가 발생하면 `newgrp docker`를 실행하거나 로그아웃 후 다시 로그인하세요.

---

## 1단계: 소스 코드 가져오기

```bash
git clone --recurse-submodules <repo-url>
cd Simulation-Generation-Agent
```

> `--recurse-submodules` 없이 이미 클론한 경우 다음을 실행하세요:
> ```bash
> git submodule update --init --recursive
> ```

---

## 2단계: API 키 설정

```bash
cp .env.example .env
```

`.env`를 열고 API 키를 설정하세요 (로컬 실행용):
```
OPENAI_API_KEY=sk-your-key-here
```

> **Docker 사용자**: Docker 실행 시 `.env` 파일을 만들 필요가 없습니다. API 키는 런타임에 `-e` 플래그를 통해 직접 전달됩니다. `.env` 파일은 로컬(비Docker) 실행에만 필요합니다.

---

## 3단계: Docker 이미지 빌드

이 저장소에는 완전한 환경을 자동으로 설정하는 `Dockerfile`이 포함되어 있습니다.

```bash
docker build -t simgen-agent .
```

> **디스크 공간**: 약 50GB 필요. **첫 빌드**: 약 30-60분 소요. Docker 레이어 캐싱 덕분에 이후 빌드는 훨씬 빠릅니다.

### Dockerfile이 수행하는 작업

저장소 루트에 있는 `Dockerfile`은 다음을 포함하는 독립적인 이미지를 빌드합니다:

| Layer | What's installed |
|-------|-----------------|
| Base | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |
| Python | 3.11 (venv at `/opt/isaaclab-env`) |
| Isaac Sim | 5.1.0 (pip from `pypi.nvidia.com`) |
| IsaacLab | v2.3.2 (source build from GitHub) |
| PyTorch | 2.7.0 (CUDA 12.8) |
| Project | `requirements.txt` + all source code |
| Entry point | `ENTRYPOINT ["./run_agent.sh"]` |

컨테이너의 엔트리 포인트는 `run_agent.sh`이므로, 다음과 같이 실행하면:
```bash
docker run simgen-agent "Stack the blocks..."
```
컨테이너 내부에서 자동으로 `run_agent.sh "Stack the blocks..."`가 실행됩니다.

이미지가 빌드되었는지 확인:
```bash
docker images | grep simgen-agent
```

---

## 4단계: 파이프라인 실행

### 옵션 A: 자연어 입력 (가장 간단)

로봇이 수행할 작업을 설명하기만 하면 됩니다:

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/results:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="your-key-here" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

옵션 추가:
```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/results:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="your-key-here" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

### 옵션 B: JSON 입력

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  data/input_sample.json results/output.json
```

기본 `data/input_sample.json`에는 샘플 FrankaStackTray 태스크가 포함되어 있습니다. 직접 만들 수도 있습니다:
```json
{
  "tasks": [
    {"task_description": "Pick up the cube and place it on the target", "robot": "franka"}
  ],
  "config": {"target_success": 1, "max_attempts": 3}
}
```

### 옵션 C: 개별 스테이지 (고급)

```bash
# Stage 2만: IsaacLab 환경 코드 생성
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode isaac-lab --task tasks/franka/stack/franka_stack.yaml

# Stage 3만: 데이터 수집
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode data-collection --task tasks/franka/stack/franka_stack.yaml

# 배치 실행 (여러 태스크)
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode e2e-batch --config configs/docker/e2e_batch_smoke.yaml
```

### Azure OpenAI 사용

OpenAI 플랫폼 대신 Azure OpenAI를 사용하려면 `-e` 플래그로 Azure 자격 증명을 전달합니다:

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

| 변수 | 필수 | 설명 |
|------|------|------|
| `AZURE_OPENAI_API_KEY` | 예 | Azure OpenAI API 키 |
| `AZURE_OPENAI_BASE_URL` | 예 | Azure 엔드포인트 (`/openai/v1/` 접미사 포함) |
| `AZURE_OPENAI_ENDPOINT` | 예 | Azure 리소스 엔드포인트 (Stage 1에서 사용) |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | 예 | Azure에 배포된 모델 이름 (Stage 1에서 사용) |

> `OPENAI_API_KEY` 또는 위 Azure 변수 세트 중 하나만 필요합니다.

### 모델 선택

기본 모델은 `gpt-5-mini`입니다. `OPENAI_MODEL` 환경변수로 변경할 수 있습니다:

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="your-key" \
  -e OPENAI_MODEL="gpt-4o" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

### 도움말 표시

```bash
docker run --rm simgen-agent --help
```

### 실행 화면 예시

파이프라인이 실행되면 터미널에 깔끔한 진행 상황이 표시됩니다:

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

각 스테이지가 실행되는 동안 회전 인디케이터가 표시됩니다. 상세 로그는 파일에 저장되며, 터미널에는 깔끔한 상태 라인만 표시됩니다.

---

## 5단계: 결과 확인

결과는 호스트 머신의 `./artifacts/`에 저장됩니다 (컨테이너 내부의 `/workspace/artifacts`에서 매핑).

```bash
ls artifacts/
```

일반적인 출력 구조:
```
artifacts/
├── isaaclab/              # Stage 2: 생성된 환경 코드
│   └── 20260402_*/        # 타임스탬프가 포함된 실행 디렉토리
│       ├── env_cfg.py     # 환경 설정
│       ├── run_env.py     # 환경 실행기
│       ├── result.json    # 성공/실패 상태
│       └── debug/         # 스크린샷 (전면, 상단, 손목)
└── data_collection/       # Stage 3: 수집된 에피소드
    └── TaskName_*/
        ├── collection_results.json
        ├── raw_dataset/   # 에피소드 데이터
        └── videos/        # 녹화된 영상
```

JSON 입력 모드의 경우, 결과는 `results/output.json`에도 기록됩니다.

---

## 문제 해결

### GPU가 감지되지 않음

```
Error: could not select device driver "nvidia"
```

**해결 방법**: NVIDIA Container Toolkit을 설치하거나 재설치하세요:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### IsaacSim 설치 중 빌드 실패

IsaacSim pip 패키지는 약 15GB입니다. 네트워크 문제로 실패한 경우:
```bash
# 캐시 없이 재시도
docker build --no-cache -t simgen-agent .
```

### 메모리 부족 (OOM)

IsaacLab 시뮬레이션은 최소 **8GB GPU 메모리** (Stage 2만) 또는 **16GB+** (Stage 3 포함 전체 파이프라인)가 필요합니다. OOM 오류가 발생하면:
- 다른 GPU 집약적 애플리케이션을 종료하세요
- `--episodes`를 1로 줄이세요
- `--mode isaac-lab`으로 Stage 2만 먼저 테스트하세요

### API 키가 작동하지 않음

```
Either OPENAI_API_KEY or AZURE_OPENAI_API_KEY must be set.
```

**해결 방법**: 키가 `-e`로 올바르게 전달되었는지 확인하세요:
```bash
# 방법 1: 인라인
-e OPENAI_API_KEY="sk-your-key"

# 방법 2: .env 파일에서
-e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)"

# 방법 3: 먼저 export
export OPENAI_API_KEY="sk-your-key"
docker run ... -e OPENAI_API_KEY ...
```

---

## 레퍼런스

### 환경 변수

| 변수 | 필수 | 설명 |
|------|------|------|
| `OPENAI_API_KEY` | 예 (또는 Azure) | OpenAI 플랫폼 API 키 |
| `AZURE_OPENAI_API_KEY` | 예 (또는 OpenAI) | Azure OpenAI API 키 |
| `AZURE_OPENAI_BASE_URL` | Azure 사용 시 | Azure 엔드포인트 URL |
| `OPENAI_BASE_URL` | 아니오 | 커스텀 OpenAI 호환 엔드포인트 |
| `HF_TOKEN` | 아니오 | HuggingFace 토큰 (데이터셋 업로드용) |
| `OPENAI_MODEL` | 아니오 | 기본 모델 오버라이드 (기본값: gpt-5-mini) |

### `run_agent.sh` 모드

| 모드 | 설명 |
|------|------|
| *(위치 인자)* | 전체 파이프라인: 자연어 또는 JSON 입력 |
| `--mode e2e-batch` | 여러 태스크의 배치 실행 |
| `--mode isaac-lab` | Stage 2만: 환경 코드 생성 |
| `--mode data-collection` | Stage 3만: 데이터 수집 |

### 이미지 사양

| 컴포넌트 | 버전 |
|----------|------|
| 베이스 이미지 | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |
| Python | 3.11 |
| Isaac Sim | 5.1.0 (pip) |
| IsaacLab | v2.3.2 (source) |
| PyTorch | 2.7.0 (CUDA 12.8) |

### 검증 스크립트

```bash
# 정적 감사 (시크릿, 경로, 구조)
python3 scripts/audit_release_repo.py

# 전체 Docker 검증 (빌드 + 실행 + 테스트)
python3 scripts/validate_docker_release.py

# 전체 소크 테스트 (모든 태스크)
python3 scripts/validate_docker_release.py --run-full-soak
```
