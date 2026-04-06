# 설치 가이드

[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../getting_started.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](getting_started.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](../zh/getting_started.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](../ja/getting_started.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](../de/getting_started.md)

대상: 처음 이 레포를 실행하는 사용자
세부 CLI 옵션과 결과 해석: `docs/usage.md`

## 1. 공통 설치

```bash
git clone --recurse-submodules <repo-url>
cd Simulation-Generation-Agent

pip install -e .
cp .env.example .env
```

### Submodule (AutoDataCollector)

이 프로젝트는 `external/AutoDataCollector`를 git submodule로 포함합니다. Data Collection (Stage 3)에서 judge 프롬프트와 IK 유틸리티를 참조합니다.

`--recurse-submodules`로 클론하면 자동으로 받아집니다. 빠졌을 경우 수동 초기화:

```bash
git submodule update --init --recursive
ls external/AutoDataCollector/  # 파일이 있어야 정상
```

필수 `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1/
```

선택 설정:

```bash
# VLM backends (환경 검증 + 에피소드 성공 판정용)
# ANTHROPIC_API_KEY=...
# GOOGLE_API_KEY=...

# Paths
# ISAACLAB_PATH=~/workspace/IsaacLab

# 토큰 사용량 추적
# TOKEN_USAGE_FILE=outputs/token_usage.jsonl
```

## 2. IsaacLab 설치 (주력 경로)

```bash
cd ~/workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

프로젝트에서 IsaacLab 위치를 인식시키는 방법:

```bash
export ISAACLAB_PATH=~/workspace/IsaacLab
```

또는 `configs/isaaclab_agent_config.yaml`의 `isaaclab.path`를 직접 설정한다.

연결 확인:

```bash
# 로컬 (conda 환경)
conda run -n env_isaaclab --no-capture-output \
  python -c "import isaaclab; print('IsaacLab import OK')"

# Docker 환경에서는 conda 대신 venv가 사용되므로 아래처럼 확인
# python -c "import isaaclab; print('IsaacLab import OK')"

python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run
```

## 3. Isaac Sim + MCP 설치 (선택)

Isaac Sim은 시각 검증 경로에서만 필요하다.

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

정상 기동 시 `localhost:8766`에서 TCP 소켓을 수신한다.

연결 확인:

```bash
python3 tests/test_components.py connection
```

## 4. Data Collection 설치 (선택)

Data Collection은 IsaacLab 위에서 동작한다.

```bash
pip install -e ".[data-collection]"
pip install pin
```

확인:

```bash
python3 -c "from src.agent.data_collection.adc_imports import is_adc_available; print(is_adc_available())"
python3 -c "import pinocchio; print('Pinocchio OK')"
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --help
```

## 5. Task Spec Agent 설치 (NL → YAML)

Task Spec Agent는 자연어 입력을 YAML 태스크 명세로 변환하는 Stage 1 파이프라인입니다.

```bash
pip install langchain langchain-openai langchain-community faiss-cpu sentence-transformers
```

벡터 스토어는 첫 실행 시 `data/vector_store/`에 자동 생성됩니다 (tasks/ 디렉토리의 기존 YAML을 인덱싱).

확인:

```bash
cd scripts/task_spec_agent
python3 -c "from rag_match_yaml_generator import RAGYAMLGenerator; print('RAG OK')"
cd ../..
```

단독 실행:

```bash
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube" --robot franka --output outputs/test_task.yaml
```

## 6. 첫 성공 실행 권장 순서

### 인터랙티브 모드 (처음 사용하는 분에게 권장)

```bash
./run_agent.sh
```

인수 없이 실행하면 인터랙티브 TUI 대시보드가 열립니다. 설정(F1), 실행 이력(F2)을 확인하고, 태스크 설명을 직접 입력할 수 있습니다.

### CLI 모드 — 자연어 한 줄

```bash
./run_agent.sh "Stack the blocks inside the tray on the table"
```

자연어 태스크 설명을 인수로 전달하면 NL→YAML→IsaacLab→DataCollection 전체 파이프라인이 자동 실행됩니다.

실행 시 터미널에 깔끔한 진행 상태가 표시됩니다:
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



실행 시 터미널에 진행 상태가 표시됩니다:
옵션:
```bash
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

### Full Pipeline (13개 태스크 일괄)

```bash
bash scripts/run_full_test.sh
```

13개 Franka 태스크에 대해 3단계를 순차 실행합니다:
1. **NL → YAML**: `task_spec_agent`가 자연어를 YAML로 변환
2. **YAML → IsaacLab**: LLM이 환경 Python 코드를 생성하고 실행 검증
3. **CaP → Data**: LLM이 스킬 코드를 생성하고, IK로 실행하고, 성공 판정 후 데이터 수집

### IsaacLab 단독 실행 (Stage 2만)

```bash
./run_agent.sh --mode isaac-lab --task tasks/franka/stack/franka_stack.yaml
```

성공 기준:
- `outputs/isaaclab/<run_dir>/env_cfg.py`
- `outputs/isaaclab/<run_dir>/result.json`

### Data Collection 단독 실행 (Stage 3만)

```bash
./run_agent.sh --mode data-collection --task tasks/franka/stack/franka_stack.yaml
```

성공 기준:
- `outputs/data_collection/<run_dir>/collection_results.json`
- `outputs/data_collection/<run_dir>/raw_dataset/`

### 학습 준비 경로

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets

python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

## 7. Docker로 실행

로컬 환경 세팅 없이 Docker 컨테이너로 바로 실행할 수 있습니다.

사전 요구: [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
# submodule 초기화 + 이미지 빌드
git submodule update --init --recursive
docker build -t simgen-agent .

# 자연어 입력으로 전체 파이프라인 실행
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --robot franka --episodes 5

# JSON 입력
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  data/input_sample.json results/output.json

# 사용법 확인
docker run --rm simgen-agent --help
```

> **출력 경로 차이**: Docker에서는 `/workspace/artifacts`에 결과가 저장됩니다 (호스트 `./artifacts/`로 마운트). 로컬에서는 `outputs/`에 저장됩니다. Docker에서 로컬과 동일한 경로를 사용하려면 `-e SIMGEN_ARTIFACT_ROOT=/workspace/Simulation-Generation-Agent/outputs`를 추가하세요.

상세 Docker 가이드: [README.docker.md](../../README.docker.md)

## 8. 다음에 볼 문서

- 파이프라인 아키텍처 상세: [docs/architecture.md](architecture.md)
- 실제 옵션/출력 구조/결과 해석: [docs/usage.md](usage.md)
