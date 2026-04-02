# Simulation-Generation-Agent (RAPIDS)

자연어 태스크 설명에서 시작해 YAML 태스크 명세 생성, IsaacLab 시뮬레이션 환경 코드 자동 구성, Code-as-Policies(CaP) 기반 로봇 조작 실행, 학습 가능 데이터셋 수집까지 연결하는 로봇 시뮬레이션 자동화 프레임워크입니다.

## 파이프라인 개요

```
  [자연어 입력]                                              [학습 가능 데이터셋]
  "큐브를 집어서 쌓아라"                                       raw_dataset/
      │                                                          ▲
      ▼                                                          │
 ┌──────────┐     ┌──────────────┐     ┌──────────────┐    ┌───────────┐
 │ Stage 1  │────▶│   Stage 2    │────▶│   Stage 3    │───▶│  Dataset  │
 │ Task Def │     │  Sim Gen     │     │ Data Collect  │    │  Export   │
 │ NL→YAML  │     │ YAML→Env    │     │ CaP→Episodes  │    │ (LeRobot) │
 └──────────┘     └──────────────┘     └──────────────┘    └───────────┘
```

| Stage | 무엇을 하는가 | 핵심 기술 |
|-------|-------------|----------|
| **1. Task Definition** | 자연어 → 구조화된 YAML 태스크 명세 | LangChain RAG (FAISS 벡터 검색) + LLM few-shot 생성 |
| **2. Simulation Generation** | YAML → IsaacLab 환경 Python 코드 | LLM 코드 생성 + PhysX 실행 검증 + 에러 자동 수정 + VLM 환경 검증 |
| **3. Data Collection** | 환경 위에서 로봇 조작 + 성공 데이터 수집 | CaP 스킬 코드 생성 + Pinocchio IK + Geometry/VLM 이중 평가 |

상세 아키텍처: `docs/architecture.md`

## Quick Start

### 1. 설치

```bash
git clone --recurse-submodules <repo-url>
cd Simulation-Generation-Agent

# submodule이 빠졌을 경우
git submodule update --init --recursive

pip install -e .
cp .env.example .env
```

필수 `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1/
```

### 2. IsaacLab 연결

```bash
export ISAACLAB_PATH=~/workspace/IsaacLab
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run
```

### 3. Full Pipeline 실행 (NL → YAML → IsaacLab → CaP → Video)

```bash
bash scripts/run_full_test.sh
```

13개 Franka 태스크에 대해 자연어 입력 → YAML 생성 → 환경 코드 생성 → CaP 실행 → 데이터 수집을 순차적으로 실행합니다.

### 4. 개별 Stage 실행

```bash
# Stage 1: NL → YAML
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube and stack it" \
  --robot franka --output task.yaml

# Stage 2: YAML → IsaacLab 환경 코드
python3 scripts/run_isaac_lab.py task.yaml --evaluate

# Stage 3: 데이터 수집
pip install -e ".[data-collection]" && pip install pin
python3 scripts/run_data_collection.py task.yaml \
  --env-dir outputs/isaaclab/<run_dir> --target-success 5
```

### 5. E2E Batch (대규모 데이터 수집)

```bash
python3 scripts/run_e2e_batch.py configs/e2e_batch_franka50.yaml
python3 scripts/run_e2e_batch.py configs/e2e_batch_franka50.yaml --resume  # 중단 후 재개
```

## 실행 진입점

| 스크립트 | 실행 범위 | 설명 |
|---------|----------|------|
| `scripts/task_spec_agent/task_spec_agent.py` | Stage 1 | NL → YAML 변환 |
| `scripts/run_isaac_lab.py` | Stage 2 | YAML → 환경 코드 생성 + 평가 |
| `scripts/run_data_collection.py` | Stage 3 | 환경 위에서 데이터 수집 |
| `scripts/run_full_test.sh` | Stage 1→2→3 | 13개 태스크 Full Pipeline |
| `scripts/run_e2e_batch.py` | Stage 2→3 + 후처리 | config 기반 대규모 배치 |
| `run_agent.sh` | Stage 2→3 + 후처리 | Docker/릴리스 엔트리포인트 |

## 지원 로봇

| 로봇 | DOF | 그리퍼 | 비고 |
|------|-----|--------|------|
| Franka Panda | 9 (7+2) | Parallel Jaw | 기본 테스트 로봇 |
| UR10e | 12 (6+6) | Robotiq 2F-85 | 산업용 |
| OpenARM | 9 (7+2) | Parallel Jaw | 저비용 오픈소스 |
| SO-101 | 6 (5+1) | Parallel Jaw | 교육용 소형 |

## 토큰 사용량 추적

```bash
export TOKEN_USAGE_FILE=outputs/token_usage.jsonl
export TOKEN_USAGE_LOG=1  # 실시간 콘솔 로그
bash scripts/run_full_test.sh
# 실행 완료 시 step별/model별 토큰 리포트 출력
```

## 프로젝트 구조

```text
Simulation-Generation-Agent/
├── run_agent.sh                  # Docker/릴리스 엔트리포인트
├── scripts/
│   ├── task_spec_agent/          # Stage 1: NL → YAML
│   │   ├── task_spec_agent.py    #   메인 오케스트레이터
│   │   ├── nl_parser.py          #   자연어 파싱 (LLM)
│   │   ├── task_decomposer.py    #   태스크 분해 (LLM)
│   │   ├── feasibility_validator.py  # 물리적 실현 가능성 검증
│   │   ├── rag_yaml_generator.py #   RAG + LLM YAML 생성
│   │   └── llm_client.py         #   멀티 LLM 클라이언트
│   ├── run_isaac_lab.py          # Stage 2 진입점
│   ├── run_data_collection.py    # Stage 3 진입점
│   ├── run_e2e_batch.py          # E2E Batch 진입점
│   └── run_full_test.sh          # Full Pipeline (Stage 1→2→3)
├── src/agent/
│   ├── common/                   # LLM 클라이언트, 토큰 트래커, MCP
│   ├── isaac_lab/                # Stage 2: 환경 코드 생성 + 평가
│   │   ├── agent.py              #   IsaacLabAgent (LLM 코드 생성)
│   │   └── evaluator/            #   4-카테고리 100점 평가
│   ├── isaac_sim/                # Isaac Sim 시각 검증 (부가 경로)
│   ├── data_collection/          # Stage 3: 데이터 수집
│   │   ├── pipeline.py           #   DataCollectionPipeline
│   │   ├── sim_skills.py         #   6-DOF IK 로봇 제어
│   │   ├── sim_judge.py          #   Geometry + VLM 성공 판정
│   │   ├── cap_generator.py      #   CaP 코드 생성
│   │   └── e2e_orchestrator.py   #   E2E 배치 오케스트레이터
│   ├── kinematics/               # IK/FK 엔진 (Pinocchio)
│   └── task_search/              # task YAML 카탈로그/검색
├── configs/                      # 설정 파일
│   ├── robot_profiles/           #   로봇별 프로파일 (관절, 그리퍼, IK)
│   └── docker/                   #   Docker 전용 배치 설정
├── tasks/                        # task YAML corpus (79개)
├── prompts/                      # LLM 시스템 프롬프트
├── assets/                       # 로컬 USD/URDF 자산
└── docs/                         # 사용자 문서
```

## Task Corpus

- 총 task YAML: 82개
- 로봇: Franka (28), OpenARM (24), SO-101 (13), UR10e (17)
- 카테고리: Stack, Lift, Pick&Place, Sort, Cabinet, Assembly, Peg Insert, Reach
## 문서

| 문서 | 역할 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | 3-Stage 파이프라인 아키텍처, 모듈 관계, 데이터 흐름 |
| [docs/getting_started.md](docs/getting_started.md) | 설치, 환경변수, submodule 세팅, 첫 실행 |
| [docs/usage.md](docs/usage.md) | CLI 사용법, 옵션, 출력 구조, 결과 해석 |
| [README.docker.md](README.docker.md) | Docker 빌드/실행, 심사위원 실행 가이드 |
| [docs/README.md](docs/README.md) | 문서 네비게이션, 권장 읽기 순서 |
| [LICENSE](LICENSE) | MIT 라이선스 |

## Docker 빌드 및 실행

### 빌드

```bash
docker build -t simgen-agent .
```

### 실행

```bash
docker run --rm --gpus all \
  -v $(pwd)/input_data:/workspace/Simulation-Generation-Agent/data \
  -v $(pwd)/output_data:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="sk-..." \
  simgen-agent \
  data/input_sample.json results/output.json
```

### 필요 환경변수

- `OPENAI_API_KEY` (필수) 또는 `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_BASE_URL`

상세 Docker 가이드: [README.docker.md](README.docker.md)

### run_agent.sh 사용법

`run_agent.sh`는 Docker 컨테이너의 엔트리포인트이자 릴리스용 단일 실행 진입점입니다.

```bash
# 심사 제출 모드 (기본)
./run_agent.sh <input_file.json> [output_file.json]

# 고급 모드
./run_agent.sh --mode e2e-batch                    # 대규모 배치 수집
./run_agent.sh --mode isaac-lab --task <yaml>      # 단일 태스크 환경 생성
./run_agent.sh --mode data-collection --task <yaml> # 단일 태스크 데이터 수집
./run_agent.sh --help                               # 전체 옵션 확인
```

## 참고

- 생성 산출물은 로컬 실행 시 `outputs/`, Docker 실행 시 `/workspace/artifacts`에 저장됩니다.
- IK 엔진(`src/agent/kinematics/`)은 Pinocchio 기반이며 `pip install pin`이 필요합니다. Pinocchio 미설치 시 IsaacLab DifferentialIK로 자동 fallback됩니다.
- Isaac Sim MCP 시각 검증은 로컬 개발 환경 전용이며 Docker에서는 지원되지 않습니다.
