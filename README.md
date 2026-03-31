# Simulation-Generation-Agent

자연어 태스크 설명에서 시작해 YAML 생성, IsaacLab 환경 구성, CaP 기반 실행, 데이터 수집까지 연결하는 로봇 시뮬레이션 자동화 프레임워크입니다.

## 문서 시작점

- 설치와 첫 실행: `docs/getting_started.md`
- CLI 사용법과 출력 해석: `docs/usage.md`
- dataset/export/preprocess 기준: `docs/dataset_alignment_and_export.md`
- task 상태 보드: `docs/tasks_overview.md`
- 문제 해결: `docs/troubleshooting.md`

## 파이프라인

### Full Pipeline (NL → YAML → IsaacLab → CAP → Video)

```bash
bash scripts/run_full_test.sh
```

3단계로 구성:
1. **NL → YAML**: 자연어 → RAG 기반 task YAML 생성 (`task_spec_agent`)
2. **YAML → IsaacLab**: LLM 코드 생성 → `ManagerBasedRLEnv` 구성
3. **CAP → Execution → Video**: CaP 코드 생성 → 시뮬레이션 실행 → 성공 판정

### 개별 파이프라인

| 파이프라인 | 설명 | 진입점 |
|-----------|------|--------|
| Task Spec Agent | NL → YAML 변환 | `scripts/task_spec_agent/task_spec_agent.py` |
| IsaacLab | YAML → 환경 코드 생성 + 평가 | `scripts/run_isaac_lab.py` |
| Data Collection | IsaacLab env 위에서 데이터 수집 | `scripts/run_data_collection.py` |
| E2E Batch | config 기반 대규모 배치 수집 | `scripts/run_e2e_batch.py` |

## Quick Start

### 1. 설치

```bash
git clone <repo-url>
cd Simulation-Generation-Agent

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

### 3. Full Pipeline 실행

```bash
bash scripts/run_full_test.sh
```

### 4. Data Collection

```bash
pip install -e ".[data-collection]"
pip install pin

python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/<run_dir>
```

## 토큰 사용량 추적

API 비용 관리를 위해 파이프라인 전체의 토큰 사용량을 추적합니다.

```bash
export TOKEN_USAGE_FILE=outputs/token_usage.jsonl
bash scripts/run_full_test.sh
# 실행 완료 시 step별/model별 토큰 리포트 출력
```

## 프로젝트 구조

```text
Simulation-Generation-Agent/
├── src/common/              # LLM 클라이언트, 토큰 트래커
├── src/isaac_lab/           # IsaacLab 코드 생성 + 평가
├── src/isaac_sim/           # Isaac Sim 시각 검증
├── src/data_collection/     # 데이터 수집 / export / preprocess
├── src/kinematics/          # IK/FK 엔진, 보간, 좌표 변환
├── src/task_search/         # task YAML 카탈로그/검색
├── scripts/                 # CLI 진입점
│   └── task_spec_agent/     # NL→YAML RAG 파이프라인
├── configs/                 # 설정 파일
├── tasks/                   # task YAML corpus
├── prompts/                 # LLM 시스템 프롬프트
├── data/vector_store/       # RAG FAISS 인덱스
├── assets/                  # 로컬 USD/URDF 자산
└── docs/                    # 사용자 문서
```

## 현재 task corpus

- 총 task YAML: 78
- 로봇: franka, openarm, so101, ur10e
- 상세: `docs/tasks_overview.md`, `docs/task_taxonomy.md`

## 참고

- 생성 산출물은 `outputs/` 아래에 저장되며 git에 커밋하지 않습니다.
- IK 엔진(`src/kinematics/`)은 Pinocchio 기반이며 `pip install pin`이 필요합니다.
- MCP 기반 Isaac Sim 검증은 `localhost:8766`에 의존합니다.
