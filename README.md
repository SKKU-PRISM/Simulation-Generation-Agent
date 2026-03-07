# Simulation-Generation-Agent

YAML task document 기반 로보틱스 시뮬레이션 환경 자동 구성 프레임워크입니다. 현재 기준 주력 경로는 **IsaacLab Pipeline**이며, 같은 task YAML을 기준으로 **Isaac Sim 시각 검증**과 **Data Collection**까지 확장할 수 있습니다.

## 핵심 파이프라인

1. **IsaacLab Pipeline (주력)**  
   YAML -> LLM 코드 생성 -> `isaaclab.sh` 실행 -> 100점 평가
2. **Isaac Sim Pipeline (보조 시각 검증)**  
   YAML -> MCP로 씬 빌드 -> 스크린샷 -> VLM 평가 -> 기준 점수 도달 시 종료
3. **Data Collection (IsaacLab 확장)**
   생성된 IsaacLab 환경 위에서 detect -> plan (LLM) -> execute (6-DOF IK) -> judge (VLM / geometric) -> record 반복 후 LeRobot v3.0 데이터셋 생성
   Multi-camera (top + wrist + front), 3-tier 성공 판정 (VLM > goal verification > env termination), robot base frame 좌표계 포함

## Quick Start

### 1. 공통 설치

```bash
git clone <repo-url>
cd Simulation-Generation-Agent

pip install -e .
cp .env.example .env
```

`.env`에는 최소 다음 값이 필요합니다.

```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/v1/
```

### 2. IsaacLab 설치 및 연결

```bash
cd ~/workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

그다음 프로젝트 루트에서:

```bash
export ISAACLAB_PATH=~/workspace/IsaacLab
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
```

### 3. 선택 확장

```bash
# Isaac Sim visual validation
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --backend auto

# Data Collection
pip install -e ".[data-collection]"
pip install pin                    # Pinocchio (6-DOF IK 필수)
git submodule update --init external/AutoDataCollector
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml
```

## 대표 명령어

```bash
# IsaacLab: 생성 + 실행
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml

# IsaacLab: 생성 + 실행 + 평가
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate

# IsaacLab: 기존 결과 재평가
python3 scripts/evaluate.py outputs/isaaclab/<run_dir> tasks/franka/stack/franka_stack.yaml

# Isaac Sim: 자동 씬 빌드 + 스크린샷 + VLM 평가
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --backend auto

# Data Collection: 생성된 env 위에서 에피소드 수집
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/<run_dir>
```

## 문서

- `docs/getting_started.md`: 설치 순서와 외부 의존성 연결
- `docs/usage.md`: 각 CLI 사용법과 출력 구조
- `docs/evaluation.md`: IsaacLab 평가 체계와 결과 해석
- `docs/task_yaml_spec.md`: task YAML 포맷과 자산 규약
- `docs/tasks_overview.md`: 전체 task 목록과 IsaacLab/ADC 검증 상태 보드
- `docs/task_descriptions_ko.md`: task 이름, 한글 설명, ADC 검증 여부 빠른 참고용 문서
- `docs/troubleshooting.md`: 자주 발생하는 런타임/VLM/MCP 문제

## 프로젝트 구조

```text
Simulation-Generation-Agent/
├── src/common/              # Azure/OpenAI, MCP 공용 클라이언트
├── src/isaac_lab/           # 주력 파이프라인: 코드 생성 + 평가
├── src/isaac_sim/           # 시각 검증 파이프라인
├── src/data_collection/     # IsaacLab 기반 데이터 수집
├── src/task_search/         # task YAML 카탈로그/검색 메타데이터
├── scripts/                 # CLI 진입점
├── configs/                 # agent/pipeline/eval/robot profile 설정
├── tasks/                   # 78개 task YAML
├── docs/                    # 사용자 문서
├── assets/                  # 로컬 USD 자산 (예: SO-101)
└── external/AutoDataCollector/  # ADC 서브모듈, read-only
```

## 지원 로봇과 태스크

- **Franka Panda**: assembly, stack, lift, pick_place, cabinet, sort, peg_insert
- **OpenArm**: assembly, stack, lift, pick_place, reach, cabinet, sort
- **UR10**: assembly
- **UR10e**: stack, cabinet, pick_place, reach
- **SO-101**: assembly, stack, lift, pick_place, reach, sort

총 task YAML 수는 현재 `78`개입니다.

## 참고

- Isaac Sim 경로는 MCP TCP 소켓 `localhost:8766`에 의존합니다.
- `external/AutoDataCollector/`는 read-only 서브모듈입니다. 이 레포에서 직접 수정하지 않습니다.
- 생성 산출물은 `outputs/` 아래에 저장되며 git에 커밋하지 않습니다.
