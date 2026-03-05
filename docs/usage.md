# 사용법

이 문서는 두 파이프라인의 CLI 사용법과 전체 워크플로우를 설명합니다.

---

## IsaacLab Pipeline: 코드 생성

YAML 태스크 문서를 입력하면, LLM이 IsaacLab ManagerBasedRLEnv Python 코드를 생성하고 자동 실행합니다.

### 파이프라인 흐름

```
YAML Task Document
       |
       v
  IsaacLabAgent
       |
  1. YAML 파싱 + 참조 코드 선택
  2. LLM 프롬프트 구성
  3. Azure OpenAI (gpt-5-mini) 호출
       |
       v
  env_cfg.py + run_env.py + mdp/ 생성
       |
       v
  isaaclab.sh -p run_env.py --headless (conda env_isaaclab)
       |
   +---+---+
   |       |
 성공    실패 → 에러 traceback 추출
   |       |    → LLM에 수정 요청
   |       |    → 코드 재생성 (최대 5회)
   v       v
 완료    최종 실패
```

### 기본 명령어

```bash
# 단일 태스크 실행 (코드 생성 + IsaacLab 실행)
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml

# 코드 생성만 (실행 안 함)
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run

# 코드 생성 + 실행 + 품질 평가 (100점 만점)
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate

# 기존 출력물 재평가 (코드 생성/실행 건너뜀)
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml \
  --eval-only outputs/isaaclab/FrankaStack_20260130/

# 디렉토리 내 모든 YAML 배치 처리
python3 scripts/run_isaac_lab.py --batch tasks/franka/

# 커스텀 출력 디렉토리
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml \
  --output-dir /custom/output/path

# 커스텀 config 파일
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml \
  --config configs/my_config.yaml
```

### 출력 디렉토리 구조

코드 생성 결과는 `outputs/isaaclab/{TaskName}_{timestamp}/`에 저장됩니다:

```
outputs/isaaclab/FrankaStack_20260212_143000/
├── env_cfg.py           # @configclass 기반 환경 설정
│                        #   SceneCfg, ActionsCfg, ObservationsCfg,
│                        #   EventCfg, RewardsCfg, TerminationsCfg
├── run_env.py           # 실행 스크립트
│                        #   AppLauncher 초기화 → ManagerBasedRLEnv → step loop
├── mdp/                 # 로컬 MDP 패키지
│   ├── __init__.py      #   isaaclab.envs.mdp 재export + 커스텀 함수
│   ├── rewards.py       #   (선택) 커스텀 보상 함수
│   └── terminations.py  #   (선택) 커스텀 종료 조건
├── .success_marker      # 실행 성공 시 생성
├── error_attempt_1.txt  # (실패 시) 1차 시도 에러 traceback
├── error_attempt_2.txt  # (실패 시) 2차 시도 에러 traceback
└── eval_report.json     # (--evaluate 사용 시) 평가 결과
```

### 생성 코드 파일별 역할

| 파일 | 역할 |
|------|------|
| `env_cfg.py` | IsaacLab `ManagerBasedRLEnvCfg`를 상속하는 환경 설정. 씬 구성, 관측, 보상, 종료 조건 등을 정의 |
| `run_env.py` | `AppLauncher` 초기화 → 환경 인스턴스화 → step loop 실행. **반드시 AppLauncher가 physics import 전에 초기화되어야 함** |
| `mdp/__init__.py` | `isaaclab.envs.mdp`의 모든 함수를 re-export하고, 커스텀 함수가 있으면 추가 import |
| `mdp/rewards.py` | YAML goal에 맞는 커스텀 보상 함수 (예: 큐브 스택 높이 보상) |
| `mdp/terminations.py` | 커스텀 종료 조건 (예: 목표 달성 시 에피소드 종료) |

---

## Isaac Sim Pipeline: 씬 빌드

YAML 태스크 문서를 Isaac Sim에서 시각적으로 빌드합니다. MCP 서버가 실행 중이어야 합니다.

> **현재 제약**: Isaac Sim Pipeline의 자동 반복 루프(빌드 → 스크린샷 → VLM 평가 → 재시도)는 아직 미구현 상태입니다.
> 각 컴포넌트를 개별적으로 사용해야 합니다.

### 전제 조건

```bash
# Isaac Sim + MCP 서버가 실행 중이어야 합니다
cd ~/workspace/isaac-sim-mcp
./run_isaac_mcp_streaming.sh 0
```

### 씬 빌드

```bash
python3 scripts/build_scene.py tasks/franka/stack/franka_stack.yaml

# 커스텀 MCP 서버 주소
python3 scripts/build_scene.py tasks/franka/stack/franka_stack.yaml \
  --host localhost --port 8766
```

### 컴포넌트 테스트

```bash
# MCP 연결 테스트
python3 tests/test_components.py connection

# 씬 빌드 테스트
python3 tests/test_components.py scene tasks/franka/stack/franka_stack.yaml

# 스크린샷 캡처 테스트
python3 tests/test_components.py screenshot outputs/test.png

# VLM 평가 테스트
python3 tests/test_components.py vlm <screenshot_path> <task_yaml_path>

# 전체 통합 테스트 (VLM 제외)
python3 tests/test_components.py full --skip-vlm

# 전체 통합 테스트 (VLM 포함)
python3 tests/test_components.py full --document tasks/franka/stack/franka_stack.yaml
```

---

## Standalone 평가

기존에 생성된 IsaacLab 코드를 평가만 실행합니다.

```bash
# 정적 분석 + 런타임 검증
python3 scripts/evaluate.py \
  outputs/isaaclab/FrankaStack_20260212/ \
  tasks/franka/stack/franka_stack.yaml

# 정적 분석만 (IsaacLab 환경 불필요)
python3 scripts/evaluate.py --skip-runtime \
  outputs/isaaclab/FrankaStack_20260212/ \
  tasks/franka/stack/franka_stack.yaml
```

종료 코드:
- `0` = 점수 70점 이상 (합격)
- `1` = 점수 70점 미만 (불합격)

평가 결과 해석은 [평가 시스템](evaluation.md)을 참고하세요.

---

## Data Collection: 에피소드 수집 (IsaacLab 확장)

IsaacLab 환경 위에서 LLM 스킬 플래닝 기반 자동 에피소드 수집을 수행하고, LeRobot v3.0 데이터셋을 생성합니다.

### 파이프라인 흐름

```
YAML Task Document
       |
       v
  DataCollectionPipeline
       |
  1. YAML 파싱 → 로봇 타입 감지
  2. IsaacLab 환경 생성 (또는 기존 환경 로드)
  3. collect_data.py 동적 생성
       |
       v
  conda subprocess (env_isaaclab)
       |
  Episode Loop (최대 50회):
    ├── scene detect (오브젝트 위치 쿼리)
    ├── LLM skill plan (스킬 시퀀스 생성)
    ├── execute skills (IK → 관절 제어)
    ├── camera capture (매 스텝 RGB 이미지)
    ├── VLM judge (before/after 비교)
    └── record (npy + png)
       |
       v
  raw_dataset/ → LeRobot v3.0 변환
```

### 전제 조건

```bash
# Data Collection 의존성 설치
pip install -e ".[data-collection]"

# (선택) ADC 서브모듈 초기화 — Pinocchio IK 사용 시
git submodule update --init external/AutoDataCollector
```

### 기본 명령어

```bash
# 단일 태스크 데이터 수집 (환경 자동 생성 + 50 에피소드)
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml

# 기존 생성된 IsaacLab 환경 사용 (환경 생성 건너뜀)
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/FrankaStack_20260219/

# 에피소드 수 + 데이터셋 ID 설정
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --episodes 100 --repo-id "local/franka_stack"

# VLM 판정 없이 전체 녹화 (모든 에피소드 저장)
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --no-vlm-judge

# GUI 모드 (시각 확인용)
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --gui

# 배치 수집 (디렉토리 내 모든 YAML)
python3 scripts/run_data_collection.py --batch tasks/franka/ --episodes 10

# 커스텀 config 파일
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --config configs/my_data_collection_config.yaml
```

### CLI 옵션 요약

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `yaml_path` | 태스크 YAML 경로 | (필수) |
| `--batch <dir>` | 디렉토리 내 모든 YAML 배치 처리 | - |
| `--env-dir <path>` | 기존 IsaacLab 환경 디렉토리 | 자동 생성 |
| `--episodes <n>` | 수집 에피소드 수 | 50 (config) |
| `--repo-id <id>` | 데이터셋 ID | `local/sim_dataset` |
| `--fps <n>` | 녹화 FPS | 20 (control rate) |
| `--no-vlm-judge` | VLM 성공 판정 비활성화 | false |
| `--gui` | GUI 모드로 실행 | headless |
| `--config <path>` | 커스텀 config 파일 | `configs/data_collection_config.yaml` |
| `-v, --verbose` | 상세 로그 출력 | - |

### 출력 디렉토리 구조

```
outputs/data_collection/FrankaStack_20260301_120000/
├── collect_data.py              # 생성된 수집 스크립트
├── pipeline_config.json         # subprocess용 직렬화된 설정
├── raw_dataset/                 # Stage 1: raw 에피소드 데이터
│   ├── episodes/
│   │   ├── episode_000000/
│   │   │   ├── states.npy       # (T, N_dof) float32 — 관절 위치
│   │   │   ├── actions.npy      # (T, N_dof) float32 — 목표 관절 위치
│   │   │   ├── images/          # {frame:06d}.png RGB 이미지
│   │   │   └── skills.json      # 프레임별 스킬 메타데이터
│   │   └── ...
│   └── metadata.json            # 로봇 정보 + 에피소드 통계
├── collection_results.json      # 에피소드별 성공/실패 + VLM Judge 판정
└── COLLECTION_SUCCESS_MARKER    # 완료 마커
```

### ADC 서브모듈 (선택)

AutoDataCollector(ADC)는 `external/AutoDataCollector/`에 git submodule로 포함됩니다.

| 기능 | ADC 있을 때 | ADC 없을 때 (fallback) |
|------|------------|----------------------|
| IK 솔버 | Pinocchio IK (multi-solution, 정밀) | IsaacLab DifferentialIK (Jacobian) |
| 궤적 보간 | ADC interpolation.py | 인라인 smoothstep 보간 |
| VLM Judge 프롬프트 | ADC CoT 6단계 프롬프트 | 내장 fallback 프롬프트 |

```bash
# ADC 초기화
git submodule update --init external/AutoDataCollector

# 또는 외부 경로 지정
export ADC_PATH=/path/to/AutoDataCollector
```

---

## 전체 워크플로우 예시

### 예시: 새로운 Franka 태스크 생성부터 실행까지

```bash
# 1. 태스크 YAML 작성
#    tasks/franka/stack/franka_stack_v2.yaml 작성

# 2. IsaacLab 코드 생성 + 실행 + 평가
python3 scripts/run_isaac_lab.py \
  tasks/franka/stack/franka_stack_v2.yaml \
  --evaluate

# 3. 결과 확인
cat outputs/isaaclab/FrankaStackV2_*/eval_report.json
```

### 예시: 배치 실행

```bash
# franka 디렉토리의 모든 태스크를 일괄 처리
python3 scripts/run_isaac_lab.py --batch tasks/franka/

# 결과 확인
ls outputs/isaaclab/
```

### 예시: 환경 생성 → 데이터 수집 통합 워크플로우

```bash
# 1. IsaacLab 환경 생성 + 실행
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml

# 2. 생성된 환경으로 데이터 수집 (50 에피소드)
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/FrankaStack_20260301/

# 3. 결과 확인
cat outputs/data_collection/FrankaStack_*/collection_results.json
```

---

## 관련 문서

- [설치 가이드](getting_started.md) — 환경 구성, ADC 서브모듈 설치
- [Task YAML 명세](task_yaml_spec.md) — 태스크 문서 포맷
- [평가 시스템](evaluation.md) — 평가 기준 및 결과 해석
- [트러블슈팅](troubleshooting.md) — 자주 발생하는 문제
