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

---

## 관련 문서

- [설치 가이드](getting_started.md) — 환경 구성
- [Task YAML 명세](task_yaml_spec.md) — 태스크 문서 포맷
- [평가 시스템](evaluation.md) — 평가 기준 및 결과 해석
- [트러블슈팅](troubleshooting.md) — 자주 발생하는 문제
