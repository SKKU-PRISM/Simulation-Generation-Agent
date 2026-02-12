# 평가 시스템

이 문서는 IsaacLab Pipeline의 코드 품질 평가 시스템 (100점 만점, 4개 카테고리)을 설명합니다.

---

## 개요

LLM이 생성한 IsaacLab 코드가 YAML 태스크 문서를 얼마나 정확하게 구현했는지를 자동으로 평가합니다.

```
평가 입력:
  1. 생성된 코드 (env_cfg.py, run_env.py, mdp/)
  2. 원본 YAML 태스크 문서
     ↓
  Phase 1: 정적 분석 (IsaacLab 불필요)
  Phase 2: 런타임 분석 (IsaacLab 필요)
     ↓
  eval_report.json (4카테고리, 100점 만점)
```

---

## 4카테고리 점수 체계

| 카테고리 | 배점 | 검사 내용 |
|----------|:----:|----------|
| **Scene Fidelity** | 30 | YAML에 명시된 에셋이 코드에 올바르게 반영되었는가 |
| **MDP Correctness** | 25 | Observation, Action, Reward, Termination, Event가 올바른가 |
| **Task Alignment** | 25 | YAML의 goal이 보상/종료 함수에 반영되었는가 |
| **Runtime Validity** | 20 | 실제 IsaacLab 환경에서 실행 가능한가 |

### Scene Fidelity (30점)

YAML `assets` 섹션과 생성된 `env_cfg.py`의 씬 구성을 비교합니다.

- **로봇 설정** (10점): robot_type, asset_path, position, initial_joints 일치 여부
- **오브젝트 수** (10점): YAML에 정의된 rigid/static 오브젝트가 코드에 모두 존재하는지
- **오브젝트 속성** (10점): position, physics 속성, randomization 범위 일치

### MDP Correctness (25점)

ManagerBasedRLEnv의 MDP 구성요소가 유효한지 검사합니다.

- **Observations** (5점): `ObservationsCfg`에 유효한 observation 함수가 정의되었는지
- **Actions** (5점): `ActionsCfg`에 올바른 action space가 정의되었는지
- **Rewards** (5점): `RewardsCfg`에 `RewTerm`이 정의되었는지, `weight` 파라미터 포함 여부
- **Terminations** (5점): `TerminationsCfg`에 `time_out` 등 기본 종료 조건이 있는지
- **Events** (5점): `EventCfg`에 `reset_scene_to_default` 등 이벤트가 있는지

### Task Alignment (25점)

YAML의 `goal`이 코드의 보상/종료 함수에 반영되었는지 평가합니다.

- **Goal 매핑** (10점): goal 조건이 보상 또는 종료 함수로 변환되었는지
- **커스텀 보상 함수** (10점): 태스크 고유의 커스텀 보상 함수가 구현되었는지 (단순 빌트인만 사용하면 감점)
- **Threshold 반영** (5점): YAML의 정량적 threshold가 코드에 반영되었는지

### Runtime Validity (20점)

실제 IsaacLab 환경에서의 실행 가능성을 검증합니다.

- **구문 검사** (5점): Python 구문 오류 없음
- **Import 검증** (5점): 사용된 모듈이 import 가능한지
- **환경 인스턴스화** (5점): `ManagerBasedRLEnv(cfg=...)` 생성 성공 여부
- **Step 실행** (5점): `env.step()` 실행 성공 여부 (5 step 이상)

---

## 실행 방법

### Agent에서 자동 실행

```bash
# 코드 생성 + 실행 + 평가
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
```

### 기존 출력물만 평가

```bash
# Agent 경유
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml \
  --eval-only outputs/isaaclab/FrankaStack_20260212/

# Standalone
python3 scripts/evaluate.py \
  outputs/isaaclab/FrankaStack_20260212/ \
  tasks/franka/stack/franka_stack.yaml
```

### 정적 분석만 (IsaacLab 환경 불필요)

```bash
python3 scripts/evaluate.py --skip-runtime \
  outputs/isaaclab/FrankaStack_20260212/ \
  tasks/franka/stack/franka_stack.yaml
```

`--skip-runtime` 사용 시 Runtime Validity (20점)는 0점으로 처리되고, 나머지 80점 범위에서만 평가됩니다.

---

## 평가 결과 해석

### eval_report.json 예시

```json
{
  "total_score": 87,
  "categories": {
    "scene_fidelity": {
      "score": 30,
      "max": 30,
      "details": {
        "robot_config": { "score": 10, "max": 10, "checks": [...] },
        "object_count": { "score": 10, "max": 10, "checks": [...] },
        "object_properties": { "score": 10, "max": 10, "checks": [...] }
      }
    },
    "mdp_correctness": {
      "score": 22,
      "max": 25,
      "details": { ... }
    },
    "task_alignment": {
      "score": 25,
      "max": 25,
      "details": { ... }
    },
    "runtime_validity": {
      "score": 10,
      "max": 20,
      "details": { ... }
    }
  },
  "checks": [
    { "name": "robot_type_match", "passed": true, "message": "..." },
    { "name": "custom_reward_exists", "passed": true, "message": "..." },
    ...
  ]
}
```

### 점수 기준

| 점수 범위 | 등급 | 의미 |
|-----------|------|------|
| 90-100 | 우수 | YAML을 정확히 구현, 커스텀 보상 포함, 실행 성공 |
| 70-89 | 양호 | 대부분 정확하나 일부 누락 또는 런타임 이슈 |
| 50-69 | 보통 | 기본 구조는 맞으나 goal 매핑 부족 또는 실행 실패 |
| 0-49 | 미흡 | 코드 구조 오류 또는 대부분의 검사 실패 |

`scripts/evaluate.py`의 종료 코드: 70점 이상이면 `0` (합격), 미만이면 `1` (불합격)

---

## 평가 모듈 구조

```
src/isaac_lab/evaluator/
├── __init__.py            # IsaacLabEvaluator (오케스트레이터)
├── parser.py              # GoalNormalizer + EnvCfgParser
│                          #   - GoalNormalizer: YAML goal → 정규화된 condition 리스트
│                          #   - EnvCfgParser: env_cfg.py → AST + regex 파싱
├── scene_fidelity.py      # SceneFidelityChecker (30점)
├── mdp_correctness.py     # MDPCorrectnessChecker (25점)
├── task_alignment.py      # TaskAlignmentChecker (25점)
└── runtime_validity.py    # RuntimeValidityChecker (20점)
```

### 분석 방법

| Phase | 방법 | IsaacLab 필요 |
|-------|------|:------------:|
| **정적 분석** | Python AST 파싱 + regex 패턴 매칭 | X |
| **런타임 분석** | `eval_runner.py` 동적 생성 → conda subprocess 실행 | O |

정적 분석은 `env_cfg.py`의 텍스트를 직접 분석하여:
- `@configclass` 데코레이터가 있는 클래스 추출
- `RewTerm(func=..., weight=..., params=...)` 블록을 괄호 매칭으로 추출
- asset 이름, position, physics 속성을 regex로 추출

런타임 분석은 `eval_runner.py`를 동적으로 생성하여:
- `ManagerBasedRLEnv(cfg=...)` 인스턴스화 시도
- `env.step()` 실행 (지정된 step 수만큼)
- observation 범위, reward 값 등을 `eval_results.json`에 기록

---

## 설정 파일

`configs/isaaclab_eval_config.yaml`에서 평가 파라미터를 조정할 수 있습니다:

```yaml
weights:
  scene_fidelity: 30
  mdp_correctness: 25
  task_alignment: 25
  runtime_validity: 20

tolerances:
  position: 0.05        # m, 위치 허용 오차
  rotation: 0.1          # rad, 회전 허용 오차

runtime:
  eval_steps: 10         # 런타임 평가 step 수
  timeout: 120           # 초, 런타임 타임아웃
```

---

## 관련 문서

- [사용법](usage.md) — `--evaluate`, `--eval-only` 옵션
- [Task YAML 명세](task_yaml_spec.md) — 평가 기준이 되는 YAML 포맷
