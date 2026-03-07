# 사용법

이 레포의 기본 운영 경로는 **IsaacLab Pipeline**입니다. Isaac Sim은 시각 검증용, Data Collection은 IsaacLab 위 확장 경로로 사용합니다.

## 1. IsaacLab Pipeline (권장 경로)

### 개요

```text
task YAML
  -> IsaacLabAgent
  -> Azure OpenAI로 env_cfg.py / run_env.py / mdp/ 생성
  -> isaaclab.sh 실행
  -> 필요 시 자동 재생성
  -> eval_report.json 생성
```

### 대표 명령어

```bash
# 생성 + 실행
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml

# 생성만
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run

# 생성 + 실행 + 평가
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate

# 기존 출력물 재평가
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml \
  --eval-only outputs/isaaclab/frankastack_20260219_160916

# standalone evaluator
python3 scripts/evaluate.py \
  outputs/isaaclab/frankastack_20260219_160916 \
  tasks/franka/stack/franka_stack.yaml

# batch
python3 scripts/run_isaac_lab.py --batch tasks/franka/
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `--dry-run` | 코드만 생성하고 IsaacLab 실행은 생략 |
| `--evaluate` | 성공 후 evaluator 실행 |
| `--eval-only <dir>` | 기존 생성 결과만 평가 |
| `--batch <dir>` | 디렉토리 내 YAML 일괄 처리 |
| `--output-dir <dir>` | 기본 `outputs/isaaclab` 대신 다른 출력 루트 사용 |
| `--config <path>` | agent config override |

### 출력 구조

```text
outputs/isaaclab/<task_slug>_<timestamp>/
├── env_cfg.py
├── run_env.py
├── mdp/
│   ├── __init__.py
│   ├── rewards.py            # 선택
│   ├── terminations.py       # 선택
│   └── observations.py       # 선택
├── .success_marker
├── error_attempt_1.txt       # 실패 시
├── error_attempt_2.txt       # 실패 시
└── eval_report.json          # --evaluate 사용 시
```

## 2. Isaac Sim Pipeline (시각 검증)

### 개요

현재는 자동 러너가 구현되어 있습니다.

```text
task YAML
  -> SceneBuilder
  -> ScreenshotCapture
  -> VLM Evaluator
  -> threshold 이상이면 종료, 아니면 재시도
  -> run_report.json 저장
```

### 대표 명령어

```bash
# 자동 빌드 + 스크린샷 + VLM 평가
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --backend auto

# VLM 없이 빌드 + 캡처 1회
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --skip-vlm

# 반복/기준점수 override
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml \
  --backend azure --max-iterations 3 --threshold 85
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `--skip-vlm` | VLM 평가 없이 1회 빌드/캡처만 수행 |
| `--backend {auto,azure,claude,gemini,ollama,mock}` | VLM backend 선택 |
| `--max-iterations <n>` | 최대 반복 횟수 override |
| `--threshold <n>` | 성공 기준 점수 override |
| `--output-dir <dir>` | 출력 루트 override |
| `--config <path>` | `configs/pipeline_config.yaml` 대체 |

### 저수준 명령

```bash
# 씬 빌드만
python3 scripts/build_scene.py tasks/franka/stack/franka_stack.yaml

# 컴포넌트 점검
python3 tests/test_components.py connection
python3 tests/test_components.py scene tasks/franka/stack/franka_stack.yaml
python3 tests/test_components.py screenshot outputs/test.png
python3 tests/test_components.py vlm <screenshot_path> <task_yaml_path>
python3 tests/test_components.py full --skip-vlm
```

### 출력 구조

```text
outputs/isaac_sim/<task_slug>_<timestamp>/
├── iter_01.png
├── iter_02.png
├── rgb_0000.png             # capture fallback/generated frames
├── metadata.txt
└── run_report.json
```

`run_report.json`에는 `success`, `iterations`, `best_score`, `best_iteration`, `threshold`가 저장됩니다.

## 3. Standalone 평가

```bash
# 정적 + 런타임 평가
python3 scripts/evaluate.py \
  outputs/isaaclab/<run_dir> \
  tasks/franka/stack/franka_stack.yaml

# 정적 분석만
python3 scripts/evaluate.py --skip-runtime \
  outputs/isaaclab/<run_dir> \
  tasks/franka/stack/franka_stack.yaml
```

종료 코드:

- `0`: 총점 70 이상
- `1`: 총점 70 미만

## 4. Data Collection (IsaacLab 확장)

### 개요

```text
task YAML
  -> 기존 IsaacLab env 사용 또는 자동 생성
  -> collect_data.py 동적 생성
  -> IsaacLab subprocess 실행
  -> detect -> plan (LLM) -> execute (6-DOF IK) -> judge (VLM / geometric) -> record
  -> raw_dataset + collection_results.json 저장
```

### 대표 명령어

```bash
# env 자동 생성 후 수집
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml

# 기존 env 사용
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/frankastack_20260219_160916

# 성공 에피소드 수 기준 (5개 성공할 때까지 최대 25회 시도)
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --target-success 5 --max-attempts 25

# 에피소드 수 override
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --episodes 10 --repo-id local/franka_stack

# VLM 없이 기하학적 goal verification만 사용
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --no-vlm-judge

# GUI 모드
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --gui

# batch
python3 scripts/run_data_collection.py --batch tasks/franka/ --episodes 10
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `--env-dir <path>` | 기존 IsaacLab 출력 디렉토리 사용 |
| `--episodes <n>` | 최대 episode 수 override |
| `--target-success <n>` | 목표 성공 episode 수 (실패 시 discard) |
| `--max-attempts <n>` | 전체 시도 수 상한 |
| `--repo-id <id>` | dataset ID |
| `--fps <n>` | 녹화 FPS override |
| `--no-vlm-judge` | VLM 판정 비활성화 (geometric verification 유지) |
| `--gui` | headless 대신 GUI 실행 |
| `--config <path>` | data collection config override |
| `-v`, `--verbose` | 상세 로그 |

### 성공 판정 (3-tier)

에피소드 성공 여부는 다음 우선순위로 결정됩니다:

1. **VLM Judge** (최우선): wrist + front 카메라의 before/after 이미지를 Azure OpenAI gpt-5로 비교
2. **Geometric Goal Verification**: 오브젝트 위치를 scene graph에서 재쿼리하여 기하학적 조건 확인 (stacking: XY alignment + Z height diff)
3. **Env Termination**: IsaacLab 환경의 success termination condition 확인

`--no-vlm-judge` 사용 시 VLM을 건너뛰고 2, 3 경로만 사용합니다.

### Multi-Camera 시스템

로봇 프로필 (`configs/robot_profiles/*.yaml`)의 `cameras:` 섹션에서 정의:

| 카메라 | 용도 | 데이터셋 포함 |
|--------|------|:---:|
| `top` | Near-top-down 개요 (recording) | Yes |
| `wrist` | EE body-mounted 근접 뷰 (recording + VLM judge) | Yes |
| `front` | 정면 뷰 (VLM judge 전용) | No |

### 6-DOF IK + Safe Retreat

- **Pinocchio IK** (primary): `num_random_samples=30`, orientation tolerance 0.15rad (~8.6°), tilt fallback ±15°/±30°
- **Safe retreat**: `move_to_ready(safe_retreat=True)` — EE를 z≥0.45m까지 직상방으로 올린 후 ready pose 복귀 (스택 붕괴 방지)

### 출력 구조

```text
outputs/data_collection/<TaskName>_<timestamp>/
├── collect_data.py              # 동적 생성 수집 스크립트
├── pipeline_config.json
├── debug_initial_*.png          # 첫 에피소드 카메라 디버그 이미지
├── raw_dataset/
│   ├── episodes/
│   │   └── episode_000000/
│   │       ├── states.npy       # (T, N_dof) float32
│   │       ├── actions.npy      # (T, N_dof) float32
│   │       ├── images/          # {top/, wrist/} (front 제외)
│   │       └── skills.json      # per-frame skill + goal metadata
│   └── metadata.json
├── collection_results.json
└── COLLECTION_SUCCESS_MARKER
```

### ADC 서브모듈

```bash
pip install -e ".[data-collection]"
pip install pin                    # Pinocchio (6-DOF IK 필수)
git submodule update --init external/AutoDataCollector
```

ADC가 없으면:

- IK: Pinocchio 직접 설치 필요 (`pip install pin`)
- Judge prompt: 내장 prompt fallback
- interpolation: 내부 구현 fallback

## 5. 추천 운영 순서

### IsaacLab 중심

```bash
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
```

### 시각 검증이 필요할 때만 Isaac Sim

```bash
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --backend auto
```

### 수집으로 확장할 때

```bash
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/<run_dir>
```

## 관련 문서

- `docs/getting_started.md`
- `docs/evaluation.md`
- `docs/task_yaml_spec.md`
- `docs/troubleshooting.md`
