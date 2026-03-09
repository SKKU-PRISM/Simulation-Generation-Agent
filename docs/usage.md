# 사용법

대상: 실제로 CLI를 실행하는 사용자  
이 문서가 다루는 것: 대표 명령, 주요 옵션, 출력 구조, 결과 해석  
설치와 환경 연결: `docs/getting_started.md`

이 문서는 **실행 방법과 결과 해석의 source-of-truth**다. 내부 구현 상세나 schema 배경 설명은 다른 문서로 분리한다.

## 1. IsaacLab Pipeline (권장 경로)

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
| --- | --- |
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
├── .success_marker
├── error_attempt_*.txt
└── eval_report.json          # --evaluate 사용 시
```

## 2. Isaac Sim Pipeline

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
| --- | --- |
| `--skip-vlm` | VLM 평가 없이 1회 빌드/캡처만 수행 |
| `--backend {auto,azure,claude,gemini,ollama,mock}` | VLM backend 선택 |
| `--max-iterations <n>` | 최대 반복 횟수 override |
| `--threshold <n>` | 성공 기준 점수 override |
| `--output-dir <dir>` | 출력 루트 override |
| `--config <path>` | `configs/pipeline_config.yaml` 대체 |

### 출력 구조

```text
outputs/isaac_sim/<task_slug>_<timestamp>/
├── iter_01.png
├── iter_02.png
├── rgb_0000.png
├── metadata.txt
└── run_report.json
```

## 3. Data Collection

### 대표 명령어

```bash
# env 자동 생성 후 수집
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml

# 기존 env 사용
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/frankastack_20260219_160916

# 성공 episode 기준
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --target-success 5 --max-attempts 25

# VLM 없이 기하학적 verification만 사용
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --no-vlm-judge

# batch
python3 scripts/run_data_collection.py --batch tasks/franka/ --episodes 10
```

### 주요 옵션

| 옵션 | 설명 |
| --- | --- |
| `--env-dir <path>` | 기존 IsaacLab 출력 디렉토리 사용 |
| `--episodes <n>` | 최대 episode 수 override |
| `--target-success <n>` | 목표 성공 episode 수 |
| `--max-attempts <n>` | 전체 시도 수 상한 |
| `--repo-id <id>` | dataset ID |
| `--fps <n>` | 녹화 FPS override |
| `--no-vlm-judge` | VLM 판정 비활성화 |
| `--gui` | headless 대신 GUI 실행 |
| `--config <path>` | data collection config override |
| `-v`, `--verbose` | 상세 로그 |

### 출력 구조

```text
outputs/data_collection/<TaskName>_<timestamp>/
├── collect_data.py
├── pipeline_config.json
├── cap_runs/
├── debug_initial_*.png
├── raw_dataset/
│   ├── episodes/
│   └── metadata.json
├── collection_results.json
├── COLLECTION_COMPLETE_MARKER
└── <repo_id>/                   # optional LeRobot conversion 결과
```

### 결과 해석

`collection_results.json`에서 먼저 볼 값은 아래다.

| key | 의미 |
| --- | --- |
| `pipeline_completed` | 수집/정리 경로가 끝까지 완료됐는지 |
| `target_met` | 목표 성공 episode 수를 달성했는지 |
| `successful_episodes` | 성공 episode 수 |
| `total_episodes` | 실제 시도/저장된 episode 수 |
| `raw_dataset` | raw dataset 경로 |

해석 규칙:

- `success`는 현재 `pipeline_completed` alias다.
- `pipeline_completed=true`라도 `target_met=false`일 수 있다.
- 기본 완료 기준은 `raw_dataset/` 생성이다. LeRobot 변환은 환경에 따라 스킵될 수 있다.

## 4. Dataset Export / Preprocess

### Export

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets
```

기본 schema는 `adc_compatible`이다.

### Preprocess

```bash
python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

산출물:

- `manifest.json`
- `samples.jsonl`
- `train.jsonl`
- `val.jsonl`

기본 `adc_compatible` 경로에서는 `observation.gripper_state`, `observation.tcp.robot_xyzrpy`, `skill.goal_position.robot_xyzrpy`가 학습 입력 manifest에 포함된다.

dataset field 의미와 schema 차이는 `docs/dataset_alignment_and_export.md`를 본다.

### Optional: Local LeRobot conversion

```bash
python3 scripts/convert_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --repo-id local/franka_stack_sim
python3 scripts/check_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/local/franka_stack_sim \
  --repo-id local/franka_stack_sim
```

`lerobot` 패키지가 host Python에 설치되어 있어야 한다.

### Optional: Publish to Hub

```bash
python3 scripts/publish_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/local/franka_stack_sim \
  --repo-id <org>/<dataset_name> \
  --local-repo-id local/franka_stack_sim \
  --private
```

기본 인증은 `HF_TOKEN` env var 또는 기존 `huggingface-cli login` 세션을 사용한다.

## 5. 추천 운영 순서

```bash
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --env-dir outputs/isaaclab/<run_dir>
python3 scripts/export_dataset.py outputs/data_collection/<run_dir>/raw_dataset --source-type sim_raw --output-dir outputs/exported_datasets
python3 scripts/preprocess_dataset.py outputs/exported_datasets/<export_dir> --output-dir outputs/preprocessed_datasets
```

시각 검증이 필요할 때만 `scripts/run_isaac_sim.py`를 추가한다.

## 관련 문서

- `docs/getting_started.md`
- `docs/evaluation.md`
- `docs/dataset_alignment_and_export.md`
- `docs/troubleshooting.md`
