# 사용법

[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../usage.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](usage.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](../zh/usage.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](../ja/usage.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](../de/usage.md)

대상: 실제로 CLI를 실행하는 사용자  
이 문서가 다루는 것: 대표 명령, 주요 옵션, 출력 구조, 결과 해석  
설치와 환경 연결: `docs/getting_started.md`

이 문서는 **실행 방법과 결과 해석의 source-of-truth**다. 내부 구현 상세나 schema 배경 설명은 다른 문서로 분리한다.
파이프라인 아키텍처: [docs/architecture.md](architecture.md)

## run_agent.sh — 전체 파이프라인 실행 (메인 기능)

자연어 태스크 설명 하나만 넣으면 NL→YAML→IsaacLab→DataCollection 전체 파이프라인이 자동 실행됩니다.

```bash
# 자연어 입력
./run_agent.sh "Stack the blocks inside the tray on the table"
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5

# JSON 입력
./run_agent.sh data/input_sample.json results/output.json
```

### 옵션

| 옵션 | 설명 |
| --- | --- |
| `--robot <type>` | 로봇 종류: franka, ur10e, openarm, so101 (기본: franka) |
| `--episodes <n>` | 목표 성공 에피소드 수 (기본: 1) |
| `--max-attempts <n>` | 최대 시도 횟수 (기본: 3) |

### Docker에서 실행

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --episodes 5
```

> **출력 경로**: Docker 내부에서는 `/workspace/artifacts`에 결과가 저장됩니다. `-v` 옵션으로 호스트에 매핑하세요. 로컬 실행 시에는 `outputs/`에 저장됩니다.

### 결과

`results/output.json`에 구조화된 JSON으로 기록됩니다:

```json
{
  "status": "completed",
  "tasks": [{
    "name": "StackTheBlocksInsideTheTrayOnThe",
    "steps": {
      "nl_to_yaml": {"success": true},
      "yaml_to_isaaclab": {"success": true},
      "data_collection": {"success": true, "success_episodes": 5}
    }
  }]
}
```

---

## 0. Task Spec Agent (NL → YAML)

자연어 태스크 설명을 구조화된 YAML 태스크 명세로 변환합니다 (파이프라인 Stage 1).

> **참고**: `run_agent.sh "자연어 태스크"`를 사용하면 Stage 1→2→3 전체가 자동 실행됩니다.
> 아래는 Stage 1만 개별 실행하는 방법입니다.

### 대표 명령어

```bash
# 기본 사용
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube and place it on the target" \
  --robot franka --output task.yaml

# RAG 없이 템플릿 기반 생성
python3 scripts/task_spec_agent/task_spec_agent.py "Stack blocks" --no-rag

# 다른 LLM 프로바이더 사용
python3 scripts/task_spec_agent/task_spec_agent.py "Sort the colored blocks into matching colored bins" --provider huggingface

# 상세 로그
python3 scripts/task_spec_agent/task_spec_agent.py "Reach the goal" --verbose
```

### 주요 옵션

| 옵션 | 설명 |
| --- | --- |
| `--robot {franka,openarm,ur10,so101}` | 대상 로봇 (기본: franka) |
| `--output <path>` | YAML 저장 경로 (미지정 시 stdout 출력) |
| `--provider {azure,huggingface,bedrock}` | LLM 프로바이더 override (미지정 시 config 기본값 사용, default=openai) |
| `--no-rag` | RAG 대신 템플릿 기반 YAML 생성 |
| `--verbose` | DEBUG 레벨 로깅 |

### 내부 처리 단계

```
자연어 입력
  → NL Parser (actions, objects, locations 추출)
  → Task Decomposer (원자적 동작 시퀀스 분해)
  → Feasibility Validator (로봇 물리적 실현 가능성 검증)
  → RAG YAML Generator (FAISS 벡터 검색 → 최근접 태스크 YAML 매칭)
  → task.yaml
```

## 1. IsaacLab Pipeline (Stage 2)

> **run_agent.sh로 실행**: `./run_agent.sh --mode isaac-lab --task <yaml>`
> 아래는 Python 스크립트 직접 실행 방법입니다.

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
├── debug/                    # 환경 스크린샷 (front/top/wrist)
├── result.json               # 실행 결과 + scene_verification 포함
└── eval_report.json          # --evaluate 사용 시
```

## 2. Isaac Sim Pipeline (로컬 개발 환경 전용)

> **참고**: Isaac Sim MCP 시각 검증은 로컬에서 Isaac Sim Desktop이 실행 중일 때만 사용 가능합니다. Docker 환경에서는 지원되지 않습니다.

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

## 3. Data Collection (Stage 3)

> **run_agent.sh로 실행**: `./run_agent.sh --mode data-collection --task <yaml>`
> 아래는 Python 스크립트 직접 실행 방법입니다.

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

기본 `adc_compatible` schema로 export되며, `canonical_training`은 확장 분석용이다.

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

### NL → 데이터셋 (Full Pipeline)

```bash
# 1. NL → YAML
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube and stack it" \
  --robot franka --output outputs/generated_task.yaml

# 2. YAML → IsaacLab 환경 코드
python3 scripts/run_isaac_lab.py outputs/generated_task.yaml --evaluate

# 3. 데이터 수집
python3 scripts/run_data_collection.py outputs/generated_task.yaml \
  --env-dir outputs/isaaclab/<run_dir> --target-success 10

# 4. Export / Preprocess
python3 scripts/export_dataset.py outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw --output-dir outputs/exported_datasets
python3 scripts/preprocess_dataset.py outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

### 기존 YAML → 데이터셋

```bash
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --env-dir outputs/isaaclab/<run_dir>
python3 scripts/export_dataset.py outputs/data_collection/<run_dir>/raw_dataset --source-type sim_raw --output-dir outputs/exported_datasets
python3 scripts/preprocess_dataset.py outputs/exported_datasets/<export_dir> --output-dir outputs/preprocessed_datasets
```

시각 검증이 필요할 때만 `scripts/run_isaac_sim.py`를 추가한다.

## 6. Full Pipeline (NL → Video)

`run_full_test.sh`는 13개 Franka 태스크에 대해 Stage 1→2→3을 순차적으로 실행하는 통합 스크립트다.

```bash
bash scripts/run_full_test.sh
```

내부적으로 각 태스크마다:
1. `task_spec_agent.py` — 자연어 → YAML 생성
2. `run_isaac_lab.py` — YAML → IsaacLab 환경 코드 생성/검증
3. `run_data_collection.py` — CaP 코드 생성 → 실행 → 성공 판정 → 데이터 수집

### 출력 구조

```text
outputs/test_run_<timestamp>/
├── summary.txt                  # 전체 태스크 결과 요약
├── token_usage.jsonl            # API 토큰 사용량 (TOKEN_USAGE_FILE 설정 시)
├── FrankaLift/
│   ├── step1_nl_to_yaml.log
│   ├── step2_isaaclab.log
│   ├── step3_cap_execution.log
│   └── task.yaml
├── FrankaStack/
│   └── ...
└── ...
```

### 토큰 사용량 추적

```bash
export TOKEN_USAGE_FILE=outputs/token_usage.jsonl
export TOKEN_USAGE_LOG=1  # 실시간 콘솔 로그
bash scripts/run_full_test.sh
```

## 7. E2E Batch Pipeline

config 기반으로 다수 태스크의 환경 생성 + 데이터 수집 + export + LeRobot 변환을 일괄 처리한다.

### 대표 명령어

```bash
# Batch 실행 (Docker)
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml

# 중단 후 재개
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml --resume

# 조용한 로그
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml -q
```

### Docker / run_agent.sh

```bash
# Docker 내부 (기본 모드: e2e-batch)
run_agent.sh --mode e2e-batch --resume

# 단일 태스크
run_agent.sh --mode isaac-lab --task tasks/franka/lift/franka_lift.yaml

# 데이터 수집
run_agent.sh --mode data-collection --task tasks/franka/lift/franka_lift.yaml -- --episodes 10
```

### Config 구조

E2E batch config YAML은 4개 섹션으로 구성된다:

| 섹션 | 역할 |
| --- | --- |
| `run` | output_root, resume, cleanup, export 설정 |
| `collection` | LLM/VLM 모델, max_attempts, timeout, VLM judge 사용 여부 |
| `hf` | HuggingFace Hub 업로드 설정 (선택) |
| `tasks` | 태스크 목록 (YAML 경로, 목표 demos 수, 활성화 여부) |

### 주요 옵션

| 옵션 | 설명 |
| --- | --- |
| `--resume` | 이전 실행의 성공 태스크를 건너뛰고 실패/미완료 태스크만 재실행 |
| `-q`, `--quiet` | INFO 레벨 로깅 (기본: DEBUG) |

### 출력 구조

```text
<output_root>/
├── config_snapshot.yaml         # 실행에 사용된 config 사본
├── task_runs/                   # 태스크별 수집 결과
├── task_reports/                # 태스크별 JSON 리포트
├── successful_raw/              # 성공 에피소드만 모은 raw dataset
├── exported/                    # export 결과
├── preprocessed/                # preprocess 결과
├── lerobot/                     # LeRobot 변환 결과
├── batch_report.md              # Markdown 요약 리포트
└── batch_report.json            # JSON 리포트
```

## 관련 문서

- 파이프라인 아키텍처: `docs/architecture.md`
- 설치: `docs/getting_started.md`
