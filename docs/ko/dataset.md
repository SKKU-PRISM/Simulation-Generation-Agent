# 데이터셋 워크플로우

[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../dataset.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](dataset.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](../zh/dataset.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](../ja/dataset.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](../de/dataset.md)

이 가이드는 파이프라인 실행부터 학습 가능한 데이터셋을 HuggingFace에 업로드하는 전체 워크플로우를 다룹니다.

---

## 개요

```
run_agent.sh "task description"
    |
    v
outputs/data_collection/<run>/raw_dataset/     <-- Stage 3 출력
    |
    v  scripts/export_dataset.py
outputs/exported_datasets/<name>/              <-- 정규화된 학습 스키마
    |
    v  scripts/preprocess_dataset.py
outputs/preprocessed_datasets/<name>/          <-- train.jsonl / val.jsonl
    |
    v  scripts/convert_lerobot_dataset.py
outputs/lerobot_datasets/<repo_id>/            <-- LeRobot v3.0 형식 (Parquet)
    |
    v  scripts/publish_lerobot_dataset.py
https://huggingface.co/datasets/<org>/<name>   <-- HuggingFace Hub
```

---

## 1단계: 파이프라인 실행

```bash
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

Stage 3이 완료되면 결과가 다음 위치에 저장됩니다:
- **로컬**: `outputs/data_collection/<TaskName>_<timestamp>/`
- **Docker**: `/workspace/artifacts/data_collection/` (`./artifacts/`에 마운트됨)

### 출력 확인

```bash
ls outputs/data_collection/
# FrankaStackTray_20260402_151823/

ls outputs/data_collection/FrankaStackTray_20260402_151823/
# collection_results.json   raw_dataset/   videos/

cat outputs/data_collection/FrankaStackTray_20260402_151823/collection_results.json
```

`collection_results.json` 파일에는 다음 정보가 포함됩니다:
- `pipeline_completed`: 데이터 수집 완료 여부
- `geometry_successful_episodes`: 기하학적 검증을 통과한 에피소드 수
- `total_episodes`: 시도한 총 에피소드 수

### Raw 데이터셋 구조

```
raw_dataset/
├── metadata.json              # 로봇 설정, DOF, FPS, 카메라
├── episodes/
│   ├── episode_000000/
│   │   ├── states.npy         # (T, N_dof) 관절 위치
│   │   ├── actions.npy        # (T, N_dof) 명령된 위치
│   │   ├── tcp_world_xyzrpy.npy  # (T, 6) 월드 프레임 기준 TCP
│   │   ├── tcp_robot_xyzrpy.npy  # (T, 6) 로봇 베이스 프레임 기준 TCP
│   │   ├── gripper_state.npy  # (T, 1) 그리퍼 상태
│   │   ├── goal_robot_xyzrpy.npy # (T, 6) 프레임별 목표 자세
│   │   ├── skills.json        # 프레임별 스킬 메타데이터
│   │   └── images/
│   │       ├── top/           # 000000.png, 000001.png, ...
│   │       └── wrist/
│   └── episode_000001/
│       └── ...
```

---

## 2단계: 내보내기 (선택사항)

Raw 데이터를 일관된 관절 정규화가 적용된 학습 친화적 스키마로 변환합니다.

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets
```

옵션:
- `--schema adc_compatible` (기본값): 관절값을 [-100, 100] 범위로 정규화, ADC 호환 필드명 사용
- `--schema canonical_training`: 명시적 TCP current/goal 자세를 포함하는 깔끔한 스키마
- `--no-link-images`: 심볼릭 링크 대신 이미지를 복사

---

## 3단계: 전처리 (선택사항)

프레임 단위 학습 매니페스트를 train/val 분할과 함께 생성합니다.

```bash
python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets \
  --success-only \
  --train-ratio 0.9
```

출력:
- `train.jsonl` / `val.jsonl` -- 파일 경로가 포함된 프레임별 샘플
- `manifest.json` -- 스키마 및 통계 정보

---

## 4단계: LeRobot 형식으로 변환

Raw 데이터를 [LeRobot v3.0](https://github.com/huggingface/lerobot) 형식 (Apache Parquet)으로 변환합니다.

### 사전 요구사항

```bash
pip install "lerobot>=0.4.0,<0.5.0"
```

### 변환

```bash
python3 scripts/convert_lerobot_dataset.py \
  outputs/data_collection/<run>/raw_dataset \
  --repo-id local/franka_stack_sim \
  --output-root outputs/lerobot_datasets
```

출력 구조:
```
outputs/lerobot_datasets/local/franka_stack_sim/
├── meta/
│   └── info.json           # LeRobot 메타데이터, 피처, 프레임 수
├── data/
│   └── chunk-0000/
│       └── file-00000.parquet
└── videos/                 # (비디오 녹화가 활성화된 경우)
```

### 검증

```bash
python3 scripts/check_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack_sim \
  --repo-id local/franka_stack_sim
```

검증 항목:
- 필수 피처: `observation.state`, `observation.gripper_state`, `observation.tcp.robot_xyzrpy`, `action`, `skill.goal_position.robot_xyzrpy`
- Parquet 샤드 무결성
- 프레임 수 > 0
- `meta/info.json` 존재 여부

---

## 5단계: HuggingFace에 업로드

### 사전 요구사항

```bash
pip install huggingface_hub
```

HuggingFace 토큰 설정:
```bash
export HF_TOKEN="hf_your_token_here"
# 또는: huggingface-cli login
```

### 게시

```bash
python3 scripts/publish_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack_sim \
  --repo-id your-org/franka-stack-sim \
  --local-repo-id local/franka_stack_sim \
  --private
```

옵션:
- `--private`: 비공개 데이터셋으로 생성 (기본값: 공개)
- `--token-env HF_TOKEN`: 인증을 위한 환경 변수 (기본값)
- `--token <token>`: 토큰을 직접 전달

이 스크립트는 다음을 수행합니다:
1. 로컬 데이터셋 검증
2. HuggingFace 데이터셋 저장소 생성 (존재하지 않는 경우)
3. `upload_large_folder()`를 사용하여 모든 파일 업로드
4. 저장소 URL이 포함된 JSON 리포트 출력

---

## 빠른 참조

### 올인원 (Raw에서 HuggingFace까지)

```bash
# 1. 파이프라인 실행
./run_agent.sh "Stack the blocks" --robot franka --episodes 5

# 2. 출력 디렉토리 찾기
RUN_DIR=$(ls -td outputs/data_collection/*/ | head -1)

# 3. LeRobot으로 변환
python3 scripts/convert_lerobot_dataset.py \
  ${RUN_DIR}/raw_dataset \
  --repo-id local/franka_stack \
  --output-root outputs/lerobot_datasets

# 4. 검증
python3 scripts/check_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack \
  --repo-id local/franka_stack

# 5. 업로드
python3 scripts/publish_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack \
  --repo-id your-org/franka-stack --private
```

### 배치 파이프라인 (자동화)

`e2e-batch` 모드를 사용하면 LeRobot 변환 및 HuggingFace 업로드를 포함한 전체 흐름을 자동화할 수 있습니다:

```bash
./run_agent.sh --mode e2e-batch --config configs/docker/e2e_batch_release.yaml
```

배치 설정 YAML에서 업로드를 구성합니다:
```yaml
hf:
  upload: true
  namespace: your-org
  dataset_name: franka-sim-dataset
  private: true
  token_env: HF_TOKEN
```

---

## LeRobot 데이터셋 피처

변환된 데이터셋의 각 프레임에는 다음이 포함됩니다:

| 피처 | Shape | 설명 |
|------|-------|------|
| `observation.state` | (N_dof,) | 관절 위치 |
| `observation.gripper_state` | (1,) | 그리퍼 상태 |
| `observation.tcp.world_xyzrpy` | (6,) | 월드 프레임 기준 TCP 자세 |
| `observation.tcp.robot_xyzrpy` | (6,) | 로봇 베이스 프레임 기준 TCP 자세 |
| `action` | (N_dof,) | 명령된 관절 위치 |
| `skill.natural_language` | (1,) | 스킬 설명 |
| `skill.type` | (1,) | 스킬 유형 (pick, place 등) |
| `skill.progress` | (1,) | 스킬 진행도 [0, 1] |
| `skill.goal_position.joint` | (N_dof,) | 목표 관절 위치 |
| `skill.goal_position.world_xyzrpy` | (6,) | 월드 프레임 기준 목표 자세 |
| `skill.goal_position.robot_xyzrpy` | (6,) | 로봇 베이스 프레임 기준 목표 자세 |
| `skill.goal_position.gripper` | (1,) | 목표 그리퍼 상태 |
| `observation.images.{cam}` | (480, 640, 3) | 카메라 이미지 (top, wrist) |

**로봇별 N_dof:** Franka=9, UR10e=12, OpenARM=9, SO-101=6

---

## 관련 문서

- 파이프라인 아키텍처: [architecture.md](architecture.md)
- CLI 사용법: [usage.md](usage.md)
- 설치 가이드: [getting_started.md](getting_started.md)
