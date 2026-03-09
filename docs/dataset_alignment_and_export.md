# Dataset Alignment and Export

대상: dataset을 export/preprocess하거나 학습 입력을 설계하는 사용자  
이 문서가 다루는 것: raw dataset 의미, schema 차이, export/preprocess 기본 경로  
실제 CLI 사용법은 `docs/usage.md`

이 문서는 dataset schema의 source-of-truth다. 결론은 단순하다. **raw를 그대로 섞지 말고 export 후 preprocess를 거쳐 학습에 사용한다.** 기본 export는 `adc_compatible`이다.

## 1. 왜 그대로 섞으면 안 되는가

| 항목 | ADC raw | sim raw | 바로 섞을 때 문제 |
| --- | --- | --- | --- |
| `observation.state` | normalized `[-100, 100]` | native joint units | scale mismatch |
| `action` | normalized `[-100, 100]` | native joint target | scale mismatch |
| `skill.goal_position.joint` | normalized `[-100, 100]` | native full-DOF goal joint | scale mismatch |
| `skill.goal_position.gripper` | normalized gripper | native gripper target | 의미/범위 mismatch |
| `skill.goal_position.world_xyzrpy` | ADC legacy world pose | sim world TCP goal | frame semantics mismatch |
| `skill.goal_position.robot_xyzrpy` | calibrated `base_link` 기준 | articulation root 기준 | base frame semantics mismatch |
| current TCP observation | 기본 raw schema에 없음 | `tcp_world_xyzrpy.npy`, `tcp_robot_xyzrpy.npy` 존재 | target/current 혼동 |

중요한 점:

- ADC의 `skill.goal_position.*`는 **current observation이 아니라 subgoal target**이다.
- current TCP는 원본 ADC 기본 schema에 없다.
- 대신 이 레포의 기본 `adc_compatible`는 학습에 필요한 최소 확장으로 `observation.tcp.robot_xyzrpy`와 `observation.gripper_state`를 포함한다.

## 2. 기본 경로: `adc_compatible`

목적:

- 기존 ADC field 이름/의미를 최대한 유지
- real-world ADC와 맞춘 기본 export / 기본 학습 입력
- 비교, 회귀 확인, 기존 데이터와의 parity 점검

특징:

- `state/action/goal_joint/gripper`를 `[-100, 100]`로 맞춘다.
- `goal_world_xyzrpy`는 ADC legacy semantics로 다시 구성한다.
- `observation.tcp.robot_xyzrpy`를 기본 포함한다.
- `observation.gripper_state`를 기본 포함한다.
- `state/action/goal_joint`는 **full controllable DOFs**를 유지한다.
- 기본 학습은 **per-robot**으로만 진행한다.

## 3. 확장 경로: `canonical_training`

이 schema는 world-frame TCP나 clean current/goal TCP semantics가 필요한 분석/실험용이다.

포함되는 추가 정보:

- `observation.tcp.world_xyzrpy`
- `observation.tcp.robot_xyzrpy`
- `skill.goal_position.tcp.world_xyzrpy`
- `skill.goal_position.tcp.robot_xyzrpy`

strict ADC parity가 필요한 기본 경로에는 쓰지 않는다.

## 4. 좌표계 기준

raw 기록과 export의 source-of-truth는 아래다.

- `world` = Isaac `/World`
- `robot` = articulation root
- `tcp` = `ee_frame_tcp`가 있으면 그 frame, 없으면 `ee_frame_body + offset_position`

raw dataset에는 clean simulator semantics를 남긴다.

- `goal_world_xyzrpy` = intended TCP world target
- `goal_robot_xyzrpy` = robot-base TCP target

`adc_compatible` export에서는 여기서 ADC legacy semantics를 재구성한다.

## 5. embodiment 차이 처리

기본 원칙:

- `state/action/goal_joint`는 robot별 full controllable DOFs 유지
- `goal_gripper`는 별도 scalar
- 학습은 per-robot으로만 진행
- export manifest에는 아래를 기록
  - `arm_joint_names`
  - `finger_joint_names`
  - `gripper_type`
  - `joint_shape_policy`
  - field semantics

즉 multi-embodiment 공통 tensor space를 기본값으로 두지 않는다.

## 6. 기본 학습 입력 스키마

기본 학습 입력은 `adc_compatible` 기준이다.

포함:

- `observation.state`
- `observation.gripper_state`
- `observation.tcp.robot_xyzrpy`
- `action`
- `observation.images.*`
- `skill.goal_position.joint`
- `skill.goal_position.world_xyzrpy`
- `skill.goal_position.robot_xyzrpy`
- `skill.goal_position.gripper`

제외:

- `observation.tcp.world_xyzrpy`
- `skill.goal_position.tcp.*`

## 7. 기본 실행 경로

### raw -> export

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets
```

기본 schema는 `adc_compatible`다.

### export -> preprocess

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

### raw -> local LeRobot

```bash
python3 scripts/convert_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --repo-id local/<dataset_name>
python3 scripts/check_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/local/<dataset_name> \
  --repo-id local/<dataset_name>
```

이 경로는 `lerobot` 패키지가 host Python에 설치되어 있어야 한다.

### local LeRobot -> Hub

```bash
python3 scripts/publish_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/local/<dataset_name> \
  --repo-id <org>/<dataset_name> \
  --local-repo-id local/<dataset_name> \
  --private
```

업로드는 기본적으로 `HF_TOKEN` env var 또는 기존 `huggingface-cli login` 세션을 사용한다.

## 8. 현재 검증된 것

현재 레포에서는 sim raw export/preprocess까지 검증했다.

- `outputs/exported_datasets/FrankaStack_20260308_083022_raw_dataset_adc_compatible`
- `outputs/exported_datasets/FrankaStack_20260308_083022_raw_dataset_canonical_training`
- `outputs/exported_datasets/OpenArmStack_20260308_083819_raw_dataset_canonical_training`
- `outputs/exported_datasets/SO101Stack_20260308_083356_raw_dataset_canonical_training`
- `outputs/preprocessed_datasets/raw_dataset_adc_compatible_preprocessed`

LeRobot 변환/업로드는 코드 경로가 준비되어 있지만, 실제 host 검증은 `lerobot` 설치와 Hub 인증이 준비된 환경에서 수행해야 한다.

## 9. Real ADC 검증 재개 조건

이 머신에서는 아직 real ADC dataset 실제 검증을 하지 않았다.

필요한 것:

- 실제 ADC dataset root
- `pyarrow`, `pandas`, `datasets`, `lerobot`
- 적절한 ADC robot config

예시:

```bash
python3 scripts/export_dataset.py \
  <adc_dataset_root> \
  --source-type adc_raw \
  --schema adc_compatible \
  --output-dir outputs/exported_datasets \
  --robot so101 \
  --adc-robot-config external/AutoDataCollector/robot_configs/robot/so101_robot3.yaml
```
