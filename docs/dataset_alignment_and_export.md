# Dataset Alignment and Export

이 문서는 **simulator에서 수집한 raw dataset**과 **ADC/LeRobot raw dataset**을 학습 전에 어떻게 맞춰야 하는지 정리한다.  
결론은 단순하다. **raw를 그대로 섞지 말고, export를 거친 뒤 학습해야 한다.**

## 왜 그대로 섞으면 안 되는가

현재 두 경로는 이름이 비슷해도 의미가 다르다.

| 항목 | ADC raw | sim raw | 바로 섞을 때 문제 |
|---|---|---|---|
| `observation.state` | normalized `[-100, 100]` | native joint units (`rad`, prismatic/servo native) | scale mismatch |
| `action` | normalized `[-100, 100]` | native joint target | scale mismatch |
| `skill.goal_position.joint` | normalized `[-100, 100]` | native full-DOF goal joint | scale mismatch |
| `skill.goal_position.gripper` | normalized gripper | native gripper target | 의미/범위 mismatch |
| `skill.goal_position.world_xyzrpy` | ADC legacy world pose | sim world TCP goal | frame semantics mismatch |
| `skill.goal_position.robot_xyzrpy` | calibrated `base_link` 기준 | articulation root 기준 | base frame semantics mismatch |
| current TCP observation | 기본 raw schema에 없음 | `tcp_world_xyzrpy.npy`, `tcp_robot_xyzrpy.npy` 존재 | target/current 혼동 |

중요한 점:

- ADC의 `skill.goal_position.*`는 **현재 TCP observation이 아니라 subgoal target**이다.
- 기존 ADC의 `goal_world_xyzrpy`는 strict한 world pose로 보면 안 된다.  
  position은 world 기준이지만 orientation은 robot FK 관례를 그대로 쓰는 legacy semantics가 있다.
- 따라서 `Image + State + Goal + Action`으로 학습할 때 raw를 그대로 합치면 **scale / frame / target-current semantics**가 동시에 꼬인다.

## 현재 레포의 권장 경로

학습 전 단계는 두 개로 나눈다.

### 1. `adc_compatible`

목적:

- 기존 ADC feature 이름/의미를 최대한 유지
- 비교, 회귀 확인, 기존 데이터와의 parity 점검

특징:

- `state/action/goal_joint/gripper`를 `[-100, 100]`로 맞춘다.
- `goal_world_xyzrpy`는 **ADC legacy semantics**로 다시 구성한다.
- current TCP observation은 core schema에 넣지 않는다.

이 schema는 **비교용**이다. 학습 기본 schema로 쓰지 않는다.

### 2. `canonical_training`

목적:

- 실제 VLA 학습용
- scale / frame / target-current semantics를 명확하게 고정

특징:

- `observation.state`
- `action`
- `skill.goal_position.joint`
- `skill.goal_position.gripper`

를 모두 `[-100, 100]`로 맞춘다.

그리고 TCP를 명시적으로 분리한다.

- current TCP
  - `observation.tcp.world_xyzrpy`
  - `observation.tcp.robot_xyzrpy`
- target TCP
  - `skill.goal_position.tcp.world_xyzrpy`
  - `skill.goal_position.tcp.robot_xyzrpy`

학습에는 **`canonical_training`만 사용**한다.

## 구현 위치

- exporter core: `src/data_collection/dataset_export.py`
- CLI: `scripts/export_dataset.py`

현재 지원:

- `sim_raw -> adc_compatible`
- `sim_raw -> canonical_training`
- `adc_raw -> adc_compatible`
- `adc_raw -> canonical_training`

## 사용 예시

### sim raw -> canonical training

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/FrankaStack_20260308_083022/raw_dataset \
  --source-type sim_raw \
  --schema canonical_training \
  --output-dir outputs/exported_datasets
```

### sim raw -> adc compatible

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/FrankaStack_20260308_083022/raw_dataset \
  --source-type sim_raw \
  --schema adc_compatible \
  --output-dir outputs/exported_datasets
```

### adc raw -> canonical training

```bash
python3 scripts/export_dataset.py \
  <adc_dataset_root> \
  --source-type adc_raw \
  --schema canonical_training \
  --output-dir outputs/exported_datasets \
  --robot so101 \
  --adc-robot-config external/AutoDataCollector/robot_configs/robot/so101_robot3.yaml
```

## 현재 검증된 것

현재 이 레포에서는 sim raw export까지 검증했다.

- `outputs/exported_datasets/FrankaStack_20260308_083022_raw_dataset_canonical_training`
- `outputs/exported_datasets/OpenArmStack_20260308_083819_raw_dataset_canonical_training`
- `outputs/exported_datasets/SO101Stack_20260308_083356_raw_dataset_canonical_training`
- `outputs/exported_datasets/FrankaStack_20260308_083022_raw_dataset_adc_compatible`

검증된 내용:

- `state/action/goal_joint/gripper`가 `[-100, 100]`
- canonical export에서 current/goal TCP 배열 생성
- manifest에 normalization spec / frame definition 기록

## 학습 시 권장 규칙

- raw dataset을 직접 학습에 넣지 않는다.
- export 후 dataset만 학습에 사용한다.
- multi-embodiment 공통 action/state tensor로 바로 합치지 않는다.
- **per-robot export, per-robot training**을 기본 원칙으로 둔다.
- `adc_compatible`는 학습 기본 경로가 아니라 parity/debug 용도로만 쓴다.

## Real ADC 검증 재개 조건

이 머신에서는 아직 real ADC 실제 검증을 하지 않았다. 이유는 두 가지다.

- 실제 ADC LeRobot dataset root가 없다.
- host Python에 아래 패키지가 없다.
  - `pyarrow`
  - `pandas`
  - `datasets`
  - `lerobot`

재개 조건:

1. 실제 ADC dataset 경로 확보
2. 위 패키지 설치
3. 아래 명령으로 export 1회 실행

```bash
python3 scripts/export_dataset.py \
  <adc_dataset_root> \
  --source-type adc_raw \
  --schema canonical_training \
  --output-dir outputs/exported_datasets \
  --robot so101 \
  --adc-robot-config external/AutoDataCollector/robot_configs/robot/so101_robot3.yaml
```

재개 후 확인할 것:

- `manifest.json` 생성
- `observation.state`, `action`, `skill.goal_position.joint`, `skill.goal_position.gripper`가 `[-100, 100]`
- `observation.tcp.*`, `skill.goal_position.tcp.*` 생성
- ADC 원본의 `videos/`, `data/`, `meta/` 구조를 잃지 않았는지 확인
