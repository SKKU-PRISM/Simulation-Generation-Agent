# Task YAML 명세

이 문서는 `tasks/{robot}/{category}/*.yaml` 형식의 task 문서를 설명합니다. 이 YAML은 **IsaacLab 생성의 입력이자**, 필요할 때 **Isaac Sim 시각 검증**과 **Data Collection**의 공통 입력으로도 사용됩니다.

## 기본 구조

```yaml
task:
simulation:
scene:
assets:
goal:
camera:
notes:
```

핵심은 `assets`와 `goal`입니다. IsaacLab Pipeline이 가장 먼저 이 둘을 해석합니다.

## 1. `task`

```yaml
task:
  name: FrankaStack
  description: Stack three cubes
  version: "1.0.0"
```

권장:

- `name`: PascalCase
- 파일명: lowercase + underscore

## 2. `simulation`

```yaml
simulation:
  gravity: [0, 0, -9.81]
  timestep: 0.01
  decimation: 5
  episode_length: 30.0
  render_interval: 2
```

IsaacLab evaluator는 여기서 특히 다음 값을 비교합니다.

- `timestep`
- `decimation`
- `episode_length`

## 3. `scene`

```yaml
scene:
  lighting:
    type: dome
    intensity: 3000
    color: [0.75, 0.75, 0.75]
  ground:
    enabled: true
    position: [0, 0, -1.05]
```

평가기에서 `ground`, `light`, `env_spacing` 유무를 씬 구조 점수에 반영합니다.

## 4. `assets`

지원 타입:

- `articulation`: 로봇
- `static`: 고정 자산
- `rigid`: 조작 대상 자산

공통 예시:

```yaml
- name: cube_1
  type: rigid
  source: usd
  asset_path: "{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd"
  position: [0.4, 0.0, 0.0203]
  rotation: [1, 0, 0, 0]
  scale: [1, 1, 1]
  prim_path: /World/Cube_1
```

### `articulation`

```yaml
- name: robot
  type: articulation
  robot_type: franka
  asset_path: "{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
  position: [0, 0, 0]
  initial_joints:
    panda_joint1: 0.0
```

로봇별 실행 설정은 YAML 외에 `configs/robot_profiles/*.yaml`도 함께 사용됩니다.  
Data Collection은 이 robot profile을 읽어 joint 이름, gripper 타입, IK 관련 설정을 가져옵니다.

### `static`

```yaml
- name: table
  type: static
  source: usd
  asset_path: "{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
  position: [0.5, 0, 0]
```

### `rigid`

USD 자산:

```yaml
- name: can
  type: rigid
  source: usd
  asset_path: "{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd"
  position: [0.45, 0.0, 0.05]
  physics:
    rigid_body: true
    collision: true
```

primitive 자산:

```yaml
- name: target_pad
  type: rigid
  source: primitive
  primitive: cube
  position: [0.5, 0, 0.001]
  scale: [0.08, 0.08, 0.002]
  color: [1.0, 0.0, 0.0]
```

### Assembly-specific fields

Assembly 계열 task는 아래 필드를 추가로 사용할 수 있습니다.

- `constraints`
  - 현재는 `type: fixed_joint`만 사용합니다.
  - 다물체 조립체(peg 2조각, charger base + prongs 등)를 하나의 물리 단위처럼 다루기 위한 선언입니다.
- `physics.collision_mesh: triangle`
  - cutout / slot hole가 있는 mesh에서 convex proxy 대신 tri-mesh collision이 필요함을 의미합니다.
- `goal.conditions[*].target_position`, `target_rotation`
  - placeholder goal(`matching_cutout` 등)를 concrete target pose로 resolve한 결과입니다.
- `shape_to_place_config`
  - assembling kits에서 active movable shape에 적용할 shared spawn/randomization 설정입니다.

## 5. `goal`

두 가지 포맷을 지원합니다.

### `success_criteria` 포맷

주로 stack 계열에서 사용합니다.

```yaml
goal:
  description: Stack cubes
  success_criteria:
    xy_threshold: 0.04
    height_diff: 0.0468
    height_threshold: 0.005
    gripper_must_be_open: true
```

### `conditions` 포맷

주로 pick/place, sort, cabinet 등에서 사용합니다.

```yaml
goal:
  description: Place can on tray
  conditions:
    - type: position_match
      subject: can
      target: tray
      threshold: 0.05
```

## 6. 파이프라인이 goal을 해석하는 방식

IsaacLab evaluator는 두 포맷을 내부적으로 정규화합니다.

- `success_criteria` -> stacked / xy_aligned / gripper open 계열 조건으로 변환
- `conditions` -> 그대로 condition list로 사용

즉 문서 작성자는 포맷만 맞추면 되고, 평가는 동일한 condition 기반으로 진행됩니다.

## 7. 에셋 경로 규약

### 템플릿 변수

- `{ISAAC_NUCLEUS_DIR}` -> Nucleus `/Isaac`
- `{ISAACLAB_NUCLEUS_DIR}` -> `/Isaac/IsaacLab`
- `{ADC_URDF_DIR}` -> Data Collection에서 ADC 서브모듈의 URDF 루트로 해석
- `{ISAACLAB_URDF_DIR}`, `{ISAAC_SIM_URDF_DIR}` -> robot profile에서 사용할 수 있는 URDF 템플릿

### 로컬 자산

`assets/...`로 시작하는 경로는 **레포 루트 기준 로컬 자산**으로 해석합니다.

예:

```yaml
asset_path: "assets/robots/so101/so101.usd"
```

이 규칙은 특히 SO-101 로컬 USD 자산에서 중요합니다.

`asset_url`도 과거 문서에는 존재했지만, 현재는 portable하지 않으므로 새 task에서는 사용하지 않습니다.
특히 assembly mesh asset은 `assets/assembling_kits/*.usd` 형태의 `asset_path`를 사용합니다.

## 8. 카메라

```yaml
camera:
  position: [2.0, 2.0, 1.5]
  target: [0, 0, 0.3]
  fov: 60
  resolution: [1280, 720]
```

Isaac Sim에서는 시각 검증에 사용되고, Data Collection은 별도로 robot profile의 multi-camera 설정을 사용할 수 있습니다.

## 9. 지원 로봇과 태스크

현재 task 수는 78개입니다.

| 카테고리 | Franka | OpenArm | UR10e | SO-101 | UR10(legacy) | 합계 |
|----------|:------:|:-------:|:-----:|:------:|:------------:|:----:|
| assembly | 4 | 4 | - | 4 | 4 | 16 |
| stack | 2 | 2 | 2 | 1 | - | 7 |
| lift | 2 | 2 | - | 1 | - | 5 |
| pick_place | 9 | 9 | 7 | 3 | - | 28 |
| reach | - | 1 | 1 | 1 | - | 3 |
| cabinet | 3 | 3 | 3 | - | - | 9 |
| sort | 3 | 3 | - | 3 | - | 9 |
| peg_insert | 1 | - | - | - | - | 1 |
| **합계** | **24** | **24** | **13** | **13** | **4** | **78** |

비고:

- `UR10(legacy)` assembly task는 corpus에는 남아 있지만, IsaacLab 실행 시에는 내부적으로 canonical `ur10e` 표현으로 resolve될 수 있습니다.
- `assembly` category는 기존 stack/lift보다 geometry/pose semantics가 더 강하므로, goal relation과 local asset path를 명시적으로 유지하는 것이 중요합니다.

## 10. 작성 시 주의점

- asset 이름은 evaluator와 scene parser가 그대로 참고하므로 일관되게 유지
- 숫자 threshold는 evaluator가 코드 내 literal로 찾으므로 가능한 한 명시적으로 적기
- SO-101처럼 로컬 자산을 쓰는 경우 `assets/...` 상대경로를 사용
- assembly task는 `scale`, `color`, `primitive`, `constraints`, `collision_mesh`를 생략하지 말 것
- Data Collection을 염두에 둔다면 로봇 타입과 scene object 이름을 모호하지 않게 작성

## 템플릿

기본 템플릿:

```text
tasks/templates/task_document.yaml.template
```

## 관련 문서

- `docs/usage.md`
- `docs/evaluation.md`
- `configs/robot_profiles/`
