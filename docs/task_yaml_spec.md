# Task YAML 명세

이 문서는 Task YAML 문서의 포맷, 필드 정의, 에셋 규약을 설명합니다.

템플릿 파일: `tasks/templates/task_document.yaml.template` (v2.0.0)

---

## 기본 구조

```yaml
task:           # 태스크 메타데이터
simulation:     # 시뮬레이션 파라미터
scene:          # 조명, 지면 설정
assets:         # 로봇, 테이블, 오브젝트 목록
goal:           # 목표 상태 + 성공 기준
camera:         # 카메라 위치/방향
notes:          # (선택) 참고사항
```

---

## 섹션별 필드 정의

### `task` — 태스크 메타데이터

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| `name` | string | O | 태스크 이름 (CamelCase, 예: `FrankaStack`) |
| `description` | string | O | 로봇이 수행해야 할 작업 설명 |
| `version` | string | O | 문서 버전 (SemVer, 예: `"1.0.0"`) |

### `simulation` — 시뮬레이션 파라미터

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|:----:|--------|------|
| `gravity` | [x,y,z] | O | `[0,0,-9.81]` | 중력 벡터 (m/s^2, Z-up) |
| `timestep` | float | O | `0.01` | 시뮬레이션 타임스텝 (100Hz) |
| `decimation` | int | | `5` | 제어 주기 = timestep * decimation |
| `episode_length` | float | | `30.0` | 에피소드 길이 (초) |
| `render_interval` | int | | `2` | 렌더링 간격 |
| `physx` | object | | | PhysX 엔진 파라미터 |

### `scene` — 씬 설정

```yaml
scene:
  lighting:
    type: dome           # dome, distant, sphere
    intensity: 3000      # 조명 강도
    color: [0.75, 0.75, 0.75]  # RGB (0-1)

  ground:
    enabled: true
    position: [0, 0, -1.05]   # IsaacLab 규약: z=-1.05
    asset_path: "{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd"  # (선택)
```

### `assets` — 에셋 목록

모든 에셋은 다음 공통 필드를 가집니다:

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| `name` | string | O | 고유 이름 (예: `robot`, `cube_1`) |
| `type` | string | O | `articulation`, `static`, `rigid` 중 택일 |
| `source` | string | | `usd`, `primitive`, `asset_db` |
| `position` | [x,y,z] | O | 월드 좌표 위치 (m) |
| `rotation` | [w,x,y,z] | | 쿼터니언 (wxyz 포맷, 기본: `[1,0,0,0]`) |
| `scale` | [x,y,z] | | 스케일 (기본: `[1,1,1]`) |
| `prim_path` | string | | USD prim 경로 (예: `/World/Robot`) |

#### `type: articulation` (로봇)

```yaml
- name: robot
  type: articulation
  source: usd
  robot_type: franka              # franka, openarm, ur10, so101
  asset_path: "{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
  position: [0, 0, 0]
  rotation: [1, 0, 0, 0]
  initial_joints:                 # 초기 관절 각도 (rad)
    panda_joint1: 0.0
    panda_joint2: -0.7854
    # ...
  actuators:                      # 액추에이터 설정
    shoulder:
      joint_names: ["panda_joint1", "panda_joint2", ...]
      effort_limit: 87.0          # N*m
      stiffness: 80.0
      damping: 4.0
  ee_frame:                       # 엔드이펙터 프레임
    body: panda_hand
    offset_position: [0, 0, 0.1034]
  physics:
    self_collision: true
  randomize:
    joint_position:
      distribution: gaussian
      std: 0.02
```

#### `type: static` (고정 물체 — 테이블 등)

```yaml
- name: table
  type: static
  source: usd
  asset_path: "{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
  position: [0.5, 0, 0]
  rotation: [0.707, 0, 0, 0.707]
  physics:
    collision: true
```

#### `type: rigid` (조작 대상 물체)

USD 에셋 사용:
```yaml
- name: cube_1
  type: rigid
  source: usd
  asset_path: "{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd"
  position: [0.4, 0, 0.0203]
  physics:
    rigid_body: true
    collision: true
    solver_position_iterations: 16
  randomize:
    position:
      type: absolute              # absolute 또는 relative
      x: [0.4, 0.6]
      y: [-0.1, 0.1]
      z: 0.0203                   # 고정 (테이블 높이 + 반높이)
    orientation:
      yaw: [-1.0, 1.0]           # rad
    min_separation: 0.1           # m (다른 오브젝트와 최소 거리)
```

프리미티브 사용 (asset_path 불필요):
```yaml
- name: target_pad
  type: rigid
  source: primitive
  primitive: cube
  position: [0.5, 0, 0.001]
  scale: [0.08, 0.08, 0.002]
  color: [1.0, 0.0, 0.0]
  physics:
    rigid_body: false
    collision: true
```

### `goal` — 목표 정의

두 가지 포맷이 있습니다:

**포맷 1: `success_criteria` (stack/lift 등)**

```yaml
goal:
  description: "Stack cubes: Blue -> Red -> Green"
  success_criteria:
    xy_threshold: 0.04          # m, XY 정렬 허용 오차
    height_diff: 0.0468         # m, 큐브 간 높이 차이
    height_threshold: 0.005     # m, 높이 차이 허용 오차
    gripper_must_be_open: true
```

**포맷 2: `conditions` 리스트 (pick_place/sort/cabinet 등)**

```yaml
goal:
  description: "Pick up the can and place it on the tray"
  conditions:
    - type: position_match
      subject: can
      target: tray
      threshold: 0.05           # m
    - type: height_above
      subject: can
      min_height: 0.1           # m
```

### `camera` — 카메라 설정

| 필드 | 타입 | 필수 | 설명 |
|------|------|:----:|------|
| `position` | [x,y,z] | O | 카메라 위치 (m) |
| `target` | [x,y,z] | O | 카메라가 바라보는 지점 |
| `fov` | float | | 시야각 (도, 기본: 60) |
| `resolution` | [w,h] | | 해상도 (기본: [1280, 720]) |

---

## 에셋 경로 규약

### 템플릿 변수

| 변수 | 치환값 | 용도 |
|------|--------|------|
| `{ISAAC_NUCLEUS_DIR}` | `/Isaac` | Props, 환경, 일부 로봇 |
| `{ISAACLAB_NUCLEUS_DIR}` | `/IsaacLab` | IsaacLab 전용 로봇 에셋 |

이 변수들은 런타임에 Nucleus 서버 URL과 결합됩니다.

### YCB 오브젝트

| 경로 | 물리 속성 | 비고 |
|------|----------|------|
| `{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/*.usd` | 내장 (rigid_body + collision) | 바로 사용 가능 |
| `{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/*.usd` | 없음 (geometry만) | `physics: { rigid_body: true, collision: true }` 필수 |

### 스케일 주의사항

| 에셋 | 기본 단위 | 필요한 scale |
|------|----------|-------------|
| YCB 오브젝트 | 미터 | `[1, 1, 1]` (그대로) |
| `Props/Mugs/SM_Mug_*.usd` | 센티미터 | `[0.01, 0.01, 0.01]` 필수 |
| 대부분의 Nucleus Props | 미터 | `[1, 1, 1]` |

### USD Prim 경로 규칙

- 숫자로 시작할 수 없음: `/World/006_mustard_bottle` (X) → `/World/YCB_006_mustard_bottle` (O)
- Y-up 모델을 Z-up 월드에서 세우려면: `rotation: [0.7071, 0.7071, 0, 0]` (X축 90도 회전)

---

## 지원 로봇 및 태스크

### 로봇 목록

| 로봇 | DOF | Gripper | Reach | 에셋 소스 |
|------|-----|---------|-------|----------|
| **Franka Panda** | 7+2 | Parallel jaw (8cm) | ~0.85m | Nucleus (`{ISAACLAB_NUCLEUS_DIR}`) |
| **OpenArm** | 7+2 | Parallel jaw (8.8cm) | ~0.85m | Nucleus (`{ISAAC_NUCLEUS_DIR}`) |
| **UR10** | 6 | Suction | ~1.3m | Nucleus (`{ISAAC_NUCLEUS_DIR}`) |
| **SO-101** | 5+1 | Claw (5cm) | ~0.3m | 로컬 (`assets/robots/so101/`) |

### 태스크 카테고리 (62개)

| 카테고리 | Franka | OpenArm | UR10 | SO-101 | 합계 |
|----------|:------:|:-------:|:----:|:------:|:----:|
| stack | 2 | 2 | 2 | 1 | 7 |
| lift | 2 | 2 | - | 1 | 5 |
| pick_place | 9 | 9 | 7 | 3 | 28 |
| reach | - | 1 | 1 | 1 | 3 |
| cabinet | 3 | 3 | 3 | - | 9 |
| sort | 3 | 3 | - | 3 | 9 |
| peg_insert | 1 | - | - | - | 1 |
| **합계** | **20** | **20** | **13** | **9** | **62** |

태스크 파일 경로: `tasks/{robot}/{category}/{robot}_{task}.yaml`

---

## Randomization 규칙

### Position Randomization

| 필드 | 설명 |
|------|------|
| `type: absolute` | 지정된 범위 내에서 절대 좌표 샘플링 |
| `type: relative` | 기본 위치 기준 오프셋 범위 |
| `min_separation` | 같은 타입 오브젝트 간 최소 거리 (m) |

### Orientation Randomization

```yaml
orientation:
  yaw: [-1.0, 1.0]    # rad, Z축 회전 범위
```

### Joint Position Randomization

```yaml
randomize:
  joint_position:
    distribution: gaussian
    mean: 0.0
    std: 0.02          # rad, 초기 관절 각도에 가우시안 노이즈
```

---

## 관련 문서

- [사용법](usage.md) — 태스크 실행 방법
- [평가 시스템](evaluation.md) — 코드 품질 평가
- 태스크 생성 스킬: `/design-task` (Claude Code에서 사용)
- 태스크 검증 스킬: `/validate-task` (19개 항목 검증)
