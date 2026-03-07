# IsaacLab 환경 설정 가이드

데이터 수집 파이프라인을 위한 IsaacLab `ManagerBasedRLEnv` 환경 설정 참조 문서.

---

## 1. 환경 라이프사이클

### 설정 클래스 정의

IsaacLab 환경은 `@configclass` 데코레이터가 적용된 설정 클래스들의 조합으로 정의된다.

```python
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    """환경 설정의 최상위 클래스."""
    pass
```

`ManagerBasedRLEnvCfg`는 다음 하위 설정을 포함한다:

| 하위 설정 | 클래스 | 역할 |
|-----------|--------|------|
| `scene` | `InteractiveSceneCfg` | 로봇, 오브젝트, 지형 등 씬 구성 |
| `actions` | `ActionsCfg` | 액션 공간 정의 (관절 위치/속도/토크) |
| `observations` | `ObservationsCfg` | 관측 공간 정의 |
| `events` | `EventCfg` | 리셋, 랜덤화 이벤트 |
| `terminations` | `TerminationsCfg` | 에피소드 종료 조건 |
| `rewards` | `RewardsCfg` | (선택) 보상 함수 |

### `__post_init__`에서의 초기화

`@configclass`는 Python `dataclass`를 확장한 것으로, `__post_init__`에서 하위 설정 간의 의존성을 처리한다.

```python
@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    # 하위 설정을 클래스 변수로 선언
    scene: MySceneCfg = MySceneCfg(env_spacing=2.5, num_envs=1)
    actions: MyActionsCfg = MyActionsCfg()
    observations: MyObservationsCfg = MyObservationsCfg()
    events: MyEventsCfg = MyEventsCfg()
    terminations: MyTerminationsCfg = MyTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # scene의 num_envs를 상위 설정으로 동기화
        self.scene.num_envs = self.num_envs
        # decimation과 sim dt 설정
        self.decimation = 5
        self.sim.dt = 0.01  # 100Hz physics, 20Hz control
        self.episode_length_s = 600.0  # 데이터 수집 시 긴 에피소드
```

### 환경 구성 순서

환경 인스턴스 생성 시 다음 순서로 초기화가 진행된다:

1. **`ManagerBasedRLEnv(cfg=env_cfg)` 호출** -- 설정 객체를 전달하여 환경 생성을 시작한다.
2. **`InteractiveScene` 구성** -- `scene` 설정에 따라 USD stage에 에셋(로봇, 오브젝트)을 로드한다. `ArticulationCfg`에 포함된 `ActuatorNetCfg`(stiffness, damping, effort_limit 등)가 이 단계에서 해석된다.
3. **PhysX 초기화** -- `SimulationContext`가 PhysX solver를 초기화한다. 설정된 actuator 파라미터가 PhysX joint drive에 기록된다:
   - `stiffness` -> PhysX joint의 `drive.stiffness` (위치 게인)
   - `damping` -> PhysX joint의 `drive.damping` (속도 게인)
   - `effort_limit_sim` -> PhysX joint의 `maxForce` (토크 상한)
4. **Actuator 모델 해석** -- `ImplicitActuator`, `DCMotor` 등 actuator 클래스가 인스턴스화된다. 각 actuator는 PhysX joint drive 파라미터를 직접 설정하거나 내부 토크 연산을 수행한다.
5. **Manager 초기화** -- Action, Observation, Event, Termination, Reward 매니저가 각각의 설정에 따라 초기화된다.
6. **`env.reset()` 호출 가능** -- 모든 매니저와 PhysX가 준비되면 에피소드 루프를 시작할 수 있다.

> **핵심**: `env_cfg`에 지정한 값들은 환경 구성 시점에 PhysX로 전파된다. 환경 생성 이후에는 `write_joint_stiffness_to_sim()` 등의 API로 런타임 패치가 가능하지만, 가급적 설정 단계에서 올바른 값을 지정해야 한다.

---

## 2. ImplicitActuator PD 모델

`ImplicitActuator`는 IsaacLab의 기본 actuator로, PhysX의 내장 PD 제어를 그대로 활용한다. 별도의 토크 연산 없이 PhysX joint drive 파라미터만 설정한다.

### 파라미터 정의

| 파라미터 | 단위 | 설명 |
|----------|------|------|
| `stiffness` | Nm/rad | PhysX joint drive의 위치 게인. 높을수록 목표 위치에 대한 추적력이 강해진다. |
| `damping` | Nm*s/rad | PhysX joint drive의 속도 게인. 높을수록 진동이 빠르게 감쇠된다. |
| `effort_limit_sim` | Nm | PhysX joint의 `maxForce`. 이 값을 초과하는 토크는 PhysX 레벨에서 클리핑된다. |
| `effort_limit` | Nm | Actuator 모델 내부의 토크 클리핑 한계. `ImplicitActuator`에서는 `effort_limit_sim`과 동기화된다. |

### 토크 연산 흐름

PhysX의 PD joint drive가 매 물리 스텝마다 다음을 계산한다:

```
computed_torque = stiffness * (target_pos - current_pos) + damping * (0 - current_vel)
                           ↓
              clip(computed_torque, -effort_limit)   ← actuator 내부 클리핑
                           ↓
              clip(result, -effort_limit_sim)         ← PhysX 레벨 클리핑
                           ↓
                      PhysX solver에 전달
```

`ImplicitActuator`에서는 `effort_limit`과 `effort_limit_sim`이 동기화되므로 실질적으로 한 번의 클리핑만 발생한다.

### 수치 안정성 규칙

**높은 stiffness + 낮은 effort_limit_sim = solver 포화 = 수치 불안정**

- 위치 오차가 아주 작아도 `stiffness * error`가 `effort_limit_sim`에 도달하면, PD 제어기가 포화 상태에 빠져 정밀한 위치 추적이 불가능해진다. 이는 진동, 발산, 관절 잠김 등을 유발한다.
- **ImplicitActuator에서는 `effort_limit_sim`을 매우 크게(1e9) 설정하거나 `None`으로 두어 PhysX가 자체적으로 안정성을 관리하도록 하는 것이 안전하다.**
- 실제 물리적 토크 한계는 `stiffness * max_possible_error`로 자연스럽게 제한된다. 인위적으로 낮은 `effort_limit_sim`을 지정할 필요가 없다.

---

## 3. Action Space

### JointPositionActionCfg

관절 위치를 직접 제어하는 액션 설정이다.

```python
from isaaclab.envs.mdp.actions import JointPositionActionCfg

@configclass
class MyActionsCfg:
    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_.*", "elbow_joint", "wrist_.*"],
        scale=1.0,
        use_default_offset=False,
    )
```

| 파라미터 | 설명 |
|----------|------|
| `scale` | 액션 값에 곱해지는 스케일 팩터. RL 학습 시 `0.5` (정규화된 액션), 절대 관절값 제어 시 `1.0`. |
| `use_default_offset` | `True`이면 `action * scale + default_pose`가 target이 된다. 절대 관절값을 직접 전달할 때는 반드시 `False`. |
| `joint_names` | 정규식 패턴 리스트. articulation의 kinematic chain 순서대로 매칭된다. |

**joint_names 매칭 예시:**

```python
# UR10e: 6개 팔 관절을 kinematic chain 순서로 매칭
joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

# regex 패턴도 가능 (순서는 kinematic chain에 의해 결정됨)
joint_names=["shoulder_.*", "elbow_joint", "wrist_.*"]
```

### BinaryJointPositionActionCfg

Gripper처럼 열림/닫힘 두 상태만 필요한 관절에 사용한다. 여러 gripper 관절을 1차원 binary 신호로 제어할 수 있다.

```python
from isaaclab.envs.mdp.actions import BinaryJointPositionActionCfg

gripper_action = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_finger_joint.*"],
    open_command_expr={"panda_finger_joint.*": 0.04},
    close_command_expr={"panda_finger_joint.*": 0.0},
)
```

| 파라미터 | 설명 |
|----------|------|
| `open_command_expr` | 관절별 open 위치를 정의하는 dict. 정규식 키 지원. |
| `close_command_expr` | 관절별 close 위치를 정의하는 dict. |
| **sign convention** | `action > 0` -> open 명령, `action < 0` -> close 명령. |

---

## 4. 데이터 수집용 vs RL 학습용 설정 차이

데이터 수집 파이프라인은 IK solver가 계산한 절대 관절값을 직접 전달하므로, RL 학습 환경과는 설정이 다르다.

| 항목 | RL 학습용 | 데이터 수집용 |
|------|----------|-------------|
| `action scale` | `0.5` (정규화된 액션) | `1.0` (절대 라디안 값) |
| `use_default_offset` | `True` (default pose 기준 상대값) | `False` (IK 출력을 그대로 사용) |
| `episode_length_s` | `~30s` (짧은 에피소드) | `600s` (긴 에피소드, 스킬 실행 여유) |
| terminations | 전부 활성 (학습 효율) | `time_out`만 유지 (조기 종료 방지) |
| `stiffness` | RL 최적화에 맞춘 값 | 높은 PD gains (정밀한 위치 추적) |
| `disable_gravity` | 로봇 특성에 따라 결정 | `True` (floating base 허용, 안정적 조작) |

### 데이터 수집 환경 설정 핵심 원칙

1. **절대 제어**: IK solver가 출력한 관절값을 그대로 액션으로 전달한다. `scale=1.0`, `use_default_offset=False`가 필수다.
2. **긴 에피소드**: 스킬 시퀀스(pick -> move -> place 등)를 여유 있게 실행할 수 있도록 `episode_length_s`를 충분히 크게 설정한다.
3. **조기 종료 방지**: 학습용 termination(높이 제한, 접촉 감지 등)은 비활성화한다. 데이터 수집 중 의도치 않은 에피소드 리셋은 녹화 데이터를 손상시킨다.
4. **높은 PD gains**: 위치 추적 정밀도가 데이터 품질에 직결된다. stiffness와 damping을 충분히 높게 설정하되, 수치 안정성 범위 내에서 조정한다.

---

## 5. UR10e + Robotiq 2F-85 상세

### Actuator 그룹

`UR10e_ROBOTIQ_2F_85_CFG` 기준 actuator 그룹 구성:

| 그룹 | 관절 | stiffness | damping | effort_limit_sim |
|------|------|-----------|---------|------------------|
| shoulder | `shoulder_pan_joint`, `shoulder_lift_joint` | 1320 | 72.7 | None (USD 기본: 330) |
| elbow | `elbow_joint` | 600 | 34.6 | None (USD 기본: 150) |
| wrist | `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint` | 216 | 29.4 | None (USD 기본: 28) |
| gripper_drive | `finger_joint` | 11.25 | 0.1 | 10.0 |
| gripper_finger | `*_inner_finger_joint` | 0.2 | 0.001 | 1.0 |
| gripper_passive | `*_inner_finger_knuckle`, `right_outer_knuckle` | 0.0 | 0.0 | 1.0 |

> **참고**: `effort_limit_sim`이 `None`이면 USD 에셋에 정의된 기본 `maxForce`가 적용된다. ImplicitActuator에서는 이 값을 명시적으로 높게 설정하거나 `None`으로 두는 것이 안전하다.

### 관절 순서 (kinematic chain)

```
shoulder_pan_joint(0) -> shoulder_lift_joint(1) -> elbow_joint(2)
    -> wrist_1_joint(3) -> wrist_2_joint(4) -> wrist_3_joint(5)
```

액션 텐서의 인덱스 0~5가 이 순서에 대응된다. Gripper 관절은 팔 관절 뒤에 위치한다.

### Ready Pose

```python
ready_pose = [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0]
# shoulder_pan=0, shoulder_lift=-π/2, elbow=π/2,
# wrist_1=-π/2, wrist_2=-π/2, wrist_3=0
```

이 자세에서 end-effector가 아래를 향하며, 테이블 위 작업에 적합한 초기 구성이다.

### 안전한 PD Gains 범위

데이터 수집용으로 stiffness를 높일 때의 권장 범위:

| 그룹 | stiffness 범위 | damping 범위 | 비고 |
|------|---------------|-------------|------|
| shoulder | 1320 ~ 5000 | 73 ~ 400 | 관성이 크므로 높은 stiffness 허용 |
| elbow | 600 ~ 3000 | 35 ~ 200 | shoulder보다 보수적으로 설정 |
| wrist | 216 ~ 1100 | 29 ~ 150 | 관성이 작으므로 과도한 증가 금지 (원래 값 대비 약 5배가 상한) |

**wrist 관절 주의사항**: wrist는 관성 모멘트가 작아 stiffness를 과도하게 올리면 진동이 발생한다. 원래 값(216) 대비 약 5배(~1100)를 초과하지 않는 것이 안전하다.

---

## 6. 흔한 실수와 디버깅

### 1. 모든 actuator에 균일한 stiffness 적용

**증상**: wrist 관절이 진동하거나 발산함.

**원인**: shoulder(관성이 큼)와 wrist(관성이 작음)에 동일한 stiffness를 적용하면, wrist에서 과도한 토크가 발생하여 불안정해진다.

**해결**: 관절 그룹별로 차별화된 stiffness를 적용한다. 위의 UR10e 안전 범위를 참조.

### 2. effort_limit_sim을 stiffness와 유사한 값으로 설정

**증상**: 관절이 목표에 도달하지 못하고 떨림.

**원인**: `stiffness * small_error`가 이미 `effort_limit_sim`에 도달하여 토크가 클리핑된다. PD 제어기가 포화되어 미세 제어가 불가능하다.

**해결**: ImplicitActuator에서는 `effort_limit_sim`을 매우 크게 설정하거나 `None`으로 둔다. 물리적 토크 한계는 stiffness와 위치 오차에 의해 자연스럽게 결정된다.

### 3. scale=0.5인 상태에서 절대 관절값 전달

**증상**: 로봇이 목표 위치의 절반까지만 이동함.

**원인**: `scale=0.5`이면 IK solver가 출력한 관절값이 내부적으로 `value * 0.5`로 변환된다.

**해결**: 데이터 수집 시에는 반드시 `scale=1.0`으로 설정한다.

### 4. use_default_offset=True인 상태에서 IK 출력 전달

**증상**: 로봇이 예상 외의 자세로 이동함.

**원인**: IK solver의 출력(절대 관절값)에 `default_pose`가 더해져서 관절값이 이중 적용된다. 예: IK 출력 `-π/2` + default `-π/2` = `-π`로 잘못된 목표가 설정됨.

**해결**: 절대 관절값을 사용할 때는 반드시 `use_default_offset=False`로 설정한다.

### 5. AppLauncher 전에 physics 모듈 import

**증상**: `ImportError` 또는 `ModuleNotFoundError`.

**원인**: IsaacLab의 physics 관련 모듈(`torch`, `isaaclab.envs` 등)은 `AppLauncher` 초기화 이후에만 import 가능하다. 이는 IsaacLab의 필수 패턴이다.

**해결**: 반드시 다음 순서를 지킨다:

```python
# 1. AppLauncher 먼저
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# 2. 이후에 physics 관련 import
import torch
from isaaclab.envs import ManagerBasedRLEnv
```

### 6. env.step() 후 즉시 body_state 읽기

**증상**: 읽은 EE 위치가 이전 프레임의 값임.

**원인**: `env.step()`이 반환된 직후에는 PhysX가 아직 새로운 상태를 완전히 계산하지 않았을 수 있다. 특히 body state는 다음 `scene.update()` 호출 시 갱신된다.

**해결**: `env.step()` 반환 후 바로 `articulation.data.body_state_w`를 읽지 말고, 필요하면 추가 zero-action 스텝을 넣거나 observation을 통해 상태를 확인한다.

### 검증 패턴 (기본 컨트롤 테스트)

환경 설정이 올바른지 확인하기 위한 단계별 테스트:

```python
import math
import numpy as np

# 1. 환경 생성 후 초기 관절값 확인
obs = env.reset()
art = env.scene["robot"]
actual = art.data.joint_pos[0].cpu().numpy()
expected = [0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0]  # UR10e ready pose
assert np.allclose(actual[:6], expected[:6], atol=0.01), \
    f"초기 관절값 불일치: {actual[:6]} vs {expected[:6]}"

# 2. hold position 100 스텝 -> 드리프트 확인 (< 0.01 rad)
initial_pos = art.data.joint_pos[0, :6].cpu().numpy().copy()
for _ in range(100):
    action = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    action[0, :6] = torch.tensor(initial_pos, dtype=torch.float32)
    env.step(action)
drift = np.abs(art.data.joint_pos[0, :6].cpu().numpy() - initial_pos)
assert np.all(drift < 0.01), f"관절 드리프트 과다: {drift}"

# 3. 단일 관절 이동 -> 추적 오차 확인 (< 0.05 rad)
target = initial_pos.copy()
target[0] += 0.5  # shoulder_pan 0.5 rad 이동
for _ in range(200):
    action = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    action[0, :6] = torch.tensor(target, dtype=torch.float32)
    env.step(action)
tracking_error = np.abs(art.data.joint_pos[0, 0].cpu().item() - target[0])
assert tracking_error < 0.05, f"추적 오차 과다: {tracking_error:.4f} rad"

# 4. IK -> action -> EE 위치 오차 확인 (< 0.03 m)
# IK solver로 목표 EE 위치에 대한 관절값을 계산한 뒤,
# 해당 관절값을 action으로 전달하고 EE 위치를 측정한다.
# 오차가 3cm 이내여야 한다.
```

---

## 7. 참고 파일

| 파일 | 설명 |
|------|------|
| `isaaclab/actuators/actuator_pd.py` | `ImplicitActuator` 구현. `effort_limit`과 `effort_limit_sim` 동기화 로직 포함. |
| `isaaclab/actuators/actuator_base.py` | `effort_limit_sim` 해석, `_clip_effort()` 메서드 정의. |
| `isaaclab/assets/articulation/articulation.py` | `write_joint_stiffness_to_sim()`, `write_joint_effort_limit_to_sim()` 등 런타임 패치 API. |
| `isaaclab_assets/robots/universal_robots.py` | `UR10e_CFG`, `UR10e_ROBOTIQ_2F_85_CFG` actuator 그룹 정의. |
| `src/data_collection/pipeline.py` | 데이터 수집 파이프라인 오케스트레이터. 런타임 PD gains 패치 적용 코드 포함. |
| `configs/robot_profiles/ur10e.yaml` | UR10e + Robotiq 2F-85 로봇 프로필 (관절 이름, 한계, ready pose, 카메라 등). |
| `configs/robot_profiles/franka.yaml` | Franka Emika Panda 로봇 프로필 (비교 참조용). |
| `configs/data_collection_config.yaml` | 데이터 수집 파이프라인 설정 (dt, decimation, episode 수, VLM 판정 등). |
| `src/data_collection/sim_robot_interface.py` | `SimRobotInterface` -- IsaacLab Articulation 래퍼. 관절 읽기/쓰기, gripper 제어, action 텐서 조립. |
| `src/data_collection/config.py` | `RobotSimConfig`, `DataCollectionConfig` -- 로봇 프로필 로더, 파이프라인 설정 클래스. |
