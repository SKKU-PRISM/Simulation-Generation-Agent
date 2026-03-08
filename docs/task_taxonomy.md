# Task Taxonomy

이 문서는 `.reference/task_image_gt/image (3).png`에 있는 분류 체계를 **최우선 기준**으로 삼아, 현재 `tasks/` 아래의 **78개 task spec**을 다시 분류한 결과를 정리한다.

기본 원칙은 다음과 같다.

- 1차 분류는 반드시 이미지의 대분류를 따른다:
  - `파지 (Prehensile)`
  - `배치 (Placement)`
  - `비파지 (Non-prehensile)`
  - `관절체 (Articulated)`
  - `삽입/조립 (Insertion/Assembly)`
  - `변형체 (Deformable)`
  - `회전 (Rotational)`
  - `접촉 (Contact-rich)`
- 기존 폴더명(`pick_place`, `stack`, `assembly` 등)은 **source metadata**로만 보고, 1차 분류 기준으로 쓰지 않는다.
- task가 여러 primitive를 포함하더라도, **최종 goal과 task의 핵심 성격**을 기준으로 1차 분류를 정한다.
- 표 안의 괄호 표기(`Place (Pick)` 등)는 **전제/보조 동작**을 뜻한다. 즉 primary는 `배치`지만, 실행 전제에 `파지`가 들어간다는 의미다.
- `reach`는 물체를 집지 않고 엔드이펙터만 목표 pose로 이동시키므로 `비파지 (Non-prehensile)`로 둔다.
- pure `cabinet` drawer-opening task는 손잡이를 잡더라도 핵심 성격이 articulated manipulation이므로 `관절체`로만 둔다.
- `peg-in-hole`, `plug charger` 같은 정밀 정렬 기반 삽입 task는 현재 문서에서는 `삽입/조립`으로 둔다.
- `peg insertion` 계열(`peg_insertion_side`, `peg_insert`)은 grasp가 전제되더라도 summary 상 `파지` count에서는 제외하고 `삽입/조립`으로만 센다.
- `접촉 (Contact-rich)`은 현재 repo에서는 별도 primary class가 아니라, 필요 시 삽입류 task의 성격 설명에만 사용한다.

## Summary

- 전체 task 수: **78**
- 아래 집계는 **중복 허용 count**다.
- `파지/비파지`는 grasp 필요 여부 기준이고, `배치/관절체/삽입·조립/회전`은 task 성격 기준이라 서로 합쳐서 78이 되지 않는다.
- `파지 = 전체 78 - reach 3 - pure cabinet 3 - peg insertion 5 = 67`
- `배치 = pick_place 28 + sort 9 + stack 7 + cabinet_blocks/ycb 6 = 50`

### UR 제외 집계

`ur10e` embodiment를 제외하고 `franka`, `openarm`, `so101`만 남겨서 다시 집계한 값이다.

- 대상 task 수: **61**
- 포함 robot: `franka` 24, `openarm` 24, `so101` 13

| 분류 | UR 제외 task 수 |
| --- | ---: |
| `파지 (Prehensile)` | 53 |
| `배치 (Placement)` | 39 |
| `비파지 (Non-prehensile)` | 2 |
| `관절체 (Articulated)` | 2 |
| `삽입/조립 (Insertion/Assembly)` | 10 |
| `변형체 (Deformable)` | 0 |
| `회전 (Rotational)` | 3 |
| `접촉 (Contact-rich)` | 0 |

| 분류 | 현재 task 수 | 포함 task family / 패턴 | 대표 예시 |
| --- | ---: | --- | --- |
| `파지 (Prehensile)` | 67 | `reach`, pure `cabinet`, `peg insertion` 계열을 제외한 grasp task | `FrankaLift`, `FrankaPickPlaceBox`, `OpenArmPlugCharger` |
| `배치 (Placement)` | 50 | `pick_place`, `sort`, `stack`, `cabinet_blocks`, `cabinet_ycb` | `FrankaPickPlaceBox`, `SO101LineArrange`, `UR10eStack` |
| `비파지 (Non-prehensile)` | 3 | `reach` | `OpenArmReach`, `SO101Reach`, `UR10eReach` |
| `관절체 (Articulated)` | 3 | `cabinet`에서 drawer opening만 해당 | `FrankaCabinet`, `UR10eCabinet` |
| `삽입/조립 (Insertion/Assembly)` | 13 | `assembling_kits`, `peg_insertion_side`, `plug_charger`, `peg_insert` | `FrankaAssemblingKits`, `FrankaFactoryPegInsert`, `OpenArmPlugCharger` |
| `변형체 (Deformable)` | 0 | 현재 없음 | - |
| `회전 (Rotational)` | 4 | `lift_peg_upright` | `FrankaLiftPegUpright`, `SO101LiftPegUpright` |
| `접촉 (Contact-rich)` | 0 | 현재 primary 없음. 일부 insertion task의 보조 성격으로만 사용 | `FrankaFactoryPegInsert` 성격 설명용 |

## 분류별 포함 Task

### 파지 (Prehensile) - 67

`reach` 3개, pure `cabinet` 3개, 그리고 `peg insertion` 5개를 제외한 grasp task가 여기에 들어간다.

- `assembly (12)`: `franka_assembling_kits`, `franka_lift_peg_upright`, `franka_plug_charger`, `openarm_assembling_kits`, `openarm_lift_peg_upright`, `openarm_plug_charger`, `so101_assembling_kits`, `so101_lift_peg_upright`, `so101_plug_charger`, `ur10e_assembling_kits`, `ur10e_lift_peg_upright`, `ur10e_plug_charger`
- `cabinet_blocks/ycb (6)`: `franka_cabinet_blocks`, `franka_cabinet_ycb`, `openarm_cabinet_blocks`, `openarm_cabinet_ycb`, `ur10e_cabinet_blocks`, `ur10e_cabinet_ycb`
- `lift (5)`: `franka_lift`, `franka_lift_sugar_box`, `openarm_lift`, `openarm_lift_sugar_box`, `so101_lift`
- `pick_place (28)`: `franka_multi_pick_place`, `franka_pick_place`, `franka_pick_place_bottle`, `franka_pick_place_box`, `franka_pick_place_can`, `franka_pick_place_drawer`, `franka_pick_place_gears`, `franka_pick_place_mug`, `franka_pick_place_tuna`, `openarm_multi_pick_place`, `openarm_pick_place`, `openarm_pick_place_bottle`, `openarm_pick_place_box`, `openarm_pick_place_can`, `openarm_pick_place_drawer`, `openarm_pick_place_gears`, `openarm_pick_place_mug`, `openarm_pick_place_tuna`, `so101_multi_pick_place`, `so101_pick_place`, `so101_pick_place_cylinder`, `ur10e_pick_place`, `ur10e_pick_place_bottle`, `ur10e_pick_place_box`, `ur10e_pick_place_can`, `ur10e_pick_place_drawer`, `ur10e_pick_place_mug`, `ur10e_pick_place_tuna`
- `sort (9)`: `franka_color_sort`, `franka_line_arrange`, `franka_shape_sort`, `openarm_color_sort`, `openarm_line_arrange`, `openarm_shape_sort`, `so101_color_sort`, `so101_line_arrange`, `so101_shape_sort`
- `stack (7)`: `franka_stack`, `franka_stack_tray`, `openarm_stack`, `openarm_stack_tray`, `so101_stack`, `ur10e_stack`, `ur10e_stack_tray`

### 비파지 (Non-prehensile) - 3

- `reach (3)`: `openarm_reach`, `so101_reach`, `ur10e_reach`

### 배치 (Placement) - 50

- `pick_place (28)`: `franka_multi_pick_place`, `franka_pick_place`, `franka_pick_place_bottle`, `franka_pick_place_box`, `franka_pick_place_can`, `franka_pick_place_drawer`, `franka_pick_place_gears`, `franka_pick_place_mug`, `franka_pick_place_tuna`, `openarm_multi_pick_place`, `openarm_pick_place`, `openarm_pick_place_bottle`, `openarm_pick_place_box`, `openarm_pick_place_can`, `openarm_pick_place_drawer`, `openarm_pick_place_gears`, `openarm_pick_place_mug`, `openarm_pick_place_tuna`, `so101_multi_pick_place`, `so101_pick_place`, `so101_pick_place_cylinder`, `ur10e_pick_place`, `ur10e_pick_place_bottle`, `ur10e_pick_place_box`, `ur10e_pick_place_can`, `ur10e_pick_place_drawer`, `ur10e_pick_place_mug`, `ur10e_pick_place_tuna`
- `cabinet_blocks/ycb (6)`: `franka_cabinet_blocks`, `franka_cabinet_ycb`, `openarm_cabinet_blocks`, `openarm_cabinet_ycb`, `ur10e_cabinet_blocks`, `ur10e_cabinet_ycb`
- `sort (9)`: `franka_color_sort`, `franka_line_arrange`, `franka_shape_sort`, `openarm_color_sort`, `openarm_line_arrange`, `openarm_shape_sort`, `so101_color_sort`, `so101_line_arrange`, `so101_shape_sort`
- `stack (7)`: `franka_stack`, `franka_stack_tray`, `openarm_stack`, `openarm_stack_tray`, `so101_stack`, `ur10e_stack`, `ur10e_stack_tray`

### 관절체 (Articulated) - 3

- `cabinet (3)`: `franka_cabinet`, `openarm_cabinet`, `ur10e_cabinet`

### 삽입/조립 (Insertion/Assembly) - 13

- `assembling_kits (4)`: `franka_assembling_kits`, `openarm_assembling_kits`, `so101_assembling_kits`, `ur10e_assembling_kits`
- `peg_insertion_side (4)`: `franka_peg_insertion_side`, `openarm_peg_insertion_side`, `so101_peg_insertion_side`, `ur10e_peg_insertion_side`
- `plug_charger (4)`: `franka_plug_charger`, `openarm_plug_charger`, `so101_plug_charger`, `ur10e_plug_charger`
- `peg_insert (1)`: `franka_peg_insert`

### 회전 (Rotational) - 4

- `lift_peg_upright (4)`: `franka_lift_peg_upright`, `openarm_lift_peg_upright`, `so101_lift_peg_upright`, `ur10e_lift_peg_upright`

### 변형체 / 접촉

- `변형체 (0)`: 현재 없음
- `접촉 (0)`: 현재 primary 없음

## Mapping Rules

| 이미지 분류 | 세부 동작 | 현재 task family / 패턴 |
| --- | --- | --- |
| `파지 (Prehensile)` | `Grasp-required` | `tasks/*/reach/*.yaml`, pure `tasks/*/cabinet/*_cabinet.yaml`, `tasks/*/assembly/*_peg_insertion_side.yaml`, `tasks/franka/peg_insert/*.yaml`를 제외한 나머지 grasp task |
| `비파지 (Non-prehensile)` | `Reach` | `tasks/*/reach/*.yaml` |
| `배치 (Placement)` | `Place (Pick)`, `Collect / Place (Pick)` | `tasks/*/pick_place/*.yaml` |
| `배치 (Placement)` | `Place (Pick)` | `tasks/*/cabinet/*_cabinet_blocks.yaml`, `*_cabinet_ycb.yaml` |
| `배치 (Placement)` | `Sort`, `Arrange` | `tasks/*/sort/*.yaml` |
| `배치 (Placement)` | `Stack` | `tasks/*/stack/*.yaml` |
| `관절체 (Articulated)` | `Open drawer` | `tasks/*/cabinet/*_cabinet.yaml` |
| `삽입/조립 (Insertion/Assembly)` | `Slot fitting` | `tasks/*/assembly/*_assembling_kits.yaml` |
| `삽입/조립 (Insertion/Assembly)` | `Insert`, `Peg-in-hole` | `tasks/*/assembly/*_peg_insertion_side.yaml`, `tasks/franka/peg_insert/*.yaml` |
| `삽입/조립 (Insertion/Assembly)` | `Insert` | `tasks/*/assembly/*_plug_charger.yaml` |
| `회전 (Rotational)` | `Rotate upright` | `tasks/*/assembly/*_lift_peg_upright.yaml` |
