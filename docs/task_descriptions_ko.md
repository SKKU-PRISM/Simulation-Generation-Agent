# Task 설명 (한글)

대상: task를 빠르게 훑어보는 사용자  
이 문서가 다루는 것: task 한글 설명, 추천 shortlist, 간단한 ADC 검증 표시  
상세 상태와 evidence는 `docs/tasks_overview.md`
Franka만 따로 보려면 `docs/franka_tasks_summary.md`

이 문서는 `tasks/` 아래 task들을 한국어로 빠르게 훑어보기 위한 보조 문서입니다.
설명은 각 YAML의 `goal.description` 또는 `task.description`을 기준으로 정리했고, `ADC verified`는 `run_data_collection.py` 경로까지 실제 실행한 증거가 있는지를 뜻합니다.

## 상태 표기

| 값 | 의미 |
| --- | --- |
| `yes` | ADC 파이프라인까지 실제 실행한 기록이 있음 |
| `unknown` | 아직 이 문서 기준으로 실행 증거를 연결하지 않음 |

- 총 task 수: **78**
- `franka`: **24** tasks
- `openarm`: **24** tasks
- `so101`: **13** tasks
- `ur10e`: **17** tasks

## 임시 우선 사용 후보

현재 임시 운용 대상으로는 `Franka`, `OpenArm`, 그리고 **LeRobot 기반 embodiment인 `SO-101`** 을 우선 본다.  
아래 표는 바로 써볼 가치가 있는 대표 task와, 이번에 추가된 assembly task 중 위 embodiment에 해당하는 항목만 따로 묶은 것이다. `무엇을 하는가`는 실제 task 목표를 짧게 요약한 것이다.

| Embodiment | 묶음 | 추천 task | 무엇을 하는가 | ADC verified | YAML |
| --- | --- | --- | --- | --- | --- |
| `franka` | `pick_place` | `FrankaPickPlaceBox` | sugar box를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/franka/pick_place/franka_pick_place_box.yaml` |
| `franka` | `pick_place` | `FrankaPickPlaceMug` | 머그를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/franka/pick_place/franka_pick_place_mug.yaml` |
| `franka` | `pick_place` | `FrankaPickPlaceGears` | 기어들과 너트를 집어 트레이 안에 정리한다. | `unknown` | `tasks/franka/pick_place/franka_pick_place_gears.yaml` |
| `franka` | `stack` | `FrankaStack` | 큐브 3개를 파랑-빨강-초록 순서로 쌓는다. | `yes` | `tasks/franka/stack/franka_stack.yaml` |
| `franka` | `sort` | `FrankaShapeSort` | 큐브와 구체를 shape 기준으로 다른 bin에 분류한다. | `unknown` | `tasks/franka/sort/franka_shape_sort.yaml` |
| `franka` | `new assembly` | `FrankaAssemblingKits`, `FrankaLiftPegUpright`, `FrankaPegInsertionSide`, `FrankaPlugCharger` | shape fitting, peg 세우기, peg 삽입, charger 삽입을 다루는 assembly 4종이다. | `unknown` | `tasks/franka/assembly/` |
| `openarm` | `pick_place` | `OpenArmPickPlaceBox` | sugar box를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_box.yaml` |
| `openarm` | `pick_place` | `OpenArmPickPlaceMug` | 머그를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_mug.yaml` |
| `openarm` | `pick_place` | `OpenArmPickPlaceGears` | 기어들과 너트를 집어 트레이 안에 정리한다. | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_gears.yaml` |
| `openarm` | `stack` | `OpenArmStack` | 큐브 3개를 파랑-빨강-초록 순서로 쌓는다. | `yes` | `tasks/openarm/stack/openarm_stack.yaml` |
| `openarm` | `sort` | `OpenArmShapeSort` | 큐브와 구체를 shape 기준으로 다른 bin에 분류한다. | `unknown` | `tasks/openarm/sort/openarm_shape_sort.yaml` |
| `openarm` | `new assembly` | `OpenArmAssemblingKits`, `OpenArmLiftPegUpright`, `OpenArmPegInsertionSide`, `OpenArmPlugCharger` | shape fitting, peg 세우기, peg 삽입, charger 삽입을 다루는 assembly 4종이다. | `unknown` | `tasks/openarm/assembly/` |
| `so101` | `pick_place` | `SO101PickPlace` | cube를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/so101/pick_place/so101_pick_place.yaml` |
| `so101` | `pick_place` | `SO101PickPlaceCylinder` | 원기둥을 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/so101/pick_place/so101_pick_place_cylinder.yaml` |
| `so101` | `pick_place` | `SO101MultiPickPlace` | 색 큐브 3개를 모두 금색 수집 구역으로 모은다. | `unknown` | `tasks/so101/pick_place/so101_multi_pick_place.yaml` |
| `so101` | `stack` | `SO101Stack` | 큐브 3개를 파랑-빨강-초록 순서로 쌓는다. | `yes` | `tasks/so101/stack/so101_stack.yaml` |
| `so101` | `sort` | `SO101ShapeSort` | 큐브와 구체를 shape 기준으로 다른 bin에 분류한다. | `unknown` | `tasks/so101/sort/so101_shape_sort.yaml` |
| `so101` | `new assembly` | `SO101AssemblingKits`, `SO101LiftPegUpright`, `SO101PegInsertionSide`, `SO101PlugCharger` | shape fitting, peg 세우기, peg 삽입, charger 삽입을 다루는 assembly 4종이다. | `unknown` | `tasks/so101/assembly/` |

## Franka

| 카테고리 | Task 이름 | 무엇을 하는가 | ADC verified | YAML |
| --- | --- | --- | --- | --- |
| `assembly` | `FrankaAssemblingKits` | 무작위 위치에 놓인 도형 조각을 집어 키트 트레이의 맞는 컷아웃 슬롯에 끼운다. | `unknown` | `tasks/franka/assembly/franka_assembling_kits.yaml` |
| `assembly` | `FrankaLiftPegUpright` | 테이블에 눕혀진 2색 peg를 집어 수직 자세로 세운다. | `unknown` | `tasks/franka/assembly/franka_lift_peg_upright.yaml` |
| `assembly` | `FrankaPegInsertionSide` | peg를 집어 주황색 끝을 박스 측면 구멍에 옆으로 삽입한다. | `unknown` | `tasks/franka/assembly/franka_peg_insertion_side.yaml` |
| `assembly` | `FrankaPlugCharger` | 충전기를 집어 플러그 핀을 리셉터클 슬롯에 끼운다. | `unknown` | `tasks/franka/assembly/franka_plug_charger.yaml` |
| `cabinet` | `FrankaCabinet` | 손잡이를 잡아 캐비닛 상단 서랍을 당겨 연다. | `unknown` | `tasks/franka/cabinet/franka_cabinet.yaml` |
| `cabinet` | `FrankaCabinetBlocks` | 캐비닛 위의 색 블록 3개를 모두 집어 타깃 영역으로 옮긴다. | `unknown` | `tasks/franka/cabinet/franka_cabinet_blocks.yaml` |
| `cabinet` | `FrankaCabinetYCB` | 캐비닛 위의 YCB 물체 3개를 모두 집어 타깃 영역으로 옮긴다. | `unknown` | `tasks/franka/cabinet/franka_cabinet_ycb.yaml` |
| `lift` | `FrankaLift` | cube를 집어 테이블 위 목표 자세까지 들어 올린다. | `unknown` | `tasks/franka/lift/franka_lift.yaml` |
| `lift` | `FrankaLiftSugarBox` | sugar box를 집어 테이블 위 목표 자세까지 들어 올린다. | `unknown` | `tasks/franka/lift/franka_lift_sugar_box.yaml` |
| `peg_insert` | `FrankaFactoryPegInsert` | 8mm peg를 테이블 위 8mm 소켓 구멍에 삽입한다. | `unknown` | `tasks/franka/peg_insert/franka_peg_insert.yaml` |
| `pick_place` | `FrankaMultiPickPlace` | 색 블록 3개를 모두 금색 수집 구역으로 모은다. | `unknown` | `tasks/franka/pick_place/franka_multi_pick_place.yaml` |
| `pick_place` | `FrankaPickPlace` | DexCube를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/franka/pick_place/franka_pick_place.yaml` |
| `pick_place` | `FrankaPickPlaceBottle` | 머스터드 병을 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/franka/pick_place/franka_pick_place_bottle.yaml` |
| `pick_place` | `FrankaPickPlaceBox` | sugar box를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/franka/pick_place/franka_pick_place_box.yaml` |
| `pick_place` | `FrankaPickPlaceCan` | 토마토 수프 캔을 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/franka/pick_place/franka_pick_place_can.yaml` |
| `pick_place` | `FrankaPickPlaceDrawer` | 테이블의 DexCube를 집어 작은 서랍 박스 위의 초록 마커 위치에 놓는다. | `unknown` | `tasks/franka/pick_place/franka_pick_place_drawer.yaml` |
| `pick_place` | `FrankaPickPlaceGears` | 테이블 위 기어들과 너트를 모두 집어 트레이 안에 넣는다. | `unknown` | `tasks/franka/pick_place/franka_pick_place_gears.yaml` |
| `pick_place` | `FrankaPickPlaceMug` | 머그를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/franka/pick_place/franka_pick_place_mug.yaml` |
| `pick_place` | `FrankaPickPlaceTuna` | 참치 캔을 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/franka/pick_place/franka_pick_place_tuna.yaml` |
| `sort` | `FrankaColorSort` | 색 블록을 각자 대응하는 색 영역으로 분류한다. 파랑->파랑, 빨강->빨강, 초록->초록. | `unknown` | `tasks/franka/sort/franka_color_sort.yaml` |
| `sort` | `FrankaLineArrange` | 블록 3개를 x=0.55 선상에 일렬로 배치한다. 파랑(y=-0.06), 빨강(y=0.0), 초록(y=0.06). | `unknown` | `tasks/franka/sort/franka_line_arrange.yaml` |
| `sort` | `FrankaShapeSort` | 도형 종류별로 분류한다. 큐브는 보라색 빈, 구체는 로즈색 빈에 놓는다. | `unknown` | `tasks/franka/sort/franka_shape_sort.yaml` |
| `stack` | `FrankaStack` | 큐브 3개를 파랑(아래) -> 빨강(중간) -> 초록(위) 순서로 쌓는다. | `yes` | `tasks/franka/stack/franka_stack.yaml` |
| `stack` | `FrankaStackTray` | 트레이 안에서 파랑(아래) -> 빨강(중간) -> 초록(위) 순서로 큐브를 쌓는다. | `unknown` | `tasks/franka/stack/franka_stack_tray.yaml` |

## OpenArm

| 카테고리 | Task 이름 | 무엇을 하는가 | ADC verified | YAML |
| --- | --- | --- | --- | --- |
| `assembly` | `OpenArmAssemblingKits` | 무작위 위치에 놓인 도형 조각을 집어 키트 트레이의 맞는 컷아웃 슬롯에 끼운다. | `unknown` | `tasks/openarm/assembly/openarm_assembling_kits.yaml` |
| `assembly` | `OpenArmLiftPegUpright` | 테이블에 눕혀진 2색 peg를 집어 수직 자세로 세운다. | `unknown` | `tasks/openarm/assembly/openarm_lift_peg_upright.yaml` |
| `assembly` | `OpenArmPegInsertionSide` | peg를 집어 주황색 끝을 박스 측면 구멍에 옆으로 삽입한다. | `unknown` | `tasks/openarm/assembly/openarm_peg_insertion_side.yaml` |
| `assembly` | `OpenArmPlugCharger` | 충전기를 집어 플러그 핀을 리셉터클 슬롯에 끼운다. | `unknown` | `tasks/openarm/assembly/openarm_plug_charger.yaml` |
| `cabinet` | `OpenArmCabinet` | 손잡이를 잡아 캐비닛 하단 서랍을 당겨 연다. | `unknown` | `tasks/openarm/cabinet/openarm_cabinet.yaml` |
| `cabinet` | `OpenArmCabinetBlocks` | 캐비닛 위의 색 블록 3개를 모두 집어 타깃 영역으로 옮긴다. | `unknown` | `tasks/openarm/cabinet/openarm_cabinet_blocks.yaml` |
| `cabinet` | `OpenArmCabinetYCB` | 캐비닛 위의 YCB 물체 3개를 모두 집어 타깃 영역으로 옮긴다. | `unknown` | `tasks/openarm/cabinet/openarm_cabinet_ycb.yaml` |
| `lift` | `OpenArmLift` | cube를 집어 테이블 위 목표 자세까지 들어 올린다. | `unknown` | `tasks/openarm/lift/openarm_lift.yaml` |
| `lift` | `OpenArmLiftSugarBox` | sugar box를 집어 테이블 위 목표 자세까지 들어 올린다. | `unknown` | `tasks/openarm/lift/openarm_lift_sugar_box.yaml` |
| `pick_place` | `OpenArmMultiPickPlace` | 색 블록 3개를 모두 금색 수집 구역으로 모은다. | `unknown` | `tasks/openarm/pick_place/openarm_multi_pick_place.yaml` |
| `pick_place` | `OpenArmPickPlace` | DexCube를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/openarm/pick_place/openarm_pick_place.yaml` |
| `pick_place` | `OpenArmPickPlaceBottle` | 머스터드 병을 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_bottle.yaml` |
| `pick_place` | `OpenArmPickPlaceBox` | sugar box를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_box.yaml` |
| `pick_place` | `OpenArmPickPlaceCan` | 토마토 수프 캔을 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_can.yaml` |
| `pick_place` | `OpenArmPickPlaceDrawer` | 테이블의 DexCube를 집어 작은 서랍 박스 위의 초록 마커 위치에 놓는다. | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_drawer.yaml` |
| `pick_place` | `OpenArmPickPlaceGears` | 테이블 위 기어들과 너트를 모두 집어 트레이 안에 넣는다. | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_gears.yaml` |
| `pick_place` | `OpenArmPickPlaceMug` | 머그를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_mug.yaml` |
| `pick_place` | `OpenArmPickPlaceTuna` | 참치 캔을 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_tuna.yaml` |
| `reach` | `OpenArmReach` | 명령된 엔드이펙터 6D pose(위치+자세)를 추종한다. | `unknown` | `tasks/openarm/reach/openarm_reach.yaml` |
| `sort` | `OpenArmColorSort` | 색 블록을 각자 대응하는 색 영역으로 분류한다. 파랑->파랑, 빨강->빨강, 초록->초록. | `unknown` | `tasks/openarm/sort/openarm_color_sort.yaml` |
| `sort` | `OpenArmLineArrange` | 블록 3개를 x=0.42 선상에 일렬로 배치한다. 파랑(y=-0.06), 빨강(y=0.0), 초록(y=0.06). | `unknown` | `tasks/openarm/sort/openarm_line_arrange.yaml` |
| `sort` | `OpenArmShapeSort` | 도형 종류별로 분류한다. 큐브는 보라색 빈, 구체는 로즈색 빈에 놓는다. | `unknown` | `tasks/openarm/sort/openarm_shape_sort.yaml` |
| `stack` | `OpenArmStack` | 큐브 3개를 파랑(아래) -> 빨강(중간) -> 초록(위) 순서로 쌓는다. | `yes` | `tasks/openarm/stack/openarm_stack.yaml` |
| `stack` | `OpenArmStackTray` | 트레이 안에서 파랑(아래) -> 빨강(중간) -> 초록(위) 순서로 큐브를 쌓는다. | `unknown` | `tasks/openarm/stack/openarm_stack_tray.yaml` |

## SO-101

| 카테고리 | Task 이름 | 무엇을 하는가 | ADC verified | YAML |
| --- | --- | --- | --- | --- |
| `assembly` | `SO101AssemblingKits` | 무작위 위치에 놓인 도형 조각을 집어 키트 트레이의 맞는 컷아웃 슬롯에 끼운다. | `unknown` | `tasks/so101/assembly/so101_assembling_kits.yaml` |
| `assembly` | `SO101LiftPegUpright` | 데스크톱에 눕혀진 2색 peg를 집어 수직 자세로 세운다. | `unknown` | `tasks/so101/assembly/so101_lift_peg_upright.yaml` |
| `assembly` | `SO101PegInsertionSide` | peg를 집어 주황색 끝을 박스 측면 구멍에 옆으로 삽입한다. | `unknown` | `tasks/so101/assembly/so101_peg_insertion_side.yaml` |
| `assembly` | `SO101PlugCharger` | 충전기를 집어 플러그 핀을 리셉터클 슬롯에 끼운다. | `unknown` | `tasks/so101/assembly/so101_plug_charger.yaml` |
| `lift` | `SO101Lift` | cube를 집어 데스크톱 위 목표 자세까지 들어 올린다. | `unknown` | `tasks/so101/lift/so101_lift.yaml` |
| `pick_place` | `SO101MultiPickPlace` | 색 큐브 3개를 모두 금색 수집 구역으로 모은다. | `unknown` | `tasks/so101/pick_place/so101_multi_pick_place.yaml` |
| `pick_place` | `SO101PickPlace` | cube를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/so101/pick_place/so101_pick_place.yaml` |
| `pick_place` | `SO101PickPlaceCylinder` | 원기둥을 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/so101/pick_place/so101_pick_place_cylinder.yaml` |
| `reach` | `SO101Reach` | 명령된 엔드이펙터 6D pose(위치+자세)를 추종한다. | `unknown` | `tasks/so101/reach/so101_reach.yaml` |
| `sort` | `SO101ColorSort` | 색 큐브를 각자 대응하는 색 영역으로 분류한다. 파랑->파랑, 빨강->빨강, 초록->초록. | `unknown` | `tasks/so101/sort/so101_color_sort.yaml` |
| `sort` | `SO101LineArrange` | 큐브 3개를 x=0.18 선상에 일렬로 배치한다. 파랑(y=-0.04), 빨강(y=0.0), 초록(y=0.04). | `unknown` | `tasks/so101/sort/so101_line_arrange.yaml` |
| `sort` | `SO101ShapeSort` | 도형 종류별로 분류한다. 큐브는 보라색 빈, 구체는 로즈색 빈에 놓는다. | `unknown` | `tasks/so101/sort/so101_shape_sort.yaml` |
| `stack` | `SO101Stack` | 큐브 3개를 파랑(아래) -> 빨강(중간) -> 초록(위) 순서로 쌓는다. | `yes` | `tasks/so101/stack/so101_stack.yaml` |

## UR10e

| 카테고리 | Task 이름 | 무엇을 하는가 | ADC verified | YAML |
| --- | --- | --- | --- | --- |
| `assembly` | `UR10eAssemblingKits` | 무작위 위치에 놓인 도형 조각을 집어 키트 트레이의 맞는 컷아웃 슬롯에 끼운다. | `unknown` | `tasks/ur10e/assembly/ur10e_assembling_kits.yaml` |
| `assembly` | `UR10eLiftPegUpright` | 테이블에 눕혀진 2색 peg를 집어 수직 자세로 세운다. | `unknown` | `tasks/ur10e/assembly/ur10e_lift_peg_upright.yaml` |
| `assembly` | `UR10ePegInsertionSide` | peg를 집어 주황색 끝을 박스 측면 구멍에 옆으로 삽입한다. | `unknown` | `tasks/ur10e/assembly/ur10e_peg_insertion_side.yaml` |
| `assembly` | `UR10ePlugCharger` | 충전기를 집어 플러그 핀을 리셉터클 슬롯에 끼운다. | `unknown` | `tasks/ur10e/assembly/ur10e_plug_charger.yaml` |
| `cabinet` | `UR10eCabinet` | 손잡이를 당겨 캐비닛 상단 서랍을 연다. | `unknown` | `tasks/ur10e/cabinet/ur10e_cabinet.yaml` |
| `cabinet` | `UR10eCabinetBlocks` | 캐비닛 위의 색 블록 3개를 모두 집어 타깃 영역으로 옮긴다. | `unknown` | `tasks/ur10e/cabinet/ur10e_cabinet_blocks.yaml` |
| `cabinet` | `UR10eCabinetYCB` | 캐비닛 위의 YCB 물체 3개를 모두 집어 타깃 영역으로 옮긴다. | `unknown` | `tasks/ur10e/cabinet/ur10e_cabinet_ycb.yaml` |
| `pick_place` | `UR10ePickPlace` | Robotiq 2F-85 그리퍼로 DexCube를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place.yaml` |
| `pick_place` | `UR10ePickPlaceBottle` | Robotiq 2F-85 그리퍼로 머스터드 병을 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place_bottle.yaml` |
| `pick_place` | `UR10ePickPlaceBox` | Robotiq 2F-85 그리퍼로 sugar box를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place_box.yaml` |
| `pick_place` | `UR10ePickPlaceCan` | Robotiq 2F-85 그리퍼로 토마토 수프 캔을 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place_can.yaml` |
| `pick_place` | `UR10ePickPlaceDrawer` | Robotiq 2F-85 그리퍼로 DexCube를 집어 작은 서랍 박스 위의 초록 마커 위치에 놓는다. | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place_drawer.yaml` |
| `pick_place` | `UR10ePickPlaceMug` | Robotiq 2F-85 그리퍼로 머그를 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place_mug.yaml` |
| `pick_place` | `UR10ePickPlaceTuna` | Robotiq 2F-85 그리퍼로 참치 캔을 집어 초록 타깃 마커 위에 놓는다. | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place_tuna.yaml` |
| `reach` | `UR10eReach` | UR10e 엔드이펙터를 빨간 타깃 마커 위치로 이동시킨다. | `unknown` | `tasks/ur10e/reach/ur10e_reach.yaml` |
| `stack` | `UR10eStack` | 큐브 3개를 파랑(아래) - 빨강(중간) - 초록(위) 순서로 쌓는다. | `yes` | `tasks/ur10e/stack/ur10e_stack.yaml` |
| `stack` | `UR10eStackTray` | Robotiq 2F-85 그리퍼로 트레이 안에서 파랑(아래) -> 빨강(중간) -> 초록(위) 순서로 큐브를 쌓는다. | `unknown` | `tasks/ur10e/stack/ur10e_stack_tray.yaml` |