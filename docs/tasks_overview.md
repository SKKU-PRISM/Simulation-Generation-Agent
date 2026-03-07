# Task Overview

이 문서는 `tasks/` 아래 전체 **78개** task YAML과 현재까지 확인된 실행 상태를 한 곳에 모아 둔 상태 보드입니다.
여기서 **ADC 검증**은 `run_data_collection.py` 경로를 실제로 실행했는지를 뜻하며, **dataset_generated**는 `raw_dataset/` 생성 기준입니다.

## Status Legend

| Column | Meaning | Values |
| --- | --- | --- |
| `isaaclab_eval` | `run_isaac_lab.py ... --evaluate` 기준 평가 상태 | `pass`, `fail`, `unknown` |
| `adc_verified` | `run_data_collection.py` 경로를 실제로 실행했는지 | `yes`, `no`, `unknown` |
| `dataset_generated` | `raw_dataset/` 생성까지 확인됐는지 | `yes`, `no`, `unknown` |
| `task_success` | ADC 실행 결과에서 task goal 달성까지 확인됐는지 | `yes`, `no`, `unknown` |

## Summary

- Total tasks: **78**
- Known `isaaclab_eval=pass`: **2**
- Known `adc_verified=yes`: **2**
- Known `dataset_generated=yes`: **2**
- Known `task_success=yes`: **1**

| Robot | Tasks | Categories | IsaacLab pass | ADC verified | Dataset generated | Task success |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `franka` | 24 | assembly (4), cabinet (3), lift (2), peg_insert (1), pick_place (9), sort (3), stack (2) | 1 | 1 | 1 | 1 |
| `openarm` | 24 | assembly (4), cabinet (3), lift (2), pick_place (9), reach (1), sort (3), stack (2) | 0 | 0 | 0 | 0 |
| `so101` | 13 | assembly (4), lift (1), pick_place (3), reach (1), sort (3), stack (1) | 0 | 0 | 0 | 0 |
| `ur10` | 4 | assembly (4) | 0 | 0 | 0 | 0 |
| `ur10e` | 13 | cabinet (3), pick_place (7), reach (1), stack (2) | 1 | 1 | 1 | 0 |

## Task Descriptions

task 이름과 실제 목표를 빠르게 훑어볼 수 있는 섹션입니다. 여기의 `ADC verified`는 해당 task를 데이터 수집 파이프라인까지 실제로 실행해 봤는지 표시합니다.

### franka

| Category | Task | What it does | ADC verified | YAML |
| --- | --- | --- | --- | --- |
| `assembly` | `FrankaAssemblingKits` | Pick up the randomly misplaced shape and insert it into the matching cutout slot on the kit tray | `unknown` | `tasks/franka/assembly/franka_assembling_kits.yaml` |
| `assembly` | `FrankaLiftPegUpright` | Lift the two-tone peg from lying flat to an upright vertical position on the table | `unknown` | `tasks/franka/assembly/franka_lift_peg_upright.yaml` |
| `assembly` | `FrankaPegInsertionSide` | Pick up the peg and insert the orange end sideways into the box hole | `unknown` | `tasks/franka/assembly/franka_peg_insertion_side.yaml` |
| `assembly` | `FrankaPlugCharger` | Pick up the charger and insert its prongs into the receptacle slots | `unknown` | `tasks/franka/assembly/franka_plug_charger.yaml` |
| `cabinet` | `FrankaCabinet` | Open the top drawer of the cabinet by grasping the handle and pulling | `unknown` | `tasks/franka/cabinet/franka_cabinet.yaml` |
| `cabinet` | `FrankaCabinetBlocks` | Pick all 3 colored blocks from cabinet top and place them on the target zone | `unknown` | `tasks/franka/cabinet/franka_cabinet_blocks.yaml` |
| `cabinet` | `FrankaCabinetYCB` | Pick all 3 YCB objects from cabinet top and place them on the target zone | `unknown` | `tasks/franka/cabinet/franka_cabinet_ycb.yaml` |
| `lift` | `FrankaLift` | Lift the cube to the commanded target pose above the table | `unknown` | `tasks/franka/lift/franka_lift.yaml` |
| `lift` | `FrankaLiftSugarBox` | Lift the sugar box to the commanded target pose above the table | `unknown` | `tasks/franka/lift/franka_lift_sugar_box.yaml` |
| `peg_insert` | `FrankaFactoryPegInsert` | Insert the 8mm peg into the 8mm hole socket on the table | `unknown` | `tasks/franka/peg_insert/franka_peg_insert.yaml` |
| `pick_place` | `FrankaMultiPickPlace` | Collect all three colored blocks onto the gold collection zone | `unknown` | `tasks/franka/pick_place/franka_multi_pick_place.yaml` |
| `pick_place` | `FrankaPickPlace` | Pick the DexCube and place it on the green target marker | `unknown` | `tasks/franka/pick_place/franka_pick_place.yaml` |
| `pick_place` | `FrankaPickPlaceBottle` | Pick the mustard bottle and place it on the green target marker | `unknown` | `tasks/franka/pick_place/franka_pick_place_bottle.yaml` |
| `pick_place` | `FrankaPickPlaceBox` | Pick the sugar box and place it on the green target marker | `unknown` | `tasks/franka/pick_place/franka_pick_place_box.yaml` |
| `pick_place` | `FrankaPickPlaceCan` | Pick the tomato soup can and place it on the green target marker | `unknown` | `tasks/franka/pick_place/franka_pick_place_can.yaml` |
| `pick_place` | `FrankaPickPlaceDrawer` | Pick the DexCube from the table and place it on top of the small drawer box (green marker) | `unknown` | `tasks/franka/pick_place/franka_pick_place_drawer.yaml` |
| `pick_place` | `FrankaPickPlaceGears` | Pick all gears and the nut from the table and place them into the tray | `unknown` | `tasks/franka/pick_place/franka_pick_place_gears.yaml` |
| `pick_place` | `FrankaPickPlaceMug` | Pick the mug and place it on the green target marker | `unknown` | `tasks/franka/pick_place/franka_pick_place_mug.yaml` |
| `pick_place` | `FrankaPickPlaceTuna` | Pick the tuna fish can and place it on the green target marker | `unknown` | `tasks/franka/pick_place/franka_pick_place_tuna.yaml` |
| `sort` | `FrankaColorSort` | Sort each colored block onto its matching colored zone: blue block -> blue zone, red block -> red zone, green block -> green zone | `unknown` | `tasks/franka/sort/franka_color_sort.yaml` |
| `sort` | `FrankaLineArrange` | Arrange blocks in a line at x=0.55: blue (y=-0.06), red (y=0.0), green (y=0.06) | `unknown` | `tasks/franka/sort/franka_line_arrange.yaml` |
| `sort` | `FrankaShapeSort` | Sort objects by shape: place cubes on the purple bin, spheres on the rose bin | `unknown` | `tasks/franka/sort/franka_shape_sort.yaml` |
| `stack` | `FrankaStack` | Stack cubes in order: Cube_1/Blue (bottom) -> Cube_2/Red (middle) -> Cube_3/Green (top) | `yes` | `tasks/franka/stack/franka_stack.yaml` |
| `stack` | `FrankaStackTray` | Stack cubes inside the tray in order: Blue (bottom) -> Red (middle) -> Green (top) | `unknown` | `tasks/franka/stack/franka_stack_tray.yaml` |

### openarm

| Category | Task | What it does | ADC verified | YAML |
| --- | --- | --- | --- | --- |
| `assembly` | `OpenArmAssemblingKits` | Pick up the randomly misplaced shape and insert it into the matching cutout slot on the kit tray | `unknown` | `tasks/openarm/assembly/openarm_assembling_kits.yaml` |
| `assembly` | `OpenArmLiftPegUpright` | Lift the two-tone peg from lying flat to an upright vertical position on the table | `unknown` | `tasks/openarm/assembly/openarm_lift_peg_upright.yaml` |
| `assembly` | `OpenArmPegInsertionSide` | Pick up the peg and insert the orange end sideways into the box hole | `unknown` | `tasks/openarm/assembly/openarm_peg_insertion_side.yaml` |
| `assembly` | `OpenArmPlugCharger` | Pick up the charger and insert its prongs into the receptacle slots | `unknown` | `tasks/openarm/assembly/openarm_plug_charger.yaml` |
| `cabinet` | `OpenArmCabinet` | Open the bottom drawer of the cabinet by grasping the handle and pulling | `unknown` | `tasks/openarm/cabinet/openarm_cabinet.yaml` |
| `cabinet` | `OpenArmCabinetBlocks` | Pick all 3 colored blocks from cabinet top and place them on the target zone | `unknown` | `tasks/openarm/cabinet/openarm_cabinet_blocks.yaml` |
| `cabinet` | `OpenArmCabinetYCB` | Pick all 3 YCB objects from cabinet top and place them on the target zone | `unknown` | `tasks/openarm/cabinet/openarm_cabinet_ycb.yaml` |
| `lift` | `OpenArmLift` | Lift the cube to the commanded target pose above the table | `unknown` | `tasks/openarm/lift/openarm_lift.yaml` |
| `lift` | `OpenArmLiftSugarBox` | Lift the sugar box to the commanded target pose above the table | `unknown` | `tasks/openarm/lift/openarm_lift_sugar_box.yaml` |
| `pick_place` | `OpenArmMultiPickPlace` | Collect all three colored blocks onto the gold collection zone | `unknown` | `tasks/openarm/pick_place/openarm_multi_pick_place.yaml` |
| `pick_place` | `OpenArmPickPlace` | Pick the DexCube and place it on the green target marker | `unknown` | `tasks/openarm/pick_place/openarm_pick_place.yaml` |
| `pick_place` | `OpenArmPickPlaceBottle` | Pick the mustard bottle and place it on the green target marker | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_bottle.yaml` |
| `pick_place` | `OpenArmPickPlaceBox` | Pick the sugar box and place it on the green target marker | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_box.yaml` |
| `pick_place` | `OpenArmPickPlaceCan` | Pick the tomato soup can and place it on the green target marker | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_can.yaml` |
| `pick_place` | `OpenArmPickPlaceDrawer` | Pick the DexCube from the table and place it on top of the small drawer box (green marker) | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_drawer.yaml` |
| `pick_place` | `OpenArmPickPlaceGears` | Pick all gears and the nut from the table and place them into the tray | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_gears.yaml` |
| `pick_place` | `OpenArmPickPlaceMug` | Pick the mug and place it on the green target marker | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_mug.yaml` |
| `pick_place` | `OpenArmPickPlaceTuna` | Pick the tuna fish can and place it on the green target marker | `unknown` | `tasks/openarm/pick_place/openarm_pick_place_tuna.yaml` |
| `reach` | `OpenArmReach` | Track the commanded end-effector pose (position + orientation) in 6D workspace | `unknown` | `tasks/openarm/reach/openarm_reach.yaml` |
| `sort` | `OpenArmColorSort` | Sort each colored block onto its matching colored zone: blue block -> blue zone, red block -> red zone, green block -> green zone | `unknown` | `tasks/openarm/sort/openarm_color_sort.yaml` |
| `sort` | `OpenArmLineArrange` | Arrange blocks in a line at x=0.42: blue (y=-0.06), red (y=0.0), green (y=0.06) | `unknown` | `tasks/openarm/sort/openarm_line_arrange.yaml` |
| `sort` | `OpenArmShapeSort` | Sort objects by shape: place cubes on the purple bin, spheres on the rose bin | `unknown` | `tasks/openarm/sort/openarm_shape_sort.yaml` |
| `stack` | `OpenArmStack` | Stack cubes in order: Cube_1/Blue (bottom) -> Cube_2/Red (middle) -> Cube_3/Green (top) | `unknown` | `tasks/openarm/stack/openarm_stack.yaml` |
| `stack` | `OpenArmStackTray` | Stack cubes inside the tray in order: Blue (bottom) -> Red (middle) -> Green (top) | `unknown` | `tasks/openarm/stack/openarm_stack_tray.yaml` |

### so101

| Category | Task | What it does | ADC verified | YAML |
| --- | --- | --- | --- | --- |
| `assembly` | `SO101AssemblingKits` | Pick up the randomly misplaced shape and insert it into the matching cutout slot on the kit tray | `unknown` | `tasks/so101/assembly/so101_assembling_kits.yaml` |
| `assembly` | `SO101LiftPegUpright` | Lift the two-tone peg from lying flat to an upright vertical position on the desktop | `unknown` | `tasks/so101/assembly/so101_lift_peg_upright.yaml` |
| `assembly` | `SO101PegInsertionSide` | Pick up the peg and insert the orange end sideways into the box hole | `unknown` | `tasks/so101/assembly/so101_peg_insertion_side.yaml` |
| `assembly` | `SO101PlugCharger` | Pick up the charger and insert its prongs into the receptacle slots | `unknown` | `tasks/so101/assembly/so101_plug_charger.yaml` |
| `lift` | `SO101Lift` | Lift the cube to the commanded target pose above the desktop | `unknown` | `tasks/so101/lift/so101_lift.yaml` |
| `pick_place` | `SO101MultiPickPlace` | Collect all three colored cubes onto the gold collection zone | `unknown` | `tasks/so101/pick_place/so101_multi_pick_place.yaml` |
| `pick_place` | `SO101PickPlace` | Pick the cube and place it on the green target marker | `unknown` | `tasks/so101/pick_place/so101_pick_place.yaml` |
| `pick_place` | `SO101PickPlaceCylinder` | Pick the cylinder and place it on the green target marker | `unknown` | `tasks/so101/pick_place/so101_pick_place_cylinder.yaml` |
| `reach` | `SO101Reach` | Track the commanded end-effector pose (position + orientation) in 6D workspace | `unknown` | `tasks/so101/reach/so101_reach.yaml` |
| `sort` | `SO101ColorSort` | Sort each colored cube onto its matching colored zone: blue -> blue zone, red -> red zone, green -> green zone | `unknown` | `tasks/so101/sort/so101_color_sort.yaml` |
| `sort` | `SO101LineArrange` | Arrange cubes in a line at x=0.18: blue (y=-0.04), red (y=0.0), green (y=0.04) | `unknown` | `tasks/so101/sort/so101_line_arrange.yaml` |
| `sort` | `SO101ShapeSort` | Sort objects by shape: place cubes on the purple bin, spheres on the rose bin | `unknown` | `tasks/so101/sort/so101_shape_sort.yaml` |
| `stack` | `SO101Stack` | Stack cubes in order: Cube_1/Blue (bottom) -> Cube_2/Red (middle) -> Cube_3/Green (top) | `unknown` | `tasks/so101/stack/so101_stack.yaml` |

### ur10

| Category | Task | What it does | ADC verified | YAML |
| --- | --- | --- | --- | --- |
| `assembly` | `UR10AssemblingKits` | Pick up the randomly misplaced shape and insert it into the matching cutout slot on the kit tray | `unknown` | `tasks/ur10/assembly/ur10_assembling_kits.yaml` |
| `assembly` | `UR10LiftPegUpright` | Lift the two-tone peg from lying flat to an upright vertical position on the table | `unknown` | `tasks/ur10/assembly/ur10_lift_peg_upright.yaml` |
| `assembly` | `UR10PegInsertionSide` | Pick up the peg and insert the orange end sideways into the box hole | `unknown` | `tasks/ur10/assembly/ur10_peg_insertion_side.yaml` |
| `assembly` | `UR10PlugCharger` | Pick up the charger and insert its prongs into the receptacle slots | `unknown` | `tasks/ur10/assembly/ur10_plug_charger.yaml` |

### ur10e

| Category | Task | What it does | ADC verified | YAML |
| --- | --- | --- | --- | --- |
| `cabinet` | `UR10eCabinet` | Open the top drawer of the cabinet by pulling the handle | `unknown` | `tasks/ur10e/cabinet/ur10e_cabinet.yaml` |
| `cabinet` | `UR10eCabinetBlocks` | Pick all 3 colored blocks from cabinet top and place them on the target zone | `unknown` | `tasks/ur10e/cabinet/ur10e_cabinet_blocks.yaml` |
| `cabinet` | `UR10eCabinetYCB` | Pick all 3 YCB objects from cabinet top and place them on the target zone | `unknown` | `tasks/ur10e/cabinet/ur10e_cabinet_ycb.yaml` |
| `pick_place` | `UR10ePickPlace` | Pick the DexCube with Robotiq 2F-85 gripper and place it on the green target marker | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place.yaml` |
| `pick_place` | `UR10ePickPlaceBottle` | Pick the mustard bottle with Robotiq 2F-85 gripper and place it on the green target marker | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place_bottle.yaml` |
| `pick_place` | `UR10ePickPlaceBox` | Pick the sugar box with the Robotiq 2F-85 gripper and place it on the green target marker | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place_box.yaml` |
| `pick_place` | `UR10ePickPlaceCan` | Pick the tomato soup can with the Robotiq 2F-85 gripper and place it on the green target marker | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place_can.yaml` |
| `pick_place` | `UR10ePickPlaceDrawer` | Pick the DexCube with Robotiq 2F-85 gripper and place it on top of the small drawer box (green marker) | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place_drawer.yaml` |
| `pick_place` | `UR10ePickPlaceMug` | Pick the mug with the Robotiq 2F-85 gripper and place it on the green target marker | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place_mug.yaml` |
| `pick_place` | `UR10ePickPlaceTuna` | Pick the tuna fish can with Robotiq 2F-85 gripper and place it on the green target marker | `unknown` | `tasks/ur10e/pick_place/ur10e_pick_place_tuna.yaml` |
| `reach` | `UR10eReach` | Move the UR10e end-effector to the red target marker position | `unknown` | `tasks/ur10e/reach/ur10e_reach.yaml` |
| `stack` | `UR10eStack` | Stack the three cubes (blue on bottom, red in middle, green on top) | `yes` | `tasks/ur10e/stack/ur10e_stack.yaml` |
| `stack` | `UR10eStackTray` | Stack cubes inside the tray with Robotiq 2F-85 gripper in order: Blue (bottom) -> Red (middle) -> Green (top) | `unknown` | `tasks/ur10e/stack/ur10e_stack_tray.yaml` |

## Task Board

### franka

| Category | Task | YAML | IsaacLab | ADC verified | Dataset | Task success | Notes | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `assembly` | `FrankaAssemblingKits` | `tasks/franka/assembly/franka_assembling_kits.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `FrankaLiftPegUpright` | `tasks/franka/assembly/franka_lift_peg_upright.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `FrankaPegInsertionSide` | `tasks/franka/assembly/franka_peg_insertion_side.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `FrankaPlugCharger` | `tasks/franka/assembly/franka_plug_charger.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `cabinet` | `FrankaCabinet` | `tasks/franka/cabinet/franka_cabinet.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `cabinet` | `FrankaCabinetBlocks` | `tasks/franka/cabinet/franka_cabinet_blocks.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `cabinet` | `FrankaCabinetYCB` | `tasks/franka/cabinet/franka_cabinet_ycb.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `lift` | `FrankaLift` | `tasks/franka/lift/franka_lift.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `lift` | `FrankaLiftSugarBox` | `tasks/franka/lift/franka_lift_sugar_box.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `peg_insert` | `FrankaFactoryPegInsert` | `tasks/franka/peg_insert/franka_peg_insert.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `FrankaMultiPickPlace` | `tasks/franka/pick_place/franka_multi_pick_place.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `FrankaPickPlace` | `tasks/franka/pick_place/franka_pick_place.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `FrankaPickPlaceBottle` | `tasks/franka/pick_place/franka_pick_place_bottle.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `FrankaPickPlaceBox` | `tasks/franka/pick_place/franka_pick_place_box.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `FrankaPickPlaceCan` | `tasks/franka/pick_place/franka_pick_place_can.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `FrankaPickPlaceDrawer` | `tasks/franka/pick_place/franka_pick_place_drawer.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `FrankaPickPlaceGears` | `tasks/franka/pick_place/franka_pick_place_gears.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `FrankaPickPlaceMug` | `tasks/franka/pick_place/franka_pick_place_mug.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `FrankaPickPlaceTuna` | `tasks/franka/pick_place/franka_pick_place_tuna.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `sort` | `FrankaColorSort` | `tasks/franka/sort/franka_color_sort.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `sort` | `FrankaLineArrange` | `tasks/franka/sort/franka_line_arrange.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `sort` | `FrankaShapeSort` | `tasks/franka/sort/franka_shape_sort.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `stack` | `FrankaStack` | `tasks/franka/stack/franka_stack.yaml` | `pass` | `yes` | `yes` | `yes` | IsaacLab eval 96/100. Latest ADC verified run reached target_met=true and produced raw_dataset. | `outputs/isaaclab/frankastack_20260307_102915/eval_report.json`<br>`outputs/data_collection/FrankaStack_20260307_110013/collection_results.json` |
| `stack` | `FrankaStackTray` | `tasks/franka/stack/franka_stack_tray.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |

### openarm

| Category | Task | YAML | IsaacLab | ADC verified | Dataset | Task success | Notes | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `assembly` | `OpenArmAssemblingKits` | `tasks/openarm/assembly/openarm_assembling_kits.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `OpenArmLiftPegUpright` | `tasks/openarm/assembly/openarm_lift_peg_upright.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `OpenArmPegInsertionSide` | `tasks/openarm/assembly/openarm_peg_insertion_side.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `OpenArmPlugCharger` | `tasks/openarm/assembly/openarm_plug_charger.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `cabinet` | `OpenArmCabinet` | `tasks/openarm/cabinet/openarm_cabinet.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `cabinet` | `OpenArmCabinetBlocks` | `tasks/openarm/cabinet/openarm_cabinet_blocks.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `cabinet` | `OpenArmCabinetYCB` | `tasks/openarm/cabinet/openarm_cabinet_ycb.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `lift` | `OpenArmLift` | `tasks/openarm/lift/openarm_lift.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `lift` | `OpenArmLiftSugarBox` | `tasks/openarm/lift/openarm_lift_sugar_box.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `OpenArmMultiPickPlace` | `tasks/openarm/pick_place/openarm_multi_pick_place.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `OpenArmPickPlace` | `tasks/openarm/pick_place/openarm_pick_place.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `OpenArmPickPlaceBottle` | `tasks/openarm/pick_place/openarm_pick_place_bottle.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `OpenArmPickPlaceBox` | `tasks/openarm/pick_place/openarm_pick_place_box.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `OpenArmPickPlaceCan` | `tasks/openarm/pick_place/openarm_pick_place_can.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `OpenArmPickPlaceDrawer` | `tasks/openarm/pick_place/openarm_pick_place_drawer.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `OpenArmPickPlaceGears` | `tasks/openarm/pick_place/openarm_pick_place_gears.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `OpenArmPickPlaceMug` | `tasks/openarm/pick_place/openarm_pick_place_mug.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `OpenArmPickPlaceTuna` | `tasks/openarm/pick_place/openarm_pick_place_tuna.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `reach` | `OpenArmReach` | `tasks/openarm/reach/openarm_reach.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `sort` | `OpenArmColorSort` | `tasks/openarm/sort/openarm_color_sort.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `sort` | `OpenArmLineArrange` | `tasks/openarm/sort/openarm_line_arrange.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `sort` | `OpenArmShapeSort` | `tasks/openarm/sort/openarm_shape_sort.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `stack` | `OpenArmStack` | `tasks/openarm/stack/openarm_stack.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `stack` | `OpenArmStackTray` | `tasks/openarm/stack/openarm_stack_tray.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |

### so101

| Category | Task | YAML | IsaacLab | ADC verified | Dataset | Task success | Notes | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `assembly` | `SO101AssemblingKits` | `tasks/so101/assembly/so101_assembling_kits.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `SO101LiftPegUpright` | `tasks/so101/assembly/so101_lift_peg_upright.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `SO101PegInsertionSide` | `tasks/so101/assembly/so101_peg_insertion_side.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `SO101PlugCharger` | `tasks/so101/assembly/so101_plug_charger.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `lift` | `SO101Lift` | `tasks/so101/lift/so101_lift.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `SO101MultiPickPlace` | `tasks/so101/pick_place/so101_multi_pick_place.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `SO101PickPlace` | `tasks/so101/pick_place/so101_pick_place.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `SO101PickPlaceCylinder` | `tasks/so101/pick_place/so101_pick_place_cylinder.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `reach` | `SO101Reach` | `tasks/so101/reach/so101_reach.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `sort` | `SO101ColorSort` | `tasks/so101/sort/so101_color_sort.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `sort` | `SO101LineArrange` | `tasks/so101/sort/so101_line_arrange.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `sort` | `SO101ShapeSort` | `tasks/so101/sort/so101_shape_sort.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `stack` | `SO101Stack` | `tasks/so101/stack/so101_stack.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |

### ur10

| Category | Task | YAML | IsaacLab | ADC verified | Dataset | Task success | Notes | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `assembly` | `UR10AssemblingKits` | `tasks/ur10/assembly/ur10_assembling_kits.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `UR10LiftPegUpright` | `tasks/ur10/assembly/ur10_lift_peg_upright.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `UR10PegInsertionSide` | `tasks/ur10/assembly/ur10_peg_insertion_side.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `UR10PlugCharger` | `tasks/ur10/assembly/ur10_plug_charger.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |

### ur10e

| Category | Task | YAML | IsaacLab | ADC verified | Dataset | Task success | Notes | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cabinet` | `UR10eCabinet` | `tasks/ur10e/cabinet/ur10e_cabinet.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `cabinet` | `UR10eCabinetBlocks` | `tasks/ur10e/cabinet/ur10e_cabinet_blocks.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `cabinet` | `UR10eCabinetYCB` | `tasks/ur10e/cabinet/ur10e_cabinet_ycb.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `UR10ePickPlace` | `tasks/ur10e/pick_place/ur10e_pick_place.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `UR10ePickPlaceBottle` | `tasks/ur10e/pick_place/ur10e_pick_place_bottle.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `UR10ePickPlaceBox` | `tasks/ur10e/pick_place/ur10e_pick_place_box.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `UR10ePickPlaceCan` | `tasks/ur10e/pick_place/ur10e_pick_place_can.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `UR10ePickPlaceDrawer` | `tasks/ur10e/pick_place/ur10e_pick_place_drawer.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `UR10ePickPlaceMug` | `tasks/ur10e/pick_place/ur10e_pick_place_mug.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `pick_place` | `UR10ePickPlaceTuna` | `tasks/ur10e/pick_place/ur10e_pick_place_tuna.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `reach` | `UR10eReach` | `tasks/ur10e/reach/ur10e_reach.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `stack` | `UR10eStack` | `tasks/ur10e/stack/ur10e_stack.yaml` | `pass` | `yes` | `yes` | `no` | IsaacLab eval 91/100. ADC verified run produced raw_dataset, but target_met=false (0/3 successful episodes). | `outputs/isaaclab/ur10estack_20260307_102720/eval_report.json`<br>`outputs/data_collection/UR10eStack_20260307_104601/collection_results.json` |
| `stack` | `UR10eStackTray` | `tasks/ur10e/stack/ur10e_stack_tray.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |

