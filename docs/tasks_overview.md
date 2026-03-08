# Task Overview

대상: corpus 상태와 검증 evidence를 확인하는 사용자/기여자  
이 문서가 다루는 것: task별 상태 보드와 실행 evidence  
Task 설명과 추천 shortlist: `docs/task_descriptions_ko.md`

이 문서는 `tasks/` corpus의 **상태 보드 source-of-truth**다. task가 무엇을 하는지는 `docs/task_descriptions_ko.md`, 분류/집계는 `docs/task_taxonomy.md`를 본다.

## Status Legend

| Column | Meaning | Values |
| --- | --- | --- |
| `isaaclab_eval` | `run_isaac_lab.py ... --evaluate` 기준 평가 상태 | `pass`, `fail`, `unknown` |
| `adc_verified` | `run_data_collection.py` 경로를 실제로 실행했는지 | `yes`, `no`, `unknown` |
| `dataset_generated` | `raw_dataset/` 생성까지 확인됐는지 | `yes`, `no`, `unknown` |
| `task_success` | ADC 실행 결과에서 task goal 달성까지 확인됐는지 | `yes`, `no`, `unknown` |

## Summary

- Total tasks: **78**
- Known `isaaclab_eval=pass`: **4**
- Known `adc_verified=yes`: **4**
- Known `dataset_generated=yes`: **4**
- Known `task_success=yes`: **2**

| Robot | Tasks | Categories | IsaacLab pass | ADC verified | Dataset generated | Task success |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `franka` | 24 | assembly (4), cabinet (3), lift (2), peg_insert (1), pick_place (9), sort (3), stack (2) | 1 | 1 | 1 | 1 |
| `openarm` | 24 | assembly (4), cabinet (3), lift (2), pick_place (9), reach (1), sort (3), stack (2) | 1 | 1 | 1 | 1 |
| `so101` | 13 | assembly (4), lift (1), pick_place (3), reach (1), sort (3), stack (1) | 1 | 1 | 1 | 0 |
| `ur10e` | 17 | assembly (4), cabinet (3), pick_place (7), reach (1), stack (2) | 1 | 1 | 1 | 0 |

## 관련 문서

- `docs/task_descriptions_ko.md`
- `docs/task_taxonomy.md`

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
| `stack` | `FrankaStack` | `tasks/franka/stack/franka_stack.yaml` | `pass` | `yes` | `yes` | `yes` | IsaacLab eval 96/100. ADC verified run reached `target_met=true`, produced `raw_dataset/`, and was later reused for export/preprocess verification. | `outputs/isaaclab/frankastack_20260307_102915/eval_report.json`<br>`outputs/data_collection/FrankaStack_20260308_083022/collection_results.json` |
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
| `stack` | `OpenArmStack` | `tasks/openarm/stack/openarm_stack.yaml` | `pass` | `yes` | `yes` | `yes` | Source-level generator fixes let `run_isaac_lab.py --evaluate` pass without manual env edits. Differential IK motion reaches the stack waypoints, and at least one verified ADC run completed the full stack task. A later dataset-schema rerun produced raw data but did not meet the target again. | `outputs/isaaclab/openarmstack_20260307_182545/eval_report.json`<br>`outputs/motion_eval/OpenArmStack_20260307_191640/motion_report.json`<br>`outputs/data_collection/OpenArmStack_20260307_191829/collection_results.json` |
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
| `stack` | `SO101Stack` | `tasks/so101/stack/so101_stack.yaml` | `pass` | `yes` | `yes` | `no` | IsaacLab generation/evaluation, motion sanity, and ADC smoke all run without manual env edits. The remaining blocker is not action delivery: EE tracking is within a few mm at grasp, but the claw still closes without lifting the cube, so the failure is in grasp/contact geometry. | `outputs/isaaclab/so101stack_20260307_190110/eval_report.json`<br>`outputs/motion_eval/SO101Stack_20260307_193751/motion_report.json`<br>`outputs/data_collection/SO101Stack_20260308_083356/collection_results.json` |

### ur10e

| Category | Task | YAML | IsaacLab | ADC verified | Dataset | Task success | Notes | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `assembly` | `UR10eAssemblingKits` | `tasks/ur10e/assembly/ur10e_assembling_kits.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `UR10eLiftPegUpright` | `tasks/ur10e/assembly/ur10e_lift_peg_upright.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `UR10ePegInsertionSide` | `tasks/ur10e/assembly/ur10e_peg_insertion_side.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
| `assembly` | `UR10ePlugCharger` | `tasks/ur10e/assembly/ur10e_plug_charger.yaml` | `unknown` | `unknown` | `unknown` | `unknown` | - | - |
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
