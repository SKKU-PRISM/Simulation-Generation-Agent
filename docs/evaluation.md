# 평가 시스템

이 문서는 **IsaacLab Pipeline**의 자동 평가기를 설명합니다. 이 레포의 주력 경로는 IsaacLab이며, 평가는 생성된 `env_cfg.py`가 YAML task를 얼마나 정확하게 구현했는지 점수화합니다.

## 실행 방법

```bash
# 생성 + 실행 + 평가
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate

# 기존 출력물 평가
python3 scripts/evaluate.py \
  outputs/isaaclab/frankastack_20260219_160916 \
  tasks/franka/stack/franka_stack.yaml

# 정적 분석만
python3 scripts/evaluate.py --skip-runtime \
  outputs/isaaclab/frankastack_20260219_160916 \
  tasks/franka/stack/franka_stack.yaml
```

## 평가 구조

평가는 총 100점이며 4개 카테고리로 구성됩니다.

| 카테고리 | 배점 | 실제 체크 |
|----------|:----:|----------|
| `scene_fidelity` | 30 | `asset_completeness`, `asset_config`, `physics_config`, `robot_config`, `scene_structure` |
| `mdp_correctness` | 25 | `observation_coverage`, `observation_validity`, `action_space`, `reward_structure`, `termination_coverage`, `event_coverage` |
| `task_alignment` | 25 | `goal_condition_mapping`, `threshold_preservation`, `custom_mdp_validity` |
| `runtime_validity` | 20 | `env_creation`, `reset_step_cycle`, `reward_computation`, `physics_stability` |

## Phase 1: 정적 분석

정적 분석은 IsaacLab 런타임 없이 동작합니다.

- YAML `goal`은 `GoalNormalizer`가 정규화합니다.
- `env_cfg.py`는 `EnvCfgParser`가 AST + regex로 분석합니다.
- `InteractiveSceneCfg`의 class-level 엔티티뿐 아니라 `ManagerBasedRLEnvCfg.__post_init__`의 `self.scene.<name> = ...` 할당도 파싱합니다.
- `mdp/` 디렉토리의 custom function도 읽어서 goal mapping과 import 유효성을 검사합니다.

## Phase 2: 런타임 분석

런타임 분석은 출력 디렉토리에 `eval_runner.py`를 동적으로 생성한 뒤 `isaaclab.sh`로 실행합니다.

검사 내용:

- 환경 인스턴스화 성공 여부
- reset/step loop 수행 여부
- observation range와 NaN/Inf 여부
- reward 통계
- rigid object 위치 안정성

생성 파일:

- `eval_runner.py`
- `eval_results.json`

실행은 `configs/isaaclab_eval_config.yaml`의 `runtime` 및 `isaaclab` 설정을 사용합니다.

## 현재 점수 정책

### Scene Fidelity (30)

- `asset_completeness` 10점: YAML asset가 scene entity로 대응되는지
- `asset_config` 8점: position / USD path 정합성
- `physics_config` 5점: `timestep`, `decimation`, `episode_length`
- `robot_config` 4점: 적절한 robot cfg 사용 여부
- `scene_structure` 3점: ground / light / env_spacing

### MDP Correctness (25)

- `observation_coverage` 7점
- `observation_validity` 3점
- `action_space` 5점
- `reward_structure` 5점
- `termination_coverage` 3점
- `event_coverage` 2점

특이 정책:

- `rewards=None`이고 `allow_none_rewards: true`면 `reward_structure`는 실패가 아니라 **validation-only setup**으로 `3/5 WARN`
- 런타임 reward가 전부 0이면 `reward_computation`은 `3/5 WARN`
- observation에 NaN/Inf가 나오면 `observation_validity`가 런타임 결과로 추가 감점될 수 있음

### Task Alignment (25)

- `goal_condition_mapping` 10점
- `threshold_preservation` 8점
- `custom_mdp_validity` 7점

`goal.success_criteria`와 `goal.conditions` 모두 정규화 후 동일한 방식으로 검사합니다.

### Runtime Validity (20)

- `env_creation` 5점
- `reset_step_cycle` 5점
- `reward_computation` 5점
- `physics_stability` 5점

## 출력 파일 구조

대표 결과 파일은 `eval_report.json`입니다.

```json
{
  "total_score": 94,
  "checklist": [
    {
      "category": "scene_fidelity",
      "check": "asset_completeness",
      "status": "PASS",
      "score": 10,
      "max": 10,
      "details": "5/5 assets found"
    }
  ],
  "breakdown": {
    "scene_fidelity": { "score": 30, "max": 30 },
    "mdp_correctness": { "score": 23, "max": 25 },
    "task_alignment": { "score": 23, "max": 25 },
    "runtime_validity": { "score": 18, "max": 20 }
  },
  "issues": []
}
```

## 종료 코드

`scripts/evaluate.py`는 총점 기준으로 종료 코드를 반환합니다.

- `0`: 70점 이상
- `1`: 70점 미만

`--skip-runtime` 사용 시 런타임 카테고리는 0점으로 처리됩니다.

## 설정 포인트

`configs/isaaclab_eval_config.yaml`에서 조정할 수 있는 대표 항목:

- `evaluation.position_tolerance`
- `evaluation.rotation_tolerance`
- `evaluation.scale_tolerance`
- `evaluation.allow_none_rewards`
- `runtime.eval_steps`
- `runtime.num_envs`
- `runtime.timeout`
- `runtime.position_bound`
- `isaaclab.path`
- `isaaclab.conda_env`

## 해석 팁

- `PASS`: 해당 체크가 만점
- `WARN`: 부분 점수 또는 허용 가능한 불완전성
- `FAIL`: 0점

IsaacLab 생성 코드를 볼 때는 먼저 `breakdown`, 그다음 `checklist`, 마지막으로 `issues` 순서로 해석하는 것이 빠릅니다.

## 관련 문서

- `docs/usage.md`
- `docs/task_yaml_spec.md`
- `docs/troubleshooting.md`
