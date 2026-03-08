# 문서 안내

이 디렉터리의 문서는 역할별로 나뉘어 있습니다. 먼저 `getting_started`로 시작하고, 실제 작업 단계에 따라 `usage`, `dataset_alignment_and_export`, `troubleshooting`으로 이동하면 됩니다.

## 빠른 경로

| 문서 | 대상 | 역할 | 다음으로 볼 문서 |
| --- | --- | --- | --- |
| `docs/getting_started.md` | 처음 설치하는 사용자 | 설치, 환경변수, 첫 성공 실행 | `docs/usage.md` |
| `docs/usage.md` | 실제로 돌리는 사용자 | CLI 사용법, 출력 구조, 결과 해석 | `docs/evaluation.md`, `docs/dataset_alignment_and_export.md` |
| `docs/dataset_alignment_and_export.md` | dataset/export/preprocess를 다루는 사용자 | raw / export schema / preprocess 기준 | `docs/troubleshooting.md` |
| `docs/troubleshooting.md` | 실행 중 문제를 푸는 사용자 | 자주 나는 실패 패턴과 우선 점검 경로 | 관련 source-of-truth 문서로 역링크 |

## 참조 문서

| 문서 | 역할 |
| --- | --- |
| `docs/evaluation.md` | IsaacLab evaluator 구조와 점수 해석 |
| `docs/task_yaml_spec.md` | task YAML 작성/수정 규약 |
| `docs/tasks_overview.md` | task 상태 보드와 검증 evidence |
| `docs/task_descriptions_ko.md` | task 한글 설명과 추천 shortlist |
| `docs/task_taxonomy.md` | corpus 분류/집계표 |
| `docs/isaaclab_env_config_guide.md` | IsaacLab env config 심화 가이드 |

## 권장 읽기 순서

### 1. 처음 세팅할 때

1. `docs/getting_started.md`
2. `docs/usage.md`
3. 필요 시 `docs/troubleshooting.md`

### 2. 데이터 수집/학습 준비까지 갈 때

1. `docs/usage.md`
2. `docs/dataset_alignment_and_export.md`
3. `docs/troubleshooting.md`

### 3. task를 추가하거나 corpus를 볼 때

1. `docs/task_yaml_spec.md`
2. `docs/tasks_overview.md`
3. `docs/task_descriptions_ko.md`
4. `docs/task_taxonomy.md`

### 4. evaluator나 env config를 수정할 때

1. `docs/evaluation.md`
2. `docs/isaaclab_env_config_guide.md`
