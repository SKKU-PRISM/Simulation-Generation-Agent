#!/usr/bin/env bash
set -euo pipefail

# Edit this list for the task batch you want to run.
TASKS=(
  "tasks/franka/stack/franka_stack.yaml"
  "tasks/franka/assembly/franka_lift_peg_upright.yaml"
)

# Common collection settings.
EPISODES=5
TARGET_SUCCESS=0
MAX_ATTEMPTS=0
CONFIG_PATH=""
OUTPUT_ROOT=""

# LeRobot / Hugging Face settings.
LOCAL_REPO_PREFIX="local/multitask"
HF_REPO_PREFIX="org/multitask"
HF_PRIVATE=1
UPLOAD_TO_HF=0
TOKEN_ENV="HF_TOKEN"

# Batch behavior.
CONTINUE_ON_FAILURE=1
NO_VLM_JUDGE=0
GUI=0
VERBOSE=0

cmd=(python3 scripts/run_multitask_to_hf.py "${TASKS[@]}" --local-repo-prefix "${LOCAL_REPO_PREFIX}" --token-env "${TOKEN_ENV}")

if [[ -n "${CONFIG_PATH}" ]]; then
  cmd+=(--config "${CONFIG_PATH}")
fi
if [[ -n "${OUTPUT_ROOT}" ]]; then
  cmd+=(--output-root "${OUTPUT_ROOT}")
fi
if (( EPISODES > 0 )); then
  cmd+=(--episodes "${EPISODES}")
fi
if (( TARGET_SUCCESS > 0 )); then
  cmd+=(--target-success "${TARGET_SUCCESS}")
fi
if (( MAX_ATTEMPTS > 0 )); then
  cmd+=(--max-attempts "${MAX_ATTEMPTS}")
fi
if (( UPLOAD_TO_HF == 1 )); then
  cmd+=(--upload-to-hf --hf-repo-prefix "${HF_REPO_PREFIX}")
fi
if (( HF_PRIVATE == 1 )); then
  cmd+=(--private)
fi
if (( CONTINUE_ON_FAILURE == 0 )); then
  cmd+=(--fail-fast)
fi
if (( NO_VLM_JUDGE == 1 )); then
  cmd+=(--no-vlm-judge)
fi
if (( GUI == 1 )); then
  cmd+=(--gui)
fi
if (( VERBOSE == 1 )); then
  cmd+=(--verbose)
fi

echo "Running multi-task batch with ${#TASKS[@]} task(s)"
printf '  %s\n' "${TASKS[@]}"

"${cmd[@]}"
