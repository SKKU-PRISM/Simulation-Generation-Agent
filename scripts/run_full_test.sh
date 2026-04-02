#!/bin/bash
# E2E Full Pipeline Test: NL → YAML → IsaacLab → CAP → Execution → Video
# Usage: bash scripts/run_full_test.sh

set -a; source .env 2>/dev/null; set +a
export PYTHONPATH=.
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUTPUT_ROOT="outputs/test_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_ROOT"
export TOKEN_USAGE_FILE="$OUTPUT_ROOT/token_usage.jsonl"
export TOKEN_USAGE_LOG=1

SUMMARY_FILE="$OUTPUT_ROOT/summary.txt"
echo "=== E2E Full Pipeline Test ===" > "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Task definitions: "TaskName|NL Input"
TASKS=(
  "FrankaLift|Grab the small cube on the table and raise it up"
  "FrankaLiftSugarBox|Pick up the sugar box and hold it in the air"
  "FrankaPickPlace|Take the object and move it to the green target area"
  "FrankaPickPlaceBottle|Grasp the bottle and set it down at the marked spot"
  "FrankaPickPlaceBox|Pick up the sugar box and place it at the green target pad on the table"
  "FrankaPickPlaceCan|Pick up the tomato soup can and place it at the green target pad on the table"
  "FrankaPickPlaceDrawer|Pick up the cube from the table and place it on top of the small drawer box"
  "FrankaPickPlaceTuna|Grab the tuna can and put it at the goal position"
  "FrankaStack|Pick up one cube and stack it on top of the other cube"
  "FrankaMultiPickPlace|Pick up 3 scattered colored blocks and place them onto the gold collection zone"
  "FrankaColorSort|Sort the colored blocks into matching colored bins"
  "FrankaLineArrange|Arrange the colored blocks in a straight line on the table"
  "FrankaStackTray|Stack the blocks inside the tray on the table"
)

run_task() {
  local TASK_NAME="$1"
  local NL_INPUT="$2"
  local TASK_DIR="$OUTPUT_ROOT/$TASK_NAME"
  local YAML_PATH="$TASK_DIR/task.yaml"
  mkdir -p "$TASK_DIR"

  echo "============================================================"
  echo "[$(date +%H:%M:%S)] Task: $TASK_NAME"
  echo "  NL: $NL_INPUT"
  echo "============================================================"

  # Step 1: NL → YAML
  echo "[$(date +%H:%M:%S)] Step 1: NL → YAML..."
  $PYTHON_BIN scripts/task_spec_agent/task_spec_agent.py "$NL_INPUT" \
    --robot franka --output "$YAML_PATH" 2>&1 | tee "$TASK_DIR/step1_nl_to_yaml.log"

  if [ ! -f "$YAML_PATH" ]; then
    echo "  FAILED: YAML not generated"
    echo "$TASK_NAME | FAILED | Step 1: YAML generation failed" >> "$SUMMARY_FILE"
    return 1
  fi
  echo "  OK: $YAML_PATH"

  # Step 2: YAML → IsaacLab environment
  echo "[$(date +%H:%M:%S)] Step 2: YAML → IsaacLab environment..."
  $PYTHON_BIN scripts/run_isaac_lab.py "$YAML_PATH" 2>&1 | tee "$TASK_DIR/step2_isaaclab.log"

  # Find the latest result.json
  local LATEST_RESULT=$(ls -t outputs/isaaclab/*/result.json 2>/dev/null | head -1)
  if [ -z "$LATEST_RESULT" ]; then
    echo "  FAILED: No result.json found"
    echo "$TASK_NAME | FAILED | Step 2: IsaacLab env generation failed" >> "$SUMMARY_FILE"
    return 1
  fi

  local ENV_DIR=$($PYTHON_BIN -c "import json; print(json.load(open('$LATEST_RESULT')).get('output_dir',''))")
  local LAB_SUCCESS=$($PYTHON_BIN -c "import json; print(json.load(open('$LATEST_RESULT')).get('success', False))")

  if [ "$LAB_SUCCESS" != "True" ]; then
    echo "  FAILED: IsaacLab execution failed"
    echo "$TASK_NAME | FAILED | Step 2: IsaacLab execution failed" >> "$SUMMARY_FILE"
    return 1
  fi
  echo "  OK: $ENV_DIR"

  # Step 3: CAP → Execution → Video
  echo "[$(date +%H:%M:%S)] Step 3: CAP + Execution + Video..."
  $PYTHON_BIN scripts/run_data_collection.py "$YAML_PATH" \
    --env-dir "$ENV_DIR" \
    --gui \
    --target-success 1 \
    --max-attempts 3 2>&1 | tee "$TASK_DIR/step3_cap_execution.log"

  # Find the latest collection results
  local LATEST_COLLECTION=$(ls -td outputs/data_collection/${TASK_NAME}* 2>/dev/null | head -1)
  if [ -z "$LATEST_COLLECTION" ]; then
    echo "  FAILED: No collection output found"
    echo "$TASK_NAME | FAILED | Step 3: CAP execution failed" >> "$SUMMARY_FILE"
    return 1
  fi

  local VIDEO_PATH="$LATEST_COLLECTION/videos/front_success.mp4"
  local RESULTS_JSON="$LATEST_COLLECTION/collection_results.json"

  if [ -f "$RESULTS_JSON" ]; then
    local GEOM=$($PYTHON_BIN -c "import json; r=json.load(open('$RESULTS_JSON')); print(r.get('geometry_successful_episodes',0))")
    local VLM=$($PYTHON_BIN -c "import json; r=json.load(open('$RESULTS_JSON')); print(r.get('vlm_successful_episodes',0))")
    local TOTAL=$($PYTHON_BIN -c "import json; r=json.load(open('$RESULTS_JSON')); print(r.get('total_episodes',0))")
    echo "  Results: geometry=$GEOM vlm=$VLM total=$TOTAL"

    local VIDEO_STATUS="NO_VIDEO"
    if [ -f "$VIDEO_PATH" ]; then
      VIDEO_STATUS="$VIDEO_PATH"
    fi
    echo "$TASK_NAME | geom=$GEOM vlm=$VLM total=$TOTAL | $VIDEO_STATUS" >> "$SUMMARY_FILE"
  else
    echo "$TASK_NAME | FAILED | Step 3: No results" >> "$SUMMARY_FILE"
  fi

  echo "[$(date +%H:%M:%S)] Done: $TASK_NAME"
  echo ""
}

# Run tasks
for ENTRY in "${TASKS[@]}"; do
  IFS='|' read -r TASK_NAME NL_INPUT <<< "$ENTRY"
  run_task "$TASK_NAME" "$NL_INPUT" || true
done

echo ""
echo "============================================================"
echo "  ALL TASKS COMPLETE"
echo "  Summary: $OUTPUT_ROOT/summary.txt"
echo "============================================================"
cat "$SUMMARY_FILE"

echo ""
echo "============================================================"
echo "  TOKEN USAGE REPORT"
echo "============================================================"
if [ -f "$TOKEN_USAGE_FILE" ]; then
  $PYTHON_BIN -c "
from src.agent.common.token_tracker import TokenTracker
print(TokenTracker.report_from_file('$TOKEN_USAGE_FILE'))
"
else
  echo "  No token usage data collected."
fi
