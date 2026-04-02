#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/opt/isaaclab-env/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

DEFAULT_MODE="e2e-batch"
DEFAULT_OUTPUT_ROOT="${SIMGEN_ARTIFACT_ROOT:-/workspace/artifacts}"
DEFAULT_E2E_CONFIG="configs/docker/e2e_batch_release.yaml"
DEFAULT_DC_CONFIG="configs/docker/data_collection_release.yaml"
DEFAULT_IL_CONFIG="configs/docker/isaaclab_agent_release.yaml"

MODE="${DEFAULT_MODE}"
MODE_EXPLICIT=0
CONFIG=""
TASK=""
OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}"
RESUME=0
POSITIONAL=()
PASSTHROUGH=()

ROBOT=""
EPISODES=""
MAX_ATTEMPTS_NL=""

print_help() {
  cat <<'EOF'
Usage:
  run_agent.sh "자연어 태스크 설명" [options]                  자연어 → 전체 파이프라인 (Stage 1→2→3)
  run_agent.sh <input_file.json> [output_file]               JSON 입력 → 전체 파이프라인
  run_agent.sh --mode <mode> [options] [-- extra args]       개별 Stage 실행

자연어 입력 (메인 기능):
  첫 번째 인자로 자연어 태스크 설명을 넣으면 NL→YAML→IsaacLab→DataCollection 전체 파이프라인이 실행됩니다.
  --robot <type>         로봇 종류: franka, ur10e, openarm, so101 (기본: franka)
  --episodes <n>         목표 성공 에피소드 수 (기본: 1)
  --max-attempts <n>     최대 시도 횟수 (기본: 3)

JSON 입력:
  input_file               JSON task spec (default: ./data/input_sample.json)
  output_file              Result JSON path (default: ./results/output.json)

개별 Stage (고급):
  --mode <mode>            e2e-batch | isaac-lab | data-collection (default: e2e-batch)
  --config <path>          Config file override
  --task <yaml>            Task YAML path for isaac-lab or data-collection mode
  --output-root <dir>      Artifact root (default: /workspace/artifacts)
  --resume                 Resume a previous e2e batch
  -h, --help               Show this help

Examples:
  # 자연어 입력 → 전체 파이프라인 (가장 간단한 사용법)
  run_agent.sh "Stack the blocks inside the tray on the table"
  run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5

  # JSON 입력
  run_agent.sh data/input_sample.json results/output.json

  # 개별 Stage 실행
  run_agent.sh --mode isaac-lab --task tasks/franka/lift/franka_lift.yaml -- --dry-run
  run_agent.sh --mode data-collection --task tasks/franka/lift/franka_lift.yaml -- --episodes 2

Required environment variables (at least one group):
  OPENAI_API_KEY                          OpenAI platform
  AZURE_OPENAI_API_KEY + AZURE_OPENAI_BASE_URL   Azure OpenAI
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      MODE_EXPLICIT=1
      shift 2
      ;;
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --task)
      TASK="${2:-}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --robot)
      ROBOT="${2:-franka}"
      shift 2
      ;;
    --episodes)
      EPISODES="${2:-1}"
      shift 2
      ;;
    --max-attempts)
      MAX_ATTEMPTS_NL="${2:-3}"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    --)
      shift
      PASSTHROUGH=("$@")
      break
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

# --- Positional args without --mode: NL string or JSON file ---
if [[ ${#POSITIONAL[@]} -gt 0 && ${MODE_EXPLICIT} -eq 0 ]]; then
  FIRST_ARG="${POSITIONAL[0]}"

  # API key check (dual)
  if [[ -z "${OPENAI_API_KEY:-}" && -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
    echo "Either OPENAI_API_KEY or AZURE_OPENAI_API_KEY must be set." >&2
    exit 2
  fi

  export PYTHONPATH="${REPO_ROOT}"

  # Detect: .json → challenge mode, otherwise → NL full-pipeline mode
  if [[ "${FIRST_ARG}" == *.json ]]; then
    # --- JSON file input (challenge submission) ---
    INPUT_FILE="${FIRST_ARG}"
    OUTPUT_FILE="${POSITIONAL[1]:-./results/output.json}"
    mkdir -p "$(dirname "${OUTPUT_FILE}")"

    echo "Challenge mode: input=${INPUT_FILE} output=${OUTPUT_FILE}"
    exec "${PYTHON_BIN}" src/main.py --input "${INPUT_FILE}" --output "${OUTPUT_FILE}"
  else
    # --- Natural language input → full pipeline (Stage 1→2→3) ---
    TASK_DESC="${FIRST_ARG}"
    ROBOT="${ROBOT:-franka}"
    EPISODES="${EPISODES:-1}"
    MAX_ATTEMPTS_NL="${MAX_ATTEMPTS_NL:-3}"
    OUTPUT_FILE="${POSITIONAL[1]:-./results/output.json}"

    INPUT_JSON=$(mktemp /tmp/simgen_input_XXXXXX.json)
    cat > "${INPUT_JSON}" <<JSONEOF
{
  "tasks": [{"task_description": "${TASK_DESC}", "robot": "${ROBOT}"}],
  "config": {"target_success": ${EPISODES}, "max_attempts": ${MAX_ATTEMPTS_NL}}
}
JSONEOF

    mkdir -p "$(dirname "${OUTPUT_FILE}")"
    echo "Full pipeline mode: task=\"${TASK_DESC}\" robot=${ROBOT} episodes=${EPISODES}"
    echo "  Input JSON: ${INPUT_JSON}"
    echo "  Output: ${OUTPUT_FILE}"
    exec "${PYTHON_BIN}" src/main.py --input "${INPUT_JSON}" --output "${OUTPUT_FILE}"
  fi
fi

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  echo "Unexpected positional arguments with --mode flag: ${POSITIONAL[*]}" >&2
  print_help >&2
  exit 2
fi

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
}

resolve_path() {
  local raw="$1"
  if [[ "${raw}" = /* ]]; then
    printf '%s\n' "${raw}"
  else
    printf '%s\n' "${REPO_ROOT}/${raw}"
  fi
}

MODE="$(printf '%s' "${MODE}" | tr '[:upper:]' '[:lower:]')"
case "${MODE}" in
  e2e-batch|isaac-lab|data-collection)
    ;;
  isaac-sim|mcp)
    echo "Mode '${MODE}' is out of scope for the release container. Supported modes: e2e-batch, isaac-lab, data-collection." >&2
    exit 2
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    print_help >&2
    exit 2
    ;;
  esac

# At least one API key provider must be set
if [[ -z "${OPENAI_API_KEY:-}" && -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
  echo "Either OPENAI_API_KEY or AZURE_OPENAI_API_KEY must be set." >&2
  echo "  -e OPENAI_API_KEY=sk-...  (OpenAI platform)" >&2
  echo "  -e AZURE_OPENAI_API_KEY=... -e AZURE_OPENAI_BASE_URL=...  (Azure)" >&2
  exit 2
fi

OUTPUT_ROOT="$(resolve_path "${OUTPUT_ROOT}")"
mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export SIMGEN_ISAACLAB_PYTHON="${PYTHON_BIN}"

case "${MODE}" in
  e2e-batch)
    CONFIG="${CONFIG:-${DEFAULT_E2E_CONFIG}}"
    E2E_STEM="$(basename "${CONFIG}")"
    E2E_STEM="${E2E_STEM%.yaml}"
    E2E_STEM="${E2E_STEM%.yml}"
    E2E_LEAF="${E2E_STEM#e2e_batch_}"
    if [[ -z "${E2E_LEAF}" || "${E2E_LEAF}" == "${E2E_STEM}" ]]; then
      E2E_LEAF="${E2E_STEM}"
    fi
    CMD=("${PYTHON_BIN}" "scripts/run_e2e_batch.py" "$(resolve_path "${CONFIG}")" "--output-root" "${OUTPUT_ROOT}/e2e_batches/${E2E_LEAF}")
    if [[ ${RESUME} -eq 1 ]]; then
      CMD+=("--resume")
    fi
    ;;
  isaac-lab)
    if [[ -z "${TASK}" ]]; then
      echo "--task is required for isaac-lab mode" >&2
      exit 2
    fi
    CONFIG="${CONFIG:-${DEFAULT_IL_CONFIG}}"
    CMD=("${PYTHON_BIN}" "scripts/run_isaac_lab.py" "$(resolve_path "${TASK}")" "--config" "$(resolve_path "${CONFIG}")" "--output-dir" "${OUTPUT_ROOT}/isaaclab")
    ;;
  data-collection)
    if [[ -z "${TASK}" ]]; then
      echo "--task is required for data-collection mode" >&2
      exit 2
    fi
    CONFIG="${CONFIG:-${DEFAULT_DC_CONFIG}}"
    CMD=("${PYTHON_BIN}" "scripts/run_data_collection.py" "$(resolve_path "${TASK}")" "--config" "$(resolve_path "${CONFIG}")" "--output-dir" "${OUTPUT_ROOT}/data_collection")
    ;;
esac

if [[ ${#PASSTHROUGH[@]} -gt 0 ]]; then
  CMD+=("${PASSTHROUGH[@]}")
fi

echo "Executing: ${CMD[*]}"
exec "${CMD[@]}"
