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

print_help() {
  cat <<'EOF'
Usage:
  run_agent.sh <input_file> [output_file]        Challenge submission mode
  run_agent.sh [--mode <mode>] [options] [-- extra args]   Advanced pipeline mode

Challenge submission mode:
  input_file               JSON task spec (default: ./data/input_sample.json)
  output_file              Result JSON path (default: ./results/output.json)

Advanced pipeline options:
  --mode <mode>            e2e-batch | isaac-lab | data-collection (default: e2e-batch)
  --config <path>          Config file override
  --task <yaml>            Task YAML path for isaac-lab or data-collection mode
  --output-root <dir>      Artifact root (default: /workspace/artifacts)
  --resume                 Resume a previous e2e batch
  -h, --help               Show this help

Examples:
  # Challenge submission (reviewer)
  run_agent.sh data/input_sample.json results/output.json

  # Advanced modes
  run_agent.sh --mode e2e-batch --resume
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

# --- Challenge submission mode: positional args without --mode ---
if [[ ${#POSITIONAL[@]} -gt 0 && ${MODE_EXPLICIT} -eq 0 ]]; then
  INPUT_FILE="${POSITIONAL[0]:-./data/input_sample.json}"
  OUTPUT_FILE="${POSITIONAL[1]:-./results/output.json}"

  # API key check (dual)
  if [[ -z "${OPENAI_API_KEY:-}" && -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
    echo "Either OPENAI_API_KEY or AZURE_OPENAI_API_KEY must be set." >&2
    exit 2
  fi

  mkdir -p "$(dirname "${OUTPUT_FILE}")"
  export PYTHONPATH="${REPO_ROOT}"

  echo "Challenge mode: input=${INPUT_FILE} output=${OUTPUT_FILE}"
  exec "${PYTHON_BIN}" src/main.py --input "${INPUT_FILE}" --output "${OUTPUT_FILE}"
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
