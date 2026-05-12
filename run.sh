#!/usr/bin/env bash


# Usage:
#   sbatch scripts/run.sh <settings.json> [extra_python_args...]
#
# The script reads the "sweep" block of the settings JSON and creates one
# SLURM array task per combination of (n_qubits × depth × engine).
#
# Required tools on the compute nodes:
#   - python3 (with csbench installed in the active venv)
#   - jq       (for JSON parsing in bash)


#SBATCH --job-name=csbench
#SBATCH --output=logs/csbench_%A_%a.out   # %A = job id, %a = array task id
#SBATCH --error=logs/csbench_%A_%a.err
#SBATCH --time=02:00:00                   # ⚠ wall-clock limit per task
#SBATCH --mem=8G                          # ⚠ memory per task
#SBATCH --cpus-per-task=4                 # ⚠ CPUs per task

set -euo pipefail

# Parse arguments
SETTINGS="${1:?Usage: sbatch run.sh <settings.json>}"
shift                           # remaining args forwarded to main.py
EXTRA_ARGS=("$@")

if [[ ! -f "$SETTINGS" ]]; then
    echo "ERROR: settings file not found: $SETTINGS" >&2
    exit 1
fi

# Read sweep parameters from JSON
if ! command -v jq &>/dev/null; then
    echo "ERROR: 'jq' is required but not found in PATH." >&2
    exit 1
fi
mapfile -t N_QUBITS_LIST < <(jq -r '.sweep.n_qubits[]' "$SETTINGS")
mapfile -t DEPTH_LIST     < <(jq -r '.sweep.depth[]'    "$SETTINGS")
mapfile -t ENGINE_LIST    < <(jq -r '.execution.engines[]' "$SETTINGS")

N_Q=${#N_QUBITS_LIST[@]}
N_D=${#DEPTH_LIST[@]}
N_E=${#ENGINE_LIST[@]}
TOTAL=$(( N_Q * N_D * N_E ))
if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: sweep produces zero tasks.  Check sweep.n_qubits, sweep.depth, and execution.engines in $SETTINGS." >&2
    exit 1
fi
echo "Sweep dimensions: n_qubits=${N_Q} × depth=${N_D} × engines=${N_E} = ${TOTAL} tasks"


# If we are NOT inside a SLURM array job, re-submit
# (i.e. the user ran:  bash run.sh settings.json  instead of  sbatch run.sh …)
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # Create log directory
    mkdir -p logs

    # Re-submit ourselves as a SLURM array job
    sbatch --array="0-$(( TOTAL - 1 ))" "$0" "$SETTINGS" "${EXTRA_ARGS[@]}"
    exit 0
fi

# ---------- 4. We are inside a task: compute our (n_qubits, depth, engine) --
TASK_ID="${SLURM_ARRAY_TASK_ID}"

# Map flat task index → (engine_idx, depth_idx, n_qubits_idx)
# Order: n_qubits is the fastest-varying index so results for the same
# (engine, depth) are contiguous.
E_IDX=$(( TASK_ID / (N_Q * N_D) ))
REMAINDER=$(( TASK_ID % (N_Q * N_D) ))
D_IDX=$(( REMAINDER / N_Q ))
Q_IDX=$(( REMAINDER % N_Q ))

N_QUBITS="${N_QUBITS_LIST[$Q_IDX]}"
DEPTH="${DEPTH_LIST[$D_IDX]}"
ENGINE="${ENGINE_LIST[$E_IDX]}"

echo "Task ${TASK_ID}: n_qubits=${N_QUBITS}, depth=${DEPTH}, engine=${ENGINE}"


module load python/3.11 cuda/12.0

VENV_PATH="${VENV_PATH:-$HOME/.venvs/csbench}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    source "${VENV_PATH}/bin/activate"
fi


OUTPUT_DIR="$(jq -r '.output.filename | split("/")[:-1] | join("/")' "$SETTINGS")"
OUTPUT_FMT="$(jq -r '.output.format // "jsonl"' "$SETTINGS")"
OUTPUT_BASE="$(jq -r '.output.filename | split("/")[-1] | split(".")[0]' "$SETTINGS")"
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_BASE}.${OUTPUT_FMT}"
mkdir -p "${OUTPUT_DIR}"


python main.py \
    --settings  "${SETTINGS}"   \
    --n_qubits  "${N_QUBITS}"   \
    --depth     "${DEPTH}"      \
    --engine    "${ENGINE}"     \
    --run_id    "${TASK_ID}"    \
    --output    "${OUTPUT_FILE}" \
    "${EXTRA_ARGS[@]}"

echo "Task ${TASK_ID} finished successfully."
