#!/bin/bash
#SBATCH -J "iris_pilot"
#SBATCH -o slurm/logs/iris_sls_pilot_%x_%j.out
#SBATCH -e slurm/logs/iris_sls_pilot_%x_%j.err
#SBATCH -p ar_h200
#SBATCH --gres=gpu:h200:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300

set -euo pipefail

# Usage:
#   sbatch slurm/iris_sls_pilot.sl ce  PongNoFrameskip-v4 0 35 0.1 16 bf16 true reduce-overhead
#   sbatch slurm/iris_sls_pilot.sl sls PongNoFrameskip-v4 0 35 0.1 16 bf16 true reduce-overhead
#   sbatch slurm/iris_sls_pilot.sl sls_anneal PongNoFrameskip-v4 0 600 0.1 16 bf16 true reduce-overhead

CONDITION=${1:-ce}
GAME=${2:-PongNoFrameskip-v4}
SEED=${3:-0}
EPOCHS=${4:-35}
SLS_SMOOTHING=${5:-0.1}
SLS_TOPK=${6:-16}
PRECISION=${7:-bf16}
COMPILE=${8:-true}
COMPILE_MODE=${9:-reduce-overhead}

if [[ "$CONDITION" != "ce" && "$CONDITION" != "sls" && "$CONDITION" != "sls_anneal" ]]; then
    echo "condition must be 'ce', 'sls', or 'sls_anneal', got: $CONDITION" >&2
    exit 2
fi

cd /home/2500001/ftari001/iris-sls-baseline
mkdir -p slurm/logs

export PYTHONWARNINGS=ignore
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export CPATH="/home/2500001/ftari001/opt/python-3.10.20/include/python3.10:/home/2500001/ftari001/include/python3.9:${CPATH:-}"

if [[ -z "${TORCHINDUCTOR_CACHE_DIR:-}" ]]; then
    export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor_${SLURM_JOB_ID}"
    CLEAN_TORCHINDUCTOR_CACHE=1
fi
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
trap '[[ "${CLEAN_TORCHINDUCTOR_CACHE:-0}" == "1" ]] && rm -rf "${TORCHINDUCTOR_CACHE_DIR}"' EXIT

PYTHON=${PYTHON:-/home/2500001/ftari001/venvs/iris-sls-torch212/bin/python}
COMPILE_TAG="eager"
if [[ "$COMPILE" == "true" || "$COMPILE" == "True" ]]; then
    COMPILE_TAG="compile_${COMPILE_MODE//-/_}"
fi
RUN_NAME="iris_${CONDITION}_${GAME}_seed${SEED}_e${EPOCHS}_${PRECISION}_${COMPILE_TAG}"
RUN_DIR="experiments/${RUN_NAME}"
RESUBMIT_FLAG="${RUN_DIR}/.resubmit_requested"

SLS_ARGS=(
    "training.world_model.sls_smoothing=0.0"
    "training.world_model.sls_topk=0"
)

if [[ "$CONDITION" == "sls" || "$CONDITION" == "sls_anneal" ]]; then
    SLS_ARGS=(
        "training.world_model.sls_smoothing=${SLS_SMOOTHING}"
        "training.world_model.sls_kernel=gaussian"
        "training.world_model.sls_sigma=null"
        "training.world_model.sls_topk=${SLS_TOPK}"
    )
    if [[ "$CONDITION" == "sls_anneal" ]]; then
        SLS_ARGS+=(
            "training.world_model.sls_schedule.enabled=true"
            "training.world_model.sls_schedule.kind=cosine"
            "training.world_model.sls_schedule.start_epoch=250"
            "training.world_model.sls_schedule.end_epoch=450"
            "training.world_model.sls_schedule.final_smoothing=0.0"
        )
    fi
fi

echo "=== IRIS ${CONDITION} pilot ==="
echo "game=${GAME}"
echo "seed=${SEED}"
echo "epochs=${EPOCHS}"
echo "precision=${PRECISION}"
echo "compile=${COMPILE}"
echo "compile_mode=${COMPILE_MODE}"
echo "run_name=${RUN_NAME}"
echo "run_dir=${RUN_DIR}"
echo "sls_args=${SLS_ARGS[*]}"

handle_timeout() {
    echo "=== USR1 received at $(date); resubmitting ${RUN_NAME} ==="
    mkdir -p "${RUN_DIR}"
    touch "${RESUBMIT_FLAG}"
    sbatch --time="${IRIS_SLS_TIME_LIMIT:-08:00:00}" "$0" "$CONDITION" "$GAME" "$SEED" "$EPOCHS" "$SLS_SMOOTHING" "$SLS_TOPK" "$PRECISION" "$COMPILE" "$COMPILE_MODE"
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" || true
    exit 0
}
trap handle_timeout USR1

RESUME_ARGS=()
if [[ -f "${RUN_DIR}/checkpoints/epoch.pt" ]]; then
    RESUME_ARGS=("common.resume=True" "hydra.output_subdir=null")
    echo "=== Resuming from ${RUN_DIR}/checkpoints ==="
fi

"$PYTHON" -u src/main.py \
    "env.train.id=${GAME}" \
    "common.device=cuda:0" \
    "common.seed=${SEED}" \
    "common.epochs=${EPOCHS}" \
    "common.precision=${PRECISION}" \
    "common.compile=${COMPILE}" \
    "common.compile_mode=${COMPILE_MODE}" \
    "wandb.mode=offline" \
    "wandb.project=iris-sls-baseline" \
    "wandb.group=e2e_ce_vs_sls" \
    "wandb.name=${RUN_NAME}" \
    "hydra.run.dir=${RUN_DIR}" \
    "${RESUME_ARGS[@]}" \
    "${SLS_ARGS[@]}" &

TRAIN_PID=$!
wait "$TRAIN_PID"
