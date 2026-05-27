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

set -euo pipefail

# Usage:
#   sbatch slurm/iris_sls_pilot.sl ce  PongNoFrameskip-v4 0 35
#   sbatch slurm/iris_sls_pilot.sl sls PongNoFrameskip-v4 0 35

CONDITION=${1:-ce}
GAME=${2:-PongNoFrameskip-v4}
SEED=${3:-0}
EPOCHS=${4:-35}
SLS_SMOOTHING=${5:-0.1}
SLS_TOPK=${6:-16}

if [[ "$CONDITION" != "ce" && "$CONDITION" != "sls" ]]; then
    echo "condition must be 'ce' or 'sls', got: $CONDITION" >&2
    exit 2
fi

cd /home/2500001/ftari001/iris-sls-baseline
mkdir -p slurm/logs

export PYTHONWARNINGS=ignore
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

PYTHON=/home/2500001/ftari001/venvs/iris-sls/bin/python
RUN_NAME="iris_${CONDITION}_${GAME}_seed${SEED}_e${EPOCHS}"

SLS_ARGS=(
    "training.world_model.sls_smoothing=0.0"
    "training.world_model.sls_topk=0"
)

if [[ "$CONDITION" == "sls" ]]; then
    SLS_ARGS=(
        "training.world_model.sls_smoothing=${SLS_SMOOTHING}"
        "training.world_model.sls_kernel=gaussian"
        "training.world_model.sls_sigma=null"
        "training.world_model.sls_topk=${SLS_TOPK}"
    )
fi

echo "=== IRIS ${CONDITION} pilot ==="
echo "game=${GAME}"
echo "seed=${SEED}"
echo "epochs=${EPOCHS}"
echo "run_name=${RUN_NAME}"
echo "sls_args=${SLS_ARGS[*]}"

"$PYTHON" -u src/main.py \
    "env.train.id=${GAME}" \
    "common.device=cuda:0" \
    "common.seed=${SEED}" \
    "common.epochs=${EPOCHS}" \
    "wandb.mode=offline" \
    "wandb.project=iris-sls-baseline" \
    "wandb.group=pilot_ce_vs_sls" \
    "wandb.name=${RUN_NAME}" \
    "${SLS_ARGS[@]}"
