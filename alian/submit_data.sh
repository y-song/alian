#!/bin/bash

#SBATCH --job-name=oo_track_qa
#SBATCH --partition=std
#SBATCH --output=/home/youqi/temp/%A_%a.out
#SBATCH --error=/home/youqi/temp/%A_%a.err
#SBATCH --array=0-122
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=05:00:00
set -euo pipefail

FILE_LIST=list/OO_fix.txt
CONFIG=config/test.yaml
ANALYSIS_CODE=analysis/test/track_qa.py
OUTPUT_DIR=/rstorage/youqi/$SLURM_ARRAY_JOB_ID

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    mkdir -p "$OUTPUT_DIR"
    cp submit_data.sh "$OUTPUT_DIR"
    cp "$CONFIG" "$OUTPUT_DIR"
    cp "$ANALYSIS_CODE" "$OUTPUT_DIR"
fi

TEMP_DIR=/scratch/u/youqi/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID
OUTPUT_DIR=/rstorage/youqi/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID
mkdir -p "$TEMP_DIR"
mkdir -p "$OUTPUT_DIR"

FILE_ID=$(( SLURM_ARRAY_TASK_ID + 1 ))
FILE=$(sed -n "${FILE_ID}p" "$FILE_LIST")
cp "$FILE" "$TEMP_DIR/."
srun python "$ANALYSIS_CODE" -c "$CONFIG" -i "$TEMP_DIR/$(basename "$FILE")" -o "$TEMP_DIR/out.root"
cp "$TEMP_DIR/out.root" "$OUTPUT_DIR/."