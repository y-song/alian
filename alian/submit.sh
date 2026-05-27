#!/bin/bash

#SBATCH --job-name=youqi
#SBATCH --partition=quick
#SBATCH --output=/rstorage/youqi/%A_%a.out
#SBATCH --error=/rstorage/youqi/%A_%a.err
#SBATCH --array=0-99
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00

FILE_LIST=list/PbPb_100GeV.txt
CONFIG=config/jewel.yaml
ANALYSIS_CODE=analysis/test/analyze_jewel.py
OUTPUT_DIR=/rstorage/youqi/$SLURM_ARRAY_JOB_ID

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    mkdir -p "$OUTPUT_DIR"
    cp submit.sh "$OUTPUT_DIR"
    cp "$CONFIG" "$OUTPUT_DIR"
    cp "$ANALYSIS_CODE" "$OUTPUT_DIR"
fi

FILE_ID=$(( SLURM_ARRAY_TASK_ID + 1 ))
FILE=$(sed -n "${FILE_ID}p" "$FILE_LIST")
srun python "$ANALYSIS_CODE" -c "$CONFIG" -i "$FILE" -o "$OUTPUT_DIR/$(basename "$FILE")"