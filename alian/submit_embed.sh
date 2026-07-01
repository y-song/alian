#!/bin/bash

#SBATCH --job-name=youqi
#SBATCH --partition=quick
#SBATCH --output=/rstorage/youqi/%A_%a.out
#SBATCH --error=/rstorage/youqi/%A_%a.err
#SBATCH --array=0-50
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00

FILE_LIST_PP=list/pp.txt
FILE_LIST_OO=list/OO.txt
CONFIG=config/embed.yaml
ANALYSIS_CODE=analysis/test/embed.py
OUTPUT_DIR=/rstorage/youqi/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    mkdir -p "$OUTPUT_DIR"
    cp submit_embed.sh "$OUTPUT_DIR"
    cp "$CONFIG" "$OUTPUT_DIR"
    cp "$ANALYSIS_CODE" "$OUTPUT_DIR"
fi

FILE_ID=$(( SLURM_ARRAY_TASK_ID + 1 ))
FILE_PP=$(sed -n "${FILE_ID}p" "$FILE_LIST_PP")
FILE_OO=$(sed -n "${FILE_ID}p" "$FILE_LIST_OO")
srun python "$ANALYSIS_CODE" -c "$CONFIG" -i1 "$FILE_PP" -i2 "$FILE_OO" -o "$OUTPUT_DIR/$(basename "$FILE_PP")"