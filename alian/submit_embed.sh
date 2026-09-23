#!/bin/bash

#SBATCH --job-name=youqi
#SBATCH --partition=quick
#SBATCH --output=/home/youqi/temp/%A_%a.out
#SBATCH --error=/home/youqi/temp/%A_%a.err
#SBATCH --array=0
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00

FILE_LIST_PP=list/MC_pp_anchored_OO_hadd50.txt
FILE_LIST_OO=list/OO_mb.txt
CONFIG=config/embed.yaml
ANALYSIS_CODE=analysis/test/embed.py
OUTPUT_DIR=/rstorage/youqi/$SLURM_ARRAY_JOB_ID
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    mkdir -p "$OUTPUT_DIR"
    cp submit_embed.sh "$OUTPUT_DIR"
    cp "$CONFIG" "$OUTPUT_DIR"
    cp "$ANALYSIS_CODE" "$OUTPUT_DIR"
fi

TEMP_DIR=/scratch/u/youqi/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID
OUTPUT_DIR=/rstorage/youqi/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID
mkdir -p "$TEMP_DIR"
mkdir -p "$OUTPUT_DIR"

FILE_ID=$(( SLURM_ARRAY_TASK_ID + 1 ))
FILE_PP=$(sed -n "${FILE_ID}p" "$FILE_LIST_PP")
FILE_OO=$(sed -n "${FILE_ID}p" "$FILE_LIST_OO")
cp "$FILE_PP" "$TEMP_DIR/."
cp "$FILE_OO" "$TEMP_DIR/."
echo 'echo: srun python "$ANALYSIS_CODE" -c "$CONFIG" -i1 "$TEMP_DIR/$(basename "$FILE_PP")" -i2 "$TEMP_DIR/$(basename "$FILE_OO")" -o "$TEMP_DIR/out.root":'
echo "srun python \"$ANALYSIS_CODE\" -c \"$CONFIG\" -i1 \"$TEMP_DIR/$(basename "$FILE_PP")\" -i2 \"$TEMP_DIR/$(basename "$FILE_OO")\" -o \"$TEMP_DIR/out.root\""
srun python "$ANALYSIS_CODE" -c "$CONFIG" -i1 "$TEMP_DIR/$(basename "$FILE_PP")" -i2 "$TEMP_DIR/$(basename "$FILE_OO")" -o "$TEMP_DIR/out.root"
cp "$TEMP_DIR/out.root" "$OUTPUT_DIR/."