#!/bin/bash

# This script selects 50 files from INPUT_LIST and combines them using hadd,
# repeating this process 50 times to create 50 different combined files.

INPUT_LIST="/home/youqi/alian/alian/list/MC_pp_anchored_OO.txt"
OUTPUT_DIR="/rstorage/youqi/MC_pp_anchored_OO_hadd50"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if input list exists
if [ ! -f "$INPUT_LIST" ]; then
    echo "Error: Input file $INPUT_LIST not found."
    exit 1
fi

echo "Starting the batch process to create 50 combined files..."

for i in {1..50}
do
    OUTPUT_FILE="${OUTPUT_DIR}/combined${i}.root"

    # Randomly select 50 files from the list
    # FILES=$(shuf -n 50 "$INPUT_LIST")

    # Select files according to line number
    START_LINE=$(( (i - 1) * 50 + 1 ))
    END_LINE=$(( i * 50 ))
    FILES=$(sed -n "${START_LINE},${END_LINE}p" "$INPUT_LIST")

    COUNT=$(echo "$FILES" | wc -l)

    if [ "$COUNT" -eq 0 ]
    then
        echo "No more files found. Stopping at index $i."
        break
    fi

    echo "[$i/50] Combining $COUNT files from lines ${START_LINE}-${END_LINE} into $OUTPUT_FILE..."

    hadd "$OUTPUT_FILE" $FILES

    if [ $? -ne 0 ]; then
        echo "Error: hadd failed at index $i."
        exit 1
    fi
done

echo "Successfully created 50 combined files in $OUTPUT_DIR."
