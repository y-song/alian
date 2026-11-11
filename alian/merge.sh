#! /bin/bash

JOB_ID=2022784
N_TASKS=33   # array 0-32 in submit_embed.sh

FILE_DIR=/rstorage/youqi/$JOB_ID
# FILE_DIR=~/temp/$JOB_ID

# Build the list from explicit paths. Do not use find/wildcards here: on NFS the
# directory listing can lag behind, while a direct path lookup works.
FILES=()
for (( i=0; i<N_TASKS; i++ )); do
    f=$FILE_DIR/$i/out.root
    if [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 500 ]; then
        FILES+=("$f")
    else
        echo "WARNING: missing or empty: $f" >&2
    fi
done
echo "Merging ${#FILES[@]} / $N_TASKS files"

OUTPUT_DIR=/rstorage/youqi/$JOB_ID
# OUTPUT_DIR=~/temp/$JOB_ID
hadd -f $OUTPUT_DIR/AnalysisResultsFinal.root "${FILES[@]}"
