#! /bin/bash

JOB_ID=1652502

FILE_DIR=/rstorage/youqi/$JOB_ID
FILES=$( find "$FILE_DIR" -name "*.root" -size +500c)

OUTPUT_DIR=/rstorage/youqi/$JOB_ID
hadd $OUTPUT_DIR/AnalysisResultsFinal.root $FILES