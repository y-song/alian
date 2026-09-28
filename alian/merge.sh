#! /bin/bash

JOB_ID=1986696

FILE_DIR=/rstorage/youqi/$JOB_ID
# FILE_DIR=~/temp/$JOB_ID
FILES=$( find "$FILE_DIR" -name "*.root" -size +500c)

OUTPUT_DIR=/rstorage/youqi/$JOB_ID
# OUTPUT_DIR=~/temp/$JOB_ID
hadd $OUTPUT_DIR/AnalysisResultsFinal.root $FILES