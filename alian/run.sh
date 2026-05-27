#!/bin/bash

if [ "$1" != "" ]; then
  ANALYSIS_CODE=$1
  echo "Analysis code: $ANALYSIS_CODE"
else
  echo "Wrong command line arguments"
fi

if [ "$2" != "" ]; then
  CONFIG=$2
  echo "Config: $CONFIG"
else
  echo "Wrong command line arguments"
fi

if [ "$3" != "" ]; then
  INPUT_FILE=$3
  echo "Input file: $INPUT_FILE"
else
  echo "Wrong command line arguments"
fi

if [ "$4" != "" ]; then
  JOB_ID=$4
  echo "Job ID: $JOB_ID"
else
  echo "Wrong command line arguments"
fi

python $ANALYSIS_CODE -c $CONFIG -i $INPUT_FILE -o "/rstorage/youqi/$JOB_ID/$(basename "$INPUT_FILE")"