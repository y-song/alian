#! /bin/bash

JOB_ID=1799040

FILE_DIR=~/temp/$JOB_ID
FILES=$( find "$FILE_DIR" -name "*.root" -size +500c)

OUTPUT_DIR=~/temp/$JOB_ID
hadd $OUTPUT_DIR/AnalysisResultsFinal.root $FILES

mv $OUTPUT_DIR /rstorage/youqi/.
