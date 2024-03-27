#!/bin/bash

source ~/activate.sh

# Input collections
RC2_SUBSET_DEFS=HSC/RC2_subset/defaults

# Output collections
RC2_SUBSET_COLL=u/$USER/RC2_subset/run_1

# Pipeline definition
RC2_SUBSET_PIPE=$DRP_PIPE_DIR/pipelines/HSC/DRP-RC2_subset.yaml

# Maximum procesors
RC2_SUBSET_PIPE_PROC=32

start_time=$(date +%s)
echo -e "\nStep1\n"
pipetask run --register-dataset-types -b $REPO -i $RC2_SUBSET_DEFS -o $RC2_SUBSET_COLL -p $RC2_SUBSET_PIPE#nightlyStep1 -j $RC2_SUBSET_PIPE_PROC

echo -e "\nStep2a\n"
pipetask run --register-dataset-types -b $REPO -i $RC2_SUBSET_DEFS -o $RC2_SUBSET_COLL -p $RC2_SUBSET_PIPE#nightlyStep2a -j $RC2_SUBSET_PIPE_PROC

echo -e "\nStep2b\n"
pipetask run --register-dataset-types -b $REPO -i $RC2_SUBSET_DEFS -o $RC2_SUBSET_COLL -p $RC2_SUBSET_PIPE#nightlyStep2b -j $RC2_SUBSET_PIPE_PROC

echo -e "\nStep2c\n"
pipetask run --register-dataset-types -b $REPO -i $RC2_SUBSET_DEFS -o $RC2_SUBSET_COLL -p $RC2_SUBSET_PIPE#nightlyStep2c

echo -e "\nStep2d\n"
pipetask run --register-dataset-types -b $REPO -i $RC2_SUBSET_DEFS -o $RC2_SUBSET_COLL -p $RC2_SUBSET_PIPE#nightlyStep2d -j $RC2_SUBSET_PIPE_PROC

#echo -e "\nStep3\n"
#pipetask run --register-dataset-types -b $REPO -i $RC2_SUBSET_DEFS -o $RC2_SUBSET_COLL -p $RC2_SUBSET_PIPE#nightlyStep3 -j $RC2_SUBSET_PIPE_PROC

#echo -e "\nStep4\n"
#pipetask run --register-dataset-types -b $REPO -i $RC2_SUBSET_DEFS -o $RC2_SUBSET_COLL -p $RC2_SUBSET_PIPE#nightlyStep4 -j $RC2_SUBSET_PIPE_PROC

#echo -e "\nStep5\n"
#pipetask run --register-dataset-types -b $REPO -i $RC2_SUBSET_DEFS -o $RC2_SUBSET_COLL -p $RC2_SUBSET_PIPE#nightlyStep5 -j $RC2_SUBSET_PIPE_PROC

#echo -e "\nStep8\n"
#pipetask run --register-dataset-types -b $REPO -i $RC2_SUBSET_DEFS -o $RC2_SUBSET_COLL -p $RC2_SUBSET_PIPE#nightlyStep8 -j $RC2_SUBSET_PIPE_PROC

echo -e "\nFINISHED\n"
# Capture the end time and calculate the total runtime.
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(( (runtime % 3600) / 60 ))
seconds=$((runtime % 60))
printf "\nTotal runtime: %02d:%02d:%02d\n" $hours $minutes $seconds

