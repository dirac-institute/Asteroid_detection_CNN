#!/bin/bash

source ~/activate.sh

start_time=$(date +%s)
pipetask run --register-dataset-types \
-b $REPO \
-i HSC/RC2/defaults \
-o u/$USER/single_frame \
-p $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2_subset.yaml#singleFrame \
-j 32
echo -e "\nFINISHED, WOOHOO\n"
# Capture the end time and calculate the total runtime.
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(( (runtime % 3600) / 60 ))
seconds=$((runtime % 60))
printf "\nTotal runtime: %02d:%02d:%02d\n" $hours $minutes $seconds

