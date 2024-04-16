#!/bin/bash

# Getting the run number for the output colection name
if [ $# -lt 1 ]; then
        echo "Please specify RUN number"
		exit 1
fi
RUN_NUM=$1

# LSST stack activation, change the paths to your own
source ~/lsst_stack/loadLSST.sh
setup lsst_distrib
setup -k -r ~/lsst_stack/source_injection
setup -j -r /epyc/ssd/users/kmrakovc/DATA/rc2_subset
REPO="$RC2_SUBSET_DIR/SMALL_HSC/butler.yaml"
export REPO
BASE_DIR="/astro/users/kmrakovc/Projects/LSST_Streak_Detection"
echo -e "\nLSST stack activated\n"
start_time=$(date +%s)


# Make pipeline with injection
make_injection_pipeline -t postISRCCD -r $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2_subset.yaml -f $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2_subset_injection.yaml --overwrite
pipetask build -p $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2_subset_injection.yaml -c inject_exposure:selection="np.isin(injection_catalog['visit'], {visit})" -s $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2_subset_injection.yaml
echo -e "\nPipeline with injection created\n"

# Generate injection catalog
python3 $BASE_DIR/tools/generate_injection_catalog.py \
-r $REPO \
-i u/$USER/RC2_subset/run_1 \
-o u/$USER/injection_inputs_$RUN_NUM \
-n 20 \
-l 4 74 \
-m 25.0 25.5 \
-b 0.00 180.0
# this is the old way, do not use it
#generate_injection_catalog -a 149.8 150.7 -d 1.99 2.5 -n 2 -p source_type Trail -p trail_length 1 -p mag 15 17 19 21 23 25 27 -p beta 0 20 40 50 70 80 -w calexp -c u/$USER/RC2_subset/run_1 -b $REPO -i g r i z y -o u/$USER/injection_inputs_$RUN_NUM
echo -e "\nInjection catalog generated\n"

# Run the pipeline with injection
rm ~/inject_$RUN_NUM.txt run
pipetask --long-log --log-file ~/inject_$RUN_NUM.txt run --register-dataset-types -b $REPO -i u/$USER/RC2_subset/run_1,u/$USER/injection_inputs_$RUN_NUM -o u/$USER/single_frame_injection_$RUN_NUM -p $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2_subset_injection.yaml#nightlyStep1 -j 16 -c inject_exposure:process_all_data_ids=True
echo -e "\nFINISHED\n"

# Capture the end time and calculate the total runtime.
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(( (runtime % 3600) / 60 ))
seconds=$((runtime % 60))
printf "\nTotal runtime: %02d:%02d:%02d\n" $hours $minutes $seconds
