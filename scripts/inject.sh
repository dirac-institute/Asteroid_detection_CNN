#!/bin/bash

# LSST stack activation
source ~/activate.sh
start_time=$(date +%s)
RUN_NUM="01"

# make pipeline with injection
make_injection_pipeline -t postISRCCD -r $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2_subset.yaml -f $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2_subset_injection.yaml --overwrite
pipetask build -p $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2_subset_injection.yaml -c inject_exposure:selection="np.isin(injection_catalog['visit'], {visit})" -s $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2_subset_injection.yaml
echo -e "\npipeline with injection created\n"
#generate_injection_catalog -a 149.8 150.7 -d 1.99 2.5 -n 2 -p source_type Trail -p trail_length 1 -p mag 15 17 19 21 23 25 27 -p beta 0 20 40 50 70 80 -w calexp -c u/$USER/RC2_subset/run_1 -b $REPO -i g r i z y -o u/$USER/injection_inputs_$RUN_NUM
#echo -e "\ninjection catalog generated\n"

rm ~/inject_$RUN_NUM.txt run
# make injection
# pipetask --long-log --log-file ~/inject_$RUN_NUM.txt run --register-dataset-types -b $REPO -i u/$USER/RC2_subset/run_1,u/$USER/injection_inputs_$RUN_NUM -o u/$USER/single_frame_injection_$RUN_NUM -p $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2_subset_injection.yaml#nightlyStep1 -j 32 -c inject_exposure:process_all_data_ids=True
pipetask --long-log --log-file ~/inject_$RUN_NUM.txt run --register-dataset-types -b $REPO -i u/$USER/RC2_subset/run_1,u/$USER/injection_inputs_$RUN_NUM -o u/$USER/single_frame_injection_$RUN_NUM -p $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2_subset_injection.yaml#nightlyStep1 -j 32 -c inject_exposure:process_all_data_ids=True
echo -e "\nFINISHED\n"

# Capture the end time and calculate the total runtime.
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(( (runtime % 3600) / 60 ))
seconds=$((runtime % 60))
printf "\nTotal runtime: %02d:%02d:%02d\n" $hours $minutes $seconds
