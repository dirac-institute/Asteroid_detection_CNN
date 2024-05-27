#!/bin/bash

# Getting the run number for the output colection name
if [ $# -lt 1 ]; then
        echo "Please specify RUN number"
		exit 1
fi
RUN_NUM=$1

# LSST stack activation
source $LSST_STACK_PATH
setup lsst_distrib
setup source_injection
echo -e "\nLSST stack activated\n"
start_time=$(date +%s)


# Make pipeline with injection
make_injection_pipeline -t postISRCCD -r $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2.yaml \
-f $PROJECT_PATH/DATA/DRP-RC2_subset_injection.yaml --overwrite
pipetask build -p $PROJECT_PATH/DATA/DRP-RC2_subset_injection.yaml \
-c inject_exposure:selection="np.isin(injection_catalog['visit'], {visit})" \
-s $PROJECT_PATH/DATA/DRP-RC2_subset_injection.yaml
echo -e "\nPipeline with injection created\n"

# Generate injection catalog
python3 $PROJECT_PATH/tools/generate_injection_catalog.py \
-r $REPO_PATH \
-i $INPUT_COLL \
-o $OUTPUT_COLL/injection_inputs_$RUN_NUM \
-n 20 \
-l 4 74 \
-m 20.0 25.5 \
-b 0.00 180.0 \
echo -e "\nInjection catalog generated\n"

# Run the pipeline with injection
rm ~/inject_log_$RUN_NUM.txt
pipetask --log-file ~/inject_log_$RUN_NUM.txt run --register-dataset-types \
-b $REPO_PATH \
-i $INPUT_COLL,$OUTPUT_COLL/injection_inputs_$RUN_NUM,$VISIT_SUMMARY \
-o $OUTPUT_COLL/single_frame_injection_$RUN_NUM \
-p $PROJECT_PATH/DATA/DRP-RC2_subset_injection.yaml#nightlyStep1 \
-j 16 \
-c inject_exposure:process_all_data_ids=True
-where "detector not in (58, 50, 42, 47, 49, 41) AND visit not in (29336, 11690, 11698, 29350, 11696, 11704, 11710, 11694, 1220, 1204, 23694, 1206, 23706, 23704, 1214, 23718, 19694, 19680, 30490, 1242, 19684, 30482, 19696, 1248, 1178, 17948, 17950, 17904, 1184, 17906, 17926, 17900, 11738, 358, 11724, 346, 22632, 11740, 22662, 322) and tract != 9813 and skymap = 'hsc_rings_v1'"
echo -e "\nFINISHED\n"

# Capture the end time and calculate the total runtime.
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(( (runtime % 3600) / 60 ))
seconds=$((runtime % 60))
printf "\nTotal runtime: %02d:%02d:%02d\n" $hours $minutes $seconds
