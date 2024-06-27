#!/bin/bash

for i in "$@"; do
  case $i in
    -n=*|--name=*)
      RUN_NUM="${i#*=}"
      shift # past argument=value
      ;;
    -c=*|--cpus=*)
      CPU_NUM="${i#*=}"
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

echo "RUN_NUM  = ${RUN_NUM}"
echo "CPU_NUM  = ${CPU_NUM}"

# LSST stack activation
source $LSST_STACK_PATH || exit 1
setup lsst_distrib
setup source_injection
setup bps
echo -e "\nLSST stack activated\n"
start_time=$(date +%s)


# Make pipeline with injection
make_injection_pipeline -t postISRCCD -r $DRP_PIPE_DIR/pipelines/HSC/DRP-RC2.yaml \
-f $PROJECT_PATH/DATA/DRP-RC2_subset_injection.yaml --overwrite
pipetask build -p $PROJECT_PATH/DATA/DRP-RC2_subset_injection.yaml \
-c inject_exposure:selection="np.isin(injection_catalog['visit'], {visit})" \
-s $PROJECT_PATH/DATA/DRP-RC2_subset_injection.yaml || exit 1
echo -e "\nPipeline with injection created\n"

# Generate injection catalog
python3 $PROJECT_PATH/tools/generate_injection_catalog.py \
-r $REPO_PATH \
-i $INPUT_COLL \
-o $OUTPUT_COLL/injection_inputs_$RUN_NUM \
-n 20 \
-l 6 60 \
-m 20.0 24.0 \
-b 0.00 180.0 \
--where "$COLL_FILTER" \
--cpu_count $CPU_NUM \
--verbose || exit 1
echo -e "\nInjection catalog generated\n"

# Run the pipeline with injection
if [[ $(hostname) == *"sdf"* ]]; then
  echo -e "\nS3DF Detected, using BPS system. After job is finished, please run 'bash ~/inject_bps.sh' \n"
  cat << EOF > $HOME/inject_bps.sh
#!/bin/bash
source $LSST_STACK_PATH || exit 1
setup lsst_distrib
setup source_injection
setup bps
echo -e "\nLSST stack activated\n"
bps submit -b $REPO_PATH \
  -i $INPUT_COLL,$OUTPUT_COLL/injection_inputs_$RUN_NUM,$VISIT_SUMMARY_COLL \
  -o $OUTPUT_COLL/single_frame_injection_$RUN_NUM \
  -p $PROJECT_PATH/DATA/DRP-RC2_subset_injection.yaml#step1 \
  -d "$COLL_FILTER" \
  ${CTRL_BPS_DIR}/python/lsst/ctrl/bps/etc/bps_defaults.yaml
  allocateNodes.py -v -n 80 -c 16 -m 1-00:00:00 -q milano -g 120 s3df
rm inject_bps.sh
EOF
  echo -e "\nFINISHED\n"
else
  rm ~/inject_log_$RUN_NUM.txt
  pipetask --log-file ~/inject_log_$RUN_NUM.txt run --register-dataset-types \
  -b $REPO_PATH \
  -i $INPUT_COLL,$OUTPUT_COLL/injection_inputs_$RUN_NUM,$VISIT_SUMMARY_COLL \
  -o $OUTPUT_COLL/single_frame_injection_$RUN_NUM \
  -p $PROJECT_PATH/DATA/DRP-RC2_subset_injection.yaml#step1 \
  -j $CPU_NUM \
  -c inject_exposure:process_all_data_ids=True \
  -d "$COLL_FILTER"
  echo -e "\nFINISHED\n"
fi

# Capture the end time and calculate the total runtime.
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(( (runtime % 3600) / 60 ))
seconds=$((runtime % 60))
printf "\nTotal runtime: %02d:%02d:%02d\n" $hours $minutes $seconds