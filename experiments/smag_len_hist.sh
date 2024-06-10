#!/bin/bash

#SBATCH --job-name=EvalD
#SBATCH --mail-type=END,FAIL
#SBATCH --account=escience
#SBATCH --output=/mmfs1/home/kmrakovc/Results/Asteroids/evaluating.txt
#SBATCH --partition=gpu-a40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00

source ~/activate.sh
srun python3 mag_len_hist.py \
--model_path ../DATA/Trained_model_18796700.keras \
--tf_dataset_path ../DATA/test1.tfrecord,../DATA/test2.tfrecord,../DATA/test3.tfrecord,../DATA/test4.tfrecord  \
--collection u/kmrakovc/single_frame_injection_01,u/kmrakovc/single_frame_injection_02,u/kmrakovc/single_frame_injection_03,u/kmrakovc/single_frame_injection_04 \
--cpu_count $SLURM_CPUS_PER_TASK
