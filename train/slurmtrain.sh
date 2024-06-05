#!/bin/bash

#SBATCH --job-name=TrainD
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --account=escience
#SBATCH --output=/mmfs1/home/kmrakovc/Results/Asteroids/training_%j.txt
#SBATCH --partition=gpu-a40
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:6
#SBATCH --time=5-00:00:00

source ~/activate.sh
module load cuda/12.3.2
srun python3 main.py \
--train_dataset_path ../DATA/train1.tfrecord \
--test_dataset_path ../DATA/test1.tfrecord \
--arhitecture ../arhitecture.json \
--model_destination ../DATA/Trained_model_$SLURM_JOB_ID \
--no-multiworker \
--kernel_size 7 \
--merge_operation "concat" \
--epochs 1024 \
--alpha 0.99 \
--gamma 3.1 \
--batch_size 256 \
--start_lr 0.001 \
--decay_lr_rate 0.75 \
--decay_lr_patience 10 \
--no-verbose
