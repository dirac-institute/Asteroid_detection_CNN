#!/bin/bash

#SBATCH --job-name=TrainD
#SBATCH --mail-type=ALL
#SBATCH --account=escience
#SBATCH --output=/mmfs1/home/kmrakovc/Results/Asteroids/training_%j.txt
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00

source ~/activate.sh
module load cuda/12.3.2
srun python3 main.py \
--train_dataset_path ../DATA/train1.tfrecord \
--test_dataset_path ../DATA/test1.tfrecord \
--arhitecture ../DATA/arhitecture_tuned.json \
--model_destination ../DATA/Trained_model_$SLURM_JOB_ID \
--no-multiworker \
--kernel_size 3 \
--merge_operation "add" \
--epochs 512 \
--alpha 0.9 \
--gamma 3 \
--batch_size 512 \
--start_lr 0.001 \
--decay_lr_rate 0.75 \
--decay_lr_patience 8 \
--no-verbose
