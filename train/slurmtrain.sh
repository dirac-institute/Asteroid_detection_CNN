#!/bin/bash

#SBATCH --job-name=TrainD
#SBATCH --mail-type=ALL
#SBATCH --account=escience
#SBATCH --output=/mmfs1/home/kmrakovc/Results/Asteroids/training.txt
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --time=125:00:00

source ~/activate.sh
module load cuda/12.3.2
srun python3 main.py \
--train_dataset_path ../DATA/train1.tfrecord \
--test_dataset_path ../DATA/test1.tfrecord \
--arhitecture ../DATA/arhitecture_tuned.json \
--model_destination ../DATA/Trained_model_3 \
--no-multiworker \
--epochs 256 \
--batch_size 128 \
--start_lr 0.0005 \
--decay_lr_rate 0.75 \
--decay_lr_patience 10
