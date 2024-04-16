#!/bin/bash


if [[ $(hostname) == *"bura"* ]]; then
	worker_node_name="computes_thin"
	chief_node_name="computes_thin"
	worker_account=$USER
	chief_account=$USER
	num_workers=4
	echo "HPC Bura detected"
elif [[ $(hostname) == *"klone"* ]]; then
	worker_node_name="gpu-a40"
	worker_account="escience"
	chief_node_name="gpu-a40"
	chief_account="escience"
	num_workers=4
	echo "HPC Klone detected"
fi

source ~/activate.sh

cat << EOF > tunerchief.sh
#!/bin/bash
#SBATCH --job-name=TunerC
#SBATCH --account=$chief_account
#SBATCH --output=~/Results/Asteroids/tunerchief.txt
#SBATCH --partition=$chief_node_name
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

srun hostname

export KERASTUNER_TUNER_ID="chief"
export KERASTUNER_ORACLE_IP=\$(hostname)
export KERASTUNER_ORACLE_PORT="81460"

python3 main.py \
--train_dataset_path "../DATA/train1.tfrecord" \
--test_dataset_path "../DATA/test1.tfrecord" \
--tuner_destination "../../Tuner/" \
--arhitecture_destination "../DATA/arhitecture_tuned.json" \
--epochs 64 \
--batch_size 128 \
--class_balancing_alpha 0.95 \
--start_lr 0.001 \
--decay_lr_rate 0.95 \
--decay_lr_patience 6 \
--factor 4 \
--hyperband_iterations 1
EOF
chief_job_num=$(sbatch tunerchief.sh | tr -dc '0-9')
rm tunerchief.sh
chief_node_adress=""
while [ "$chief_node_adress" = "" ]
do
sleep 10
chief_node_adress=$(squeue -j $chief_job_num --format=%N -h)
done
for i in $(seq 1 $num_workers)
do
cat << EOF > tuner$i.sh
#!/bin/bash
#SBATCH --job-name=Tuner$i
#SBATCH --account=$worker_account
#SBATCH --output=~/Results/Asteroids/tuner$i.txt
#SBATCH --partition=$worker_node_name
#SBATCH --gpus=2
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --time=48:00:00

srun hostname

export KERASTUNER_TUNER_ID="tuner$i"
export KERASTUNER_ORACLE_IP=$chief_node_adress
export KERASTUNER_ORACLE_PORT="81460"

python3 main.py \
--train_dataset_path "../DATA/train1.tfrecord" \
--test_dataset_path "../DATA/test1.tfrecord" \
--tuner_destination "../../Tuner/" \
--arhitecture_destination "../DATA/arhitecture_tuned.json" \
--epochs 64 \
--batch_size 128 \
--class_balancing_alpha 0.95 \
--start_lr 0.001 \
--decay_lr_rate 0.95 \
--decay_lr_patience 6 \
--factor 4 \
--hyperband_iterations 1
EOF
chief_job_num=$(sbatch tuner$i.sh | tr -dc '0-9')
rm tuner$i.sh
done