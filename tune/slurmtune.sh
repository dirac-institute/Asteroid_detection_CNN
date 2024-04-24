#!/bin/bash

if [ $# -lt 1 ]; then
        echo "Please specify number of workers"
		exit 1
fi
num_workers=$1

if [[ $(hostname) == *"bura"* ]]; then
	worker_node_name="computes_thin"
	chief_node_name="computes_thin"
	worker_account=$USER
	chief_account=$USER
	home_dir="/home/kmrakovcic"
	tuner_directory="/home/kmrakovcic/Tuner"
	walltime="24:00:00"
	port=8000
	cpus_per_task=12
	gpus=""
	module_load=""
	echo "HPC Bura detected"
elif [[ $(hostname) == *"klone"* ]]; then
	worker_node_name="gpu-a40"
	worker_account="escience"
	chief_node_name="compute-bigmem"
	chief_account="astro"
	home_dir="/mmfs1/home/kmrakovc"
	tuner_directory="/mmfs1/gscratch/dirac/kmrakovc/Tuner"
	walltime="168:00:00"
	port=8000
	cpus_per_task=4
	gpus="#SBATCH --gpus=2"
	module_load="module load cuda/12.3.2"
	echo "HPC Klone detected"
fi

if [ $num_workers -gt 1 ]; then
cat << EOF > tunerchief.sh
#!/bin/bash
#SBATCH --job-name=TunerC
#SBATCH --mail-type=ALL
#SBATCH --account=$chief_account
#SBATCH --output=$home_dir/Results/Asteroids/tunerchief.txt
#SBATCH --partition=$chief_node_name
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --time=$walltime

srun hostname
source ~/activate.sh

export KERASTUNER_TUNER_ID="chief"
export KERASTUNER_ORACLE_IP=\$(hostname)
export KERASTUNER_ORACLE_PORT=$port

$module_load
python3 main.py \
--train_dataset_path "../DATA/train1.tfrecord" \
--test_dataset_path "../DATA/test1.tfrecord" \
--tuner_destination $tuner_directory \
--arhitecture_destination "../DATA/arhitecture_tuned.json" \
--epochs 64 \
--batch_size 128 \
--class_balancing_alpha 0.95 \
--start_lr 0.001 \
--decay_lr_rate 0.95 \
--decay_lr_patience 6 \
--factor 4 \
--hyperband_iterations 64
EOF
chief_job_num=$(sbatch tunerchief.sh | tr -dc '0-9')
rm tunerchief.sh
chief_node_adress=""
while [ "$chief_node_adress" = "" ]
do
sleep 2
chief_node_adress=$(squeue -j $chief_job_num --format=%N -h)
done
for i in $(seq 1 $num_workers)
do
cat << EOF > tuner$i.sh
#!/bin/bash
#SBATCH --job-name=Tuner$i
#SBATCH --account=$worker_account
#SBATCH --output=$home_dir/Results/Asteroids/tuner$[i-1].txt
#SBATCH --partition=$worker_node_name
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --time=$walltime
$gpus

srun hostname
source ~/activate.sh
export KERASTUNER_TUNER_ID="tuner$[i-1]"
export KERASTUNER_ORACLE_IP="$chief_node_adress.hyak.local"
export KERASTUNER_ORACLE_PORT=$port
echo "KERASTUNER_ORACLE_ID: \$KERASTUNER_TUNER_ID"
echo "KERASTUNER_ORACLE_IP: \$KERASTUNER_ORACLE_IP"
echo "KERASTUNER_ORACLE_PORT: \$KERASTUNER_ORACLE_PORT"
$module_load
python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))"
python3 main.py \
--train_dataset_path "../DATA/train1.tfrecord" \
--test_dataset_path "../DATA/test1.tfrecord" \
--tuner_destination $tuner_directory \
--arhitecture_destination "../DATA/arhitecture_tuned.json" \
--epochs 64 \
--batch_size 128 \
--class_balancing_alpha 0.95 \
--start_lr 0.001 \
--decay_lr_rate 0.95 \
--decay_lr_patience 6 \
--factor 4 \
--hyperband_iterations 64
EOF
sbatch tuner$i.sh
rm tuner$i.sh
done
elif [ $num_workers -eq 1 ]; then
cat << EOF > tuner.sh
#!/bin/bash
#SBATCH --job-name=Tuner
#SBATCH --account=$worker_account
#SBATCH --output=$home_dir/Results/Asteroids/tuner0.txt
#SBATCH --partition=$worker_node_name
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --time=$walltime
$gpus

srun hostname
source ~/activate.sh
$module_load
python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))"

python3 main.py \
--train_dataset_path "../DATA/train1.tfrecord" \
--test_dataset_path "../DATA/test1.tfrecord" \
--tuner_destination $tuner_directory \
--arhitecture_destination "../DATA/arhitecture_tuned.json" \
--epochs 64 \
--batch_size 128 \
--class_balancing_alpha 0.95 \
--start_lr 0.001 \
--decay_lr_rate 0.95 \
--decay_lr_patience 6 \
--factor 2 \
--hyperband_iterations 64
EOF
sbatch tuner.sh
rm tuner.sh
fi