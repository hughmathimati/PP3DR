#!/bin/bash
#SBATCH --job-name=object_net
#SBATCH --partition=vulcan-cpu
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-cpu
#SBATCH --time=4:00:00

#set -x

source ~/.bashrc
conda activate Pi3
cd /vulcanscratch/hughma/ViT/datasets
. /usr/share/Modules/init/bash
. /etc/profile.d/ummodules.sh
module load gcc
export PYTORCH_ALLOC_CONF=expandable_segments:True
python object_net_dataset.py