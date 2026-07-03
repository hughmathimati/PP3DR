#!/bin/bash
#SBATCH --job-name=dataset_tester
#SBATCH --output=dataset_tester-%j.out
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512gb
#SBATCH --gres=gpu:h200-sxm:4
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-high-h200
#SBATCH --time=36:00:00

#set -x

source ~/.bashrc
conda activate Pi3
cd /vulcanscratch/hughma/ViT/datasets
. /usr/share/Modules/init/bash
. /etc/profile.d/ummodules.sh
module load gcc
export PYTORCH_ALLOC_CONF=expandable_segments:True
python open_images_dataset.py
python youtube_vis_dataset.py