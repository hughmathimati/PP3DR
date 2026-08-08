#!/bin/bash
#SBATCH --job-name=double-upscale-finetune
#SBATCH --output=double-upscale-finetune-%j.out
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=48
#SBATCH --mem=512gb
#SBATCH --gres=gpu:h200-sxm:3
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-high-h200
#SBATCH --time=36:00:00

#set -x

source ~/.bashrc
conda activate Pi3
cd /vulcanscratch/hughma/PP3DR/
. /usr/share/Modules/init/bash
. /etc/profile.d/ummodules.sh
module load gcc
module load cuda
export PYTORCH_ALLOC_CONF=expandable_segments:True
accelerate launch training/PP3DR_trainer.py # Don't forget you need to `accelerate launch`, not just `python`!!!
