#!/bin/bash
#SBATCH --job-name=jupyter_backend
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=jupyter_log-%j.txt
#SBATCH --partition=vulcan-ampere
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-high

# 1. Load your environment
source ~/.bashrc
conda activate Pi3
cd /vulcanscratch/hughma/
. /usr/share/Modules/init/bash
. /etc/profile.d/ummodules.sh
module load gcc
module load cuda

# 2. Get the hostname of the compute node (e.g., nexusvulcan01)
NODE_HOSTNAME=$(hostname)

# 3. Pick a random port to avoid collisions with other users
PORT=$(shuf -i 8000-9000 -n 1)

# 4. Print the exact SSH command you will need to run locally
echo "=================================================================="
echo "Step A: Run this command in a NEW terminal on your LOCAL computer:"
echo "ssh -J hughma@nexusvulcan01 -N -f -L ${PORT}:localhost:${PORT} hughma@${NODE_HOSTNAME}"
echo "=================================================================="

# 5. Start the Jupyter server
jupyter lab --no-browser --port=${PORT} --ip=127.0.0.1