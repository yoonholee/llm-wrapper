#!/bin/bash
#SBATCH --job-name=yoonho_vllm
#SBATCH --nodelist=iris-hgx-2
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --output=slurm_logs/%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=yoonho@stanford.edu
#SBATCH --mail-type=FAIL,BEGIN,END

bash /iris/u/yoonho/dotfiles/slurm_init.sh 
source /iris/u/yoonho/dotfiles/activate_local_env.sh

# Constants
SECONDS_PER_DAY=86400
SLEEP_INTERVAL=300  # 5 minutes in seconds

# Print job start information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "Estimated end time: $(date -d "+$((2*SECONDS_PER_DAY)) seconds")"

# Trap Ctrl+C and job termination
cleanup() {
    echo "Job terminated at $(date)"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Main monitoring loop
while true; do
    echo "========================================"
    echo "  System Status Check - $(date)"
    echo "========================================"
    
    # GPU Status
    echo "GPU Status:"
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
    
    # CPU and Memory Status
    echo -e "\nCPU & Memory Status:"
    top -bn1 | head -n 5
    
    echo "----------------------------------------"
    sleep $SLEEP_INTERVAL
done


# Do ssh iris9