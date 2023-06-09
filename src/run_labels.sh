#!/bin/sh
#SBATCH -J train_v2GAT2                   #job name
#SBATCH --time=00-00:30:00          #requested time (DD-HH:MM:SS)
#SBATCH -p preempt,batch,largemem   #running on "mpi" partition/queue
#SBATCH -N 1                        #nodes
#SBATCH -n 16                       #total tasks across all nodes
#SBATCH --mem=16g                   #requesting RAM total
#SBATCH --output=train_v2GAT.%j.%N.out   #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=train_v2GAT.%j.%N.err    #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL             #email optitions
#SBATCH --mail-user=czhao07@tufts.edu


# conda info --envs
module load anaconda/2021.11
source activate cs150

# make sure myscript.py exists in the current directory or provide the full path to scrip
python /cluster/home/czhao07/CS150_DGL_CallGraphClassification/src/experiments.py
# python -c "import time; time.sleep(120)"


# srun --time=00-02:00:00 -p preempt --mem=8g --pty bash
# srun --time=00-01:00:00 -p interactive --mem=8g --pty bash
# use ls *.csv -lrt to track latest changes in the directory
# use sinfo -N -l | grep batch to find available nodes
# only `mixed` or `idle` nodes are available for job submission