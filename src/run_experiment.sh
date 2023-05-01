#!/bin/sh
#SBATCH -J experimt_1k          #job name
#SBATCH --time=00-00:20:00      #requested time (DD-HH:MM:SS)
#SBATCH -p batch                #running on "mpi" partition/queue
#SBATCH -N 16                   #nodes
#SBATCH -n 2                    #2 tasks total
#SBATCH -c 1                    #1 cpu cores per task
#SBATCH --mem=2g                        #requesting 2GB of RAM total
#SBATCH --output=experimt_1k.%j.%N.out  #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=experimt_1k.%j.%N.err   #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL                 #email optitions#SBATCH --mail-user=Your_Tufts_Email@tufts.edu

#[commands_you_would_like_to_exe_on_the_compute_nodes]
# for example, running a python script
# 1st, load the modulemodule 
conda activate cs150

# run python
python experiments.py 
#make sure myscript.py exists in the current directory or provide thefull path to script