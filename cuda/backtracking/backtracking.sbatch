#!/bin/bash
#SBATCH -J backtracking           # job name
#SBATCH -o backtracking.o%j       # output and error file name (%j expands to jobID)
#SBATCH -N 1               # total number of node
#SBATCH -n 1               # total number of mpi tasks requested
#SBATCH -p gtx             # queue (partition) -- normal, development, etc.
#SBATCH -t 00:10:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=lige@utexas.edu
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH -A EE382N-20
module load gcc
module load cuda
./cudaBacktracking
