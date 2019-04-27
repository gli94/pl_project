#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Stampede2 SKX nodes
#
#   *** Serial Job on SKX Normal Queue ***
#
# Last revised: 20 Oct 2017
#
# Notes:
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch skx.serial.slurm" on a Stampede2 login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#
#   -- For a good way to run multiple serial executables at the
#        same time, execute "module load launcher" followed
#        by "module help launcher".

#----------------------------------------------------

#SBATCH -J pl_proj_run_time           # Job name
#SBATCH -o pl_proj_run_time.o%j       # Name of stdout output file
#SBATCH -e pl_proj_run_time.e%j       # Name of stderr error file
#SBATCH -p skx-normal      # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=lige@utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A EE382N-20       # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...

module list
pwd
date

# Launch serial code...
./backtracking_hard
./sim_annealing_hard

#perf stat -r 50 -e cycles:u,cache-misses:u,L1-dcache-load-misses:u,L1-dcache-stores:u,LLC-loads ./${SIZE}_${OPT}

#perf stat -r 50 -e cycles:u,page-faults:u,LLC-load-misses,LLC-store-misses,LLC-stores ./${SIZE}_${OPT}

#../../../pin -t obj-intel64/dcache.so -- ./basic_4096
# Do not use ibrun or any other MPI launcher

# ---------------------------------------------------
.
