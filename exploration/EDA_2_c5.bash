#!/bin/bash -l
#
# Batch script for bash users
#
#SBATCH -n 1 
#SBATCH -t 0-07:00:00
#SBATCH -J eda 
#SBATCH -o ./logs/eda_.%J.out
#SBATCH -e ./logs/eda_.%J.err
#SBATCH -p cosma
#SBATCH -A durham
#SBATCH --exclusive

module unload python                                                             
module load python/3.6.5

python3 EDA_ITNG_halos_2.py 


