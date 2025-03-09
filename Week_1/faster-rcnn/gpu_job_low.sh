#!/bin/bash
#SBATCH --ntasks-per-node=4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D . # working directory
#SBATCH -t 0-05:00 # Runtime in D-HH:MM
#SBATCH -p mlow # Partition to submit to
#SBATCH -q masterhigh  # This way will only requeue of dcc partition
#SBATCH --mem 32000 # 4GB memory
#SBATCH --gres gpu:1 # Request of 1 gpu
#SBATCH -o out/%j.out # File to which STDOUT will be written
#SBATCH -e out/%j.err # File to which STDERR will be written

# Capture Python logs into a timestamped file in the logs directory
python inference.py
