#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH -J "ol2k"

export OMP_NUM_THREADS=10

python ol_scan.py 2k $1 1001 10
