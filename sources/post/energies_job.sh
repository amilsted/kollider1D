#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH -J "energies"

export OMP_NUM_THREADS=1

python energies.py
