#!/bin/sh
#PBS -N tci15_ol1p2k
#PBS -l nodes=1:ppn=4
#PBS -l walltime=200:00:00
#PBS -l mem=2gb

export OMP_NUM_THREADS=4

cd /home/amilsted/fauxvac/esg2_tci_test15/maxD64_dt05/
python ols_1p_2k.py
