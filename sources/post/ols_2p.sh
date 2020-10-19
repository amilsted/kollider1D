#!/bin/sh
#PBS -N tci15_ol2p
#PBS -l nodes=1:ppn=8
#PBS -l walltime=200:00:00
#PBS -l mem=2gb

export OMP_NUM_THREADS=8

cd /home/amilsted/fauxvac/esg2_tci_test15/maxD64_dt05/
python ols_2p.py
