#!/bin/sh

#PBS -lnodes=2:ppn=16
#PBS -lwalltime=0:30:00
#PBS -A nn4661k

cd $PBS_O_WORKDIR
mpirun -np 32 python fit_HH_to_HH.py
wait