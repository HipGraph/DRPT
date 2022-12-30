#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --tasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --constraint=haswell
#SBATCH --output=%j.log

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=8

srun -n 4  distributed_mrpt -input distributed-mrpt/tests/datasets/train-images-idx3-ubyte.gz train-images-idx3-ubyte -output ./ -data-set-size 60000 -dimension 784  -ntrees 8  -nn 10  -locality 1 -file_format 0
