# DRPT
This is the official implementation for IPDPS 2023 paper titled "Distributed Sparse Random Projection Trees
for Constructing K-Nearest Neighbor Graphs".

## System Requirements
Users need to have the following softwares/tools installed in their PC/server. The source code was compiled and run successfully in both Linux and macOS.
```
GCC version >= 4.9
OpenMP version >= 5.0
MPI version >=3.1
OpenBLAS version >= v0.3.0
```
Some helpful links for installation can be found at [GCC](https://gcc.gnu.org/install/), [OpenMP](https://clang-omp.github.io) and [Environment Setup](http://heather.cs.ucdavis.edu/~matloff/158/ToolsInstructions.html#compile_openmp).

## Compile DRPT
To compile DRPT, follow the following instructions:
 * Clone repository
```
git clone https://github.com/HipGraph/DRPT
```
* Traverse into  DRPT/cpp
```
cd DRPT/cpp
```
* Create build directory
```
mkdir build
```
* Compile code
```
cmake -DCMAKE_CXX_FLAGS="-fopenmp"  ..
make all
```
If all steps are successfully completed binary is created in  "bin/drpt".

## Users: Run DRPT Job on HPC Resource

Input file must be ubyte format or fbin format.
```
$ drpt -input ./drpt/tests/datasets/train-images-idx3-ubyte.gz train-images-idx3-ubyte -output ./drpt/tests/datasets/output -data-set-size 60000 -dimension 784  -ntrees 8  -nn 10  -locality 1 -file_format 0
```
Here, input options are described below:
```
-input <string>, full path of input file (required).
-output <string>, directory where output file will be stored, output will be stored in  edge list format. (default: current directory) 
-data-set-size <int>, size of total dataset.
-dimension <int>, dimension of the dataset.
-ntrees <int>, number of trees.
-nn <int>, number of nearest neighbours
-locality 1 <int>, indicator to switch on the locality based data gathering 1 indicates on and 0 indicates off
-file_format 0 <int>, 0 indicates ubyte format and 1 indicates fbin format
```

Sample SLURM batch script to run DRPT for MNIST dataset on 4 nodes each having 4 processes on cori haswell partition.
```
#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30:00
#SBATCH --nodes=4
#SBATCH --tasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --constraint=haswell
#SBATCH --output=%j.log

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=8

srun -n 16  drpt -input drpt/tests/datasets/train-images-idx3-ubyte.gz train-images-idx3-ubyte -output ./ -data-set-size 60000 -dimension 784  -ntrees 8  -nn 10  -locality 1 -file_format 0
```


## Citation
If you find this repository helpful, please cite our papers as follows:
```
@INPROCEEDINGS{10177410,
  author={Ranawaka, Isuru and Rahman, Md Khaledur and Azad, Ariful},
  booktitle={2023 IEEE International Parallel and Distributed Processing Symposium (IPDPS)}, 
  title={Distributed Sparse Random Projection Trees for Constructing K-Nearest Neighbor Graphs}, 
  year={2023},
  volume={},
  number={},
  pages={36-46},
  doi={10.1109/IPDPS54959.2023.00014}}
```

## Contact
This repository is maintained by Isuru Ranawaka. If you have questions, please ping me at `isjarana@iu.edu` or create an issue.#DRPT
