#ifndef DISTRIBUTED_MRPT_MATRIX_MULTIPLY_H
#define DISTRIBUTED_MRPT_MATRIX_MULTIPLY_H
#include <cblas.h>
#include <vector>
#include <mpi.h>
#include <string>
#include <omp.h>

using namespace std;

namespace dmrpt{
    class MathOp{
    public:
        double* multiply_mat(vector<vector<double>> A, vector<vector<double>> B);
        double* build_sparse_local_random_matrix(int rows, int cols, float density);
        double* build_sparse_projection_matrix(int rank, int world_size, int total_dimension,int levels, float density);

    };
}

#endif //DISTRIBUTED_MRPT_MATRIX_MULTIPLY_H
