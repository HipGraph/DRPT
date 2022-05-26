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
        double* multiply_mat(double *A, double *B, int A_rows, int B_cols,int A_cols,int alpha);
        double* build_sparse_local_random_matrix(int rows, int cols, float density);
        double* build_sparse_projection_matrix(int rank, int world_size, int total_dimension,int levels, float density);
        double * convert_to_row_major_format(vector<vector<double>> data);

    };
}

#endif //DISTRIBUTED_MRPT_MATRIX_MULTIPLY_H
