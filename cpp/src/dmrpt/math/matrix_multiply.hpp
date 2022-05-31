#ifndef DISTRIBUTED_MRPT_MATRIX_MULTIPLY_H
#define DISTRIBUTED_MRPT_MATRIX_MULTIPLY_H
#include <cblas.h>
#include <vector>
#include <mpi.h>
#include <string>
#include <omp.h>

using namespace std;

namespace dmrpt{
    enum class StorageFormat {RAW, COLUMN};
    class MathOp{
    public:
        double* multiply_mat(double *A, double *B, int A_rows, int B_cols,int A_cols,int alpha);
        double* build_sparse_local_random_matrix(int rows, int cols, float density);
        double* build_sparse_projection_matrix(int rank, int world_size, int total_dimension,int levels, float density);
        double* convert_to_row_major_format(vector<vector<double>> data);
        double* distributed_mean(double *data, int local_rows, int local_cols, int total_elements_per_col, dmrpt::StorageFormat format,int rank);
        double* distributed_variance(double *data, int rows, int cols,int total_elements_per_col, dmrpt::StorageFormat format,int rank);
        double* distributed_median(double *data, int rows, int cols,int total_elements_per_col,int no_of_bins, dmrpt::StorageFormat format,int rank);

    };
}

#endif //DISTRIBUTED_MRPT_MATRIX_MULTIPLY_H
