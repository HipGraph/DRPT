#ifndef DISTRIBUTED_MRPT_MATRIX_MULTIPLY_H
#define DISTRIBUTED_MRPT_MATRIX_MULTIPLY_H
#include <cblas.h>
#include <vector>
#include <mpi.h>
#include <string>
#include <omp.h>

#define VALUE_TYPE float
//#define DOUBLE_VALUE_TYPE double
#define FLOAT_VALUE_TYPE float
#define MPI_VALUE_TYPE MPI_FLOAT

using namespace std;

namespace dmrpt{
    enum class StorageFormat {RAW, COLUMN};
    class MathOp{
    public:
        VALUE_TYPE* multiply_mat(VALUE_TYPE *A, VALUE_TYPE *B, int A_rows, int B_cols,int A_cols,int alpha);
        VALUE_TYPE* build_sparse_local_random_matrix(int rows, int cols, float density, int seed);
        VALUE_TYPE* build_sparse_projection_matrix(int rank, int world_size, int total_dimension,int levels, float density, int seed);
        VALUE_TYPE* convert_to_row_major_format(vector<vector<VALUE_TYPE>> data);
        VALUE_TYPE* distributed_mean(VALUE_TYPE *data, int local_rows, int local_cols, int total_elements_per_col, dmrpt::StorageFormat format,int rank);
        VALUE_TYPE* distributed_variance(VALUE_TYPE *data, int rows, int cols,int total_elements_per_col, dmrpt::StorageFormat format,int rank);
        VALUE_TYPE* distributed_median(VALUE_TYPE *data, int rows, int cols,int total_elements_per_col,int no_of_bins, dmrpt::StorageFormat format,int rank);
        VALUE_TYPE calculate_distance(vector<VALUE_TYPE> data, vector<VALUE_TYPE> query);

    };
}

#endif //DISTRIBUTED_MRPT_MATRIX_MULTIPLY_H
