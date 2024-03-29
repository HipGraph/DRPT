#ifndef DISTRIBUTED_MRPT_MATRIX_MULTIPLY_H
#define DISTRIBUTED_MRPT_MATRIX_MULTIPLY_H
#include <cblas.h>
#include <mpi.h>
#include <omp.h>
#include <string>
#include <vector>

#define VALUE_TYPE float
//#define DOUBLE_VALUE_TYPE double
#define FLOAT_VALUE_TYPE float
#define MPI_VALUE_TYPE MPI_FLOAT

using namespace std;

namespace drpt {
enum class StorageFormat { RAW, COLUMN };
class MathOp {
 public:
  VALUE_TYPE *multiply_mat(VALUE_TYPE *A, VALUE_TYPE *B, int A_rows, int B_cols,
						   int A_cols, int alpha);
  VALUE_TYPE *build_sparse_local_random_matrix(int rows, int cols,
											   float density, int seed);
  VALUE_TYPE *build_sparse_projection_matrix(int rank, int world_size,
											 int total_dimension, int levels,
											 float density, int seed);
  VALUE_TYPE *convert_to_row_major_format(vector <vector<VALUE_TYPE>> &data);
  VALUE_TYPE *distributed_mean(vector<VALUE_TYPE> &data, vector<int> local_rows,
							   int local_cols,
							   vector<int> total_elements_per_col,
							   drpt::StorageFormat format, int rank);
  VALUE_TYPE *distributed_variance(vector<VALUE_TYPE> &data,
								   vector<int> local_rows, int cols,
								   vector<int> total_elements_per_col,
								   drpt::StorageFormat format, int rank);
  VALUE_TYPE *distributed_median(vector<VALUE_TYPE> &data,
								 vector<int> local_rows, int cols,
								 vector<int> total_elements_per_col,
								 int no_of_bins, drpt::StorageFormat format,
								 int rank);
  VALUE_TYPE calculate_distance(vector<VALUE_TYPE> &data,
								vector<VALUE_TYPE> &query);
  VALUE_TYPE
  calculate_approx_distance(vector<VALUE_TYPE> &data, vector<VALUE_TYPE> &query,
							int start_index, int end_index);
};
} // namespace drpt

#endif // DISTRIBUTED_MRPT_MATRIX_MULTIPLY_H
