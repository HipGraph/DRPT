#ifndef DISTRIBUTED_MRPT_MRPT_H
#define DISTRIBUTED_MRPT_MRPT_H

#include <cblas.h>
#include <vector>
#include "drpt_local.hpp"
#include "../math/math_operations.hpp"
#include <mpi.h>
#include <string>
#include <omp.h>

namespace drpt {
struct DataPoint {
  int src_index;
  int index;
  VALUE_TYPE distance;
  VALUE_TYPE value;
  vector<VALUE_TYPE> image_data;
};

class DRPTLocal {

 private:
  int tree_depth;
  VALUE_TYPE *projected_matrix;
  VALUE_TYPE *projection_matrix;
  int no_of_data_points;
  int ntrees;
  int starting_data_index;
  int rank;
  int world_size;

  //single tree
  drpt::StorageFormat storageFormat;
  vector <vector<VALUE_TYPE>> data;
  vector<VALUE_TYPE> splits;
  vector<int> indices;
  vector <std::vector<int>> leaf_first_indices_all;
  vector<int> leaf_first_indices;

  //multiple trees
  vector <vector<vector < VALUE_TYPE>>>
  trees_data;
  vector <vector<VALUE_TYPE>> trees_splits;
  vector <vector<int>> trees_indices;
  vector <vector<vector < int>>>
  trees_leaf_first_indices_all;
  vector <vector<int>> trees_leaf_first_indices;
  vector <vector<VALUE_TYPE>> original_data;

 public:
  DRPTLocal();

  DRPTLocal(VALUE_TYPE *projected_matrix, VALUE_TYPE *projection_matrix, int no_of_data_points, int tree_depth,
	   vector <vector<VALUE_TYPE>> original_data, int ntrees,
	   int starting_index, int rank, int world_size);

  void grow_local_tree();

  void grow_local_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
						  int depth, int i, int tree);

  vector <vector<int>> get_all_leaf_node_indices(int tree);

  void count_leaf_sizes(int datasize, int level, int depth, std::vector<int> &out_leaf_sizes);

  void count_first_leaf_indices(std::vector<int> &indices, int datasize, int depth);

  void count_first_leaf_indices_all(std::vector <std::vector<int>> &indices, int datasize, int depth_max);

};
}

#endif //DISTRIBUTED_MRPT_MRPT_H
