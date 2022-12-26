#ifndef DISTRIBUTED_MRPT_DRPT_GLOBAL_H
#define DISTRIBUTED_MRPT_DRPT_GLOBAL_H

#include "../math/math_operations.hpp"
#include "drpt_local.hpp"
#include <cblas.h>
#include <map>
#include <mpi.h>
#include <omp.h>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace dmrpt {

struct PriorityMap {
  int leaf_index;
  float priority;
};

class DRPTGlobal {

private:
  int tree_depth;
  VALUE_TYPE *projected_matrix;
  VALUE_TYPE *projection_matrix;
  int local_dataset_size;
  int ntrees;
  int starting_data_index;
  int rank;
  int world_size;
  int global_dataset_size;
  int data_dimension;

  // multiple trees
  vector<vector<vector<dmrpt::DataPoint>>> trees_data;
  vector<vector<VALUE_TYPE>> trees_splits;
  vector<vector<vector<DataPoint>>> trees_leaf_first_indices_all;
  vector<vector<vector<DataPoint>>> trees_leaf_first_indices;

  vector<vector<VALUE_TYPE>> data_points;

  vector<vector<int>> index_to_tree_leaf_mapper;

  vector<vector<vector<DataPoint>>> trees_leaf_first_indices_rearrange;

  string input_path;
  string output_path;

public:
  DRPTGlobal();

  DRPTGlobal(VALUE_TYPE *projected_matrix, VALUE_TYPE *projection_matrix,
             int no_of_data_points, int dimension, int tree_depth, int ntrees,
             int starting_index, int global_dataset_size, int rank,
             int world_size);

  void grow_global_tree(vector<vector<VALUE_TYPE>> &data_points);

  void grow_global_subtree(vector<vector<DataPoint>> &child_data_tracker,
                           vector<int> &total_size_vector, int depth, int tree);

  void calculate_tree_leaf_correlation(string outpath);

  void derive_global_datavector_sizes(vector<vector<DataPoint>> &child_data_tracker,vector<int> &global_size_vector,
		  int current_nodes, int next_split);

  vector<vector<DataPoint>>
  collect_similar_data_points(int tree, bool use_data_locality_optimization,
                              vector<set<int>> &index_distribution);
};
} // namespace dmrpt

#endif // DISTRIBUTED_MRPT_DRPT_GLOBAL_H
