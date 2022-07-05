#ifndef DISTRIBUTED_MRPT_DRPT_GLOBAL_H
#define DISTRIBUTED_MRPT_DRPT_GLOBAL_H

#include <cblas.h>
#include <vector>
#include "drpt.hpp"
#include "../math/matrix_multiply.hpp"
#include <mpi.h>
#include <string>
#include <omp.h>




namespace dmrpt {
    struct DataPoint {
        int index;
        VALUE_TYPE distance;
        VALUE_TYPE value;
    };

    class DRPTGlobal {


    private:
        int tree_depth;
        VALUE_TYPE *projected_matrix;
        VALUE_TYPE *projection_matrix;
        int no_of_data_points;
        int ntrees;
        int starting_data_index;
        int rank;
        int world_size;
        StorageFormat storage_format;
        int total_data_set_size;

        //multiple trees
        vector<vector<vector<dmrpt::DataPoint>>> trees_data;
        vector<vector<VALUE_TYPE>> trees_splits;
        vector<vector<int>> trees_indices;
        vector<vector<vector<int>>> trees_leaf_first_indices_all;
        vector<vector<int>> trees_leaf_first_indices;
        vector<vector<VALUE_TYPE>> original_data;

        vector<vector<DataPoint>> leaf_data;


    public:

        DRPTGlobal();
        DRPTGlobal(VALUE_TYPE *projected_matrix, VALUE_TYPE *projection_matrix, int no_of_data_points, int tree_depth,
             vector <vector<VALUE_TYPE>> original_data, int ntrees,
             int starting_index,int total_data_set_size, dmrpt::StorageFormat storage_format, int rank, int world_size);

        void grow_global_tree();

        void grow_global_subtree(std::vector<DataPoint> data_vector, int total_data_set_size,int depth, int i, int tree);

    };
}

#endif //DISTRIBUTED_MRPT_DRPT_GLOBAL_H
