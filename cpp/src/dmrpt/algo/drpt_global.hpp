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

    struct ImageDataPoint {
        int index;
        vector<VALUE_TYPE> value;
    };

    class DRPTGlobal {


    private:
        int tree_depth;
        VALUE_TYPE *projected_matrix;
        VALUE_TYPE *projection_matrix;
        int intial_no_of_data_points;
        int ntrees;
        int starting_data_index;
        int rank;
        int world_size;
        StorageFormat storage_format;
        int total_data_set_size;
        int donate_per;
        int data_dimension;
        int transfer_threshold;

        //multiple trees
        vector<vector<vector<dmrpt::DataPoint>>> trees_data;
        vector<vector<VALUE_TYPE>> trees_splits;
        vector<vector<int>> trees_indices;
        vector<vector<vector<int>>> trees_leaf_first_indices_all;
        vector<vector<int>> trees_leaf_first_indices;

        vector<ImageDataPoint> original_data_processed;

        vector<vector<DataPoint>> leaf_data;


    public:

        DRPTGlobal();
        DRPTGlobal(VALUE_TYPE *projected_matrix, VALUE_TYPE *projection_matrix, int no_of_data_points, int tree_depth,
             vector <vector<VALUE_TYPE>> original_data, int ntrees,
             int starting_index,int total_data_set_size,int donate_per, int transfer_threshold, dmrpt::StorageFormat storage_format,
                   int rank, int world_size);

        void grow_global_tree();

        void grow_global_subtree(std::vector<DataPoint> data_vector,int total_data_set_size,int depth, int i, int tree);

        void  send_receive_data_points_if_zero(vector<DataPoint> left_data_points,vector<DataPoint> right_data_points, int *total_counts,
                                                            int *process_counts,int *disps,int depth, int tree);

        int detect_max_rank(int *total_counts, int direction);

        int detect_min_rank(int *total_counts, int direction);

        void send_receive_data_points_if_zero(vector<DataPoint> data_points, int* total_counts,int i,int direction, int depth, int tree);

        bool is_transfer_needed(int * total_counts, int direction);

    };
}

#endif //DISTRIBUTED_MRPT_DRPT_GLOBAL_H
