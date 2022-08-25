#ifndef DISTRIBUTED_MRPT_DRPT_GLOBAL_H
#define DISTRIBUTED_MRPT_DRPT_GLOBAL_H

#include <cblas.h>
#include <vector>
#include "drpt.hpp"
#include "../math/matrix_multiply.hpp"
#include <mpi.h>
#include <string>
#include <omp.h>
#include <map>
#include <unordered_map>


namespace dmrpt {


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
        unordered_map<string, VALUE_TYPE> distance_map;

        //multiple trees
        vector <vector<vector < dmrpt::DataPoint>>>
        trees_data;
        vector <vector<VALUE_TYPE>> trees_splits;
        vector <vector<vector < DataPoint>>>
        trees_leaf_first_indices_all;
        vector <vector<vector < DataPoint>>>
        trees_leaf_first_indices;

        vector <ImageDataPoint> original_data_processed;
        vector <ImageDataPoint> final_collected_data;

        vector <vector<DataPoint>> leaf_data;


        string input_path;
        string output_path;


    public:

        DRPTGlobal();

        DRPTGlobal(VALUE_TYPE *projected_matrix, VALUE_TYPE *projection_matrix, int no_of_data_points, int tree_depth,
                   vector <vector<VALUE_TYPE>> original_data, int ntrees,
                   int starting_index, int total_data_set_size, int donate_per, int transfer_threshold,
                   dmrpt::StorageFormat storage_format,
                   int rank, int world_size, string input_path, string output_path);

        void grow_global_tree();

        void
        grow_global_subtree(vector <vector<DataPoint>> child_data_tracker, vector<int> total_size_vector,int depth,int tree);


        vector <DataPoint> send_receive_data_points_if_zero(vector <DataPoint> data_points, int *total_counts,
                                                            int *process_counts, int *disps, int depth, int direction,
                                                            int tree);

        int detect_max_rank(int *total_counts, int direction);

        int detect_min_rank(int *total_counts, int direction);

        vector <DataPoint>
        send_receive_data_points_if_zero(vector <DataPoint> data_points, int *total_counts, int i, int direction,
                                         int depth, int tree);

        bool is_transfer_needed(int *total_counts, int direction);

        void gather_sibling_indexes();


        vector <DataPoint> collect_similar_data_points_for_given_tree_index(int tree, int index);

        void collect_similar_data_points_for_all_tree_indices(int tree, int index, int depth);

        vector <DataPoint> request_data_points_for_given_index(vector <DataPoint> all_my_points);

        vector <DataPoint> send_data_points_for_requested_node(vector <DataPoint> all_my_points, int sending_rank);

        vector <vector<dmrpt::DataPoint>> calculate_nns(int tree, int nn);

        vector <vector<dmrpt::DataPoint>> gather_nns(int nn);


    };
}

#endif //DISTRIBUTED_MRPT_DRPT_GLOBAL_H
