#ifndef DISTRIBUTED_MRPT_MRPT_H
#define DISTRIBUTED_MRPT_MRPT_H

#include <cblas.h>
#include <vector>
#include "drpt.hpp"
#include "../math/matrix_multiply.hpp"
#include <mpi.h>
#include <string>
#include <omp.h>


namespace dmrpt {
    struct DataPoint {
        int src_index;
        int index;
        VALUE_TYPE distance;
        VALUE_TYPE value;
        vector<VALUE_TYPE> image_data;
    };


    class DRPT {

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
        dmrpt::StorageFormat storageFormat;
        vector<vector<VALUE_TYPE>> data;
        vector<VALUE_TYPE> splits;
        vector<int> indices;
        vector<std::vector<int>> leaf_first_indices_all;
        vector<int> leaf_first_indices;

        //multiple trees
        vector<vector<vector<VALUE_TYPE>>> trees_data;
        vector<vector<VALUE_TYPE>> trees_splits;
        vector<vector<int>> trees_indices;
        vector<vector<vector<int>>> trees_leaf_first_indices_all;
        vector<vector<int>> trees_leaf_first_indices;
        vector<vector<VALUE_TYPE>> original_data;


    public:
        DRPT();
        DRPT(VALUE_TYPE *projected_matrix, VALUE_TYPE *projection_matrix, int no_of_data_points, int tree_depth,
             vector <vector<VALUE_TYPE>> original_data, int ntrees,
             int starting_index,  int rank, int world_size);

        void grow_local_tree();

        void grow_local_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                                int depth, int i, int tree);

        vector<vector<int>> get_all_leaf_node_indices(int tree);


        vector<vector<int>> query(VALUE_TYPE *queryP, int no_datapoints, dmrpt::StorageFormat storageFormat);

        vector<vector<dmrpt::DataPoint>> batch_query(vector <vector<VALUE_TYPE>> queries,  int batch_size, int initialRank,VALUE_TYPE distance_threshold);

        void count_leaf_sizes(int datasize, int level,  int depth,std::vector<int> &out_leaf_sizes);

        void count_first_leaf_indices(std::vector<int> &indices, int datasize, int depth);

        void count_first_leaf_indices_all(std::vector<std::vector<int>> &indices, int datasize, int depth_max);

        vector<vector<dmrpt::DataPoint>> send_query_and_receive_results(vector <vector<VALUE_TYPE>> query_batch, int batch_size,
                                                                       int query_dimension, VALUE_TYPE distance_threshold);

        void receive_queries_and_evaluate_results(int sendingRank, int query_dimension,VALUE_TYPE distance_threshold);

    };
}


#endif //DISTRIBUTED_MRPT_MRPT_H
