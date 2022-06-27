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
    class DRPT {


    private:
        int tree_depth;
        double *projected_matrix;
        double *projection_matrix;
        int rows;
        int cols;
        dmrpt::StorageFormat storageFormat;
        vector<vector<double>> data;
        vector<double> splits;
        vector<int> indices;
        vector<std::vector<int>> leaf_first_indices_all;
        vector<int> leaf_first_indices;
        vector<vector<double>> original_data;
        int starting_data_index;
        int rank;
        int world_size;
    public:

        struct DataPoint {
            int index;
            double distance;
        };
        DRPT();
        DRPT(double *projected_matrix, double *projection_matrix, int rows, int cols, vector<vector<double>> original_data,
             int starting_index, dmrpt::StorageFormat storageFormat, int rank, int world_size);

        void grow_local_tree();

        void grow_local_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                                int depth, int i);

        vector<vector<int>> query(double *queryP, int no_datapoints, dmrpt::StorageFormat storageFormat);

        vector<vector<DRPT::DataPoint>> batch_query(vector <vector<double>> queries,  int batch_size, int initialRank,double distance_threshold);

        void count_leaf_sizes(int datasize, int level,  int depth,std::vector<int> &out_leaf_sizes);

        void count_first_leaf_indices(std::vector<int> &indices, int datasize, int depth);

        void count_first_leaf_indices_all(std::vector<std::vector<int>> &indices, int datasize, int depth_max);

        vector<vector<DRPT::DataPoint>> send_query_and_receive_results(vector <vector<double>> query_batch, int batch_size,
                                                                       int query_dimension, double distance_threshold);

        void receive_queries_and_evaluate_results(int sendingRank, int query_dimension,double distance_threshold);

    };
}


#endif //DISTRIBUTED_MRPT_MRPT_H
