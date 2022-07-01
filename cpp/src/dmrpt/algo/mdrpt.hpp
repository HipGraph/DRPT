#ifndef DISTRIBUTED_MRPT_MDRPT_H
#define DISTRIBUTED_MRPT_MDRPT_H
#include <cblas.h>
#include <vector>
#include "drpt.hpp"
#include "../math/matrix_multiply.hpp"
#include <mpi.h>
#include <string>
#include <omp.h>

namespace dmrpt {
    class MDRPT {

    private:
        int ntrees;
        int tree_depth;
        int data_dimension;
        dmrpt::StorageFormat storageFormat;
        vector<vector<VALUE_TYPE>> original_data;
        int starting_data_index;
        int rank;
        int world_size;
        int total_data_set_size;
        dmrpt::DRPT drpt;

    public:
        MDRPT(int ntrees, vector<vector<VALUE_TYPE>> original_data,int tree_depth,int total_data_set_size, dmrpt::StorageFormat storageFormat, int rank, int world_size);
        void grow_trees(float density);
        vector<vector<dmrpt::DRPT::DataPoint>> batch_query(int batch_size, VALUE_TYPE distance_threshold, int vote_threshold, int nn);
        vector<vector<dmrpt::DRPT::DataPoint>> get_filtered_results(vector<vector<dmrpt::DRPT::DataPoint>> results, int vote_threshold, int nn);

    };
}


#endif //DISTRIBUTED_MRPT_MDRPT_H
