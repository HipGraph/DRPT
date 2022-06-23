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
        vector<vector<double>> original_data;
        int starting_data_index;
        int rank;
        int world_size;
        vector<dmrpt::DRPT> trees;

    public:
        MDRPT(int ntrees, vector<vector<double>> original_data,int tree_depth,dmrpt::StorageFormat storageFormat, int rank, int world_size);
        void grow_trees(float density);
        vector<vector<dmrpt::DRPT::DataPoint>> batchQuery(int batch_size, double distance_threshold, int nn);
        vector<vector<dmrpt::DRPT::DataPoint>> get_vote_results(vector<vector<dmrpt::DRPT::DataPoint>> results, int vote_threshold, int nn);

    };
}


#endif //DISTRIBUTED_MRPT_MDRPT_H
