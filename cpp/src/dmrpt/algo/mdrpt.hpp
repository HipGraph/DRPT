#ifndef DISTRIBUTED_MRPT_MDRPT_H
#define DISTRIBUTED_MRPT_MDRPT_H

#include <cblas.h>
#include <vector>
#include "drpt.hpp"
#include "../math/matrix_multiply.hpp"
#include <mpi.h>
#include <string>
#include <omp.h>
#include "drpt_global.hpp"
#include<map>

namespace dmrpt {
    class MDRPT {

    private:
        int ntrees;
        int tree_depth;
        double tree_depth_ratio;
        int data_dimension;
        dmrpt::StorageFormat storage_format;
        vector <vector<VALUE_TYPE>> original_data;
        int starting_data_index;
        int donate_per;
        int rank;
        int world_size;
        int total_data_set_size;
        int transfer_threshold;
        dmrpt::DRPT drpt;
        dmrpt::DRPTGlobal drpt_global;
        int algo;
        string input_path;
        string output_path;
        vector <vector<vector < DataPoint>>>
        trees_leaf_all;

    private:
        std::map<int, vector<dmrpt::DataPoint>>
        communicate_nns(std::map<int, vector<dmrpt::DataPoint> > &local_nns, int nn);

        void calculate_nns(std::map<int, vector<dmrpt::DataPoint> > &local_nns, int tree, int nn);


    public:
        MDRPT(int ntrees, int algo, vector <vector<VALUE_TYPE>> original_data, int tree_depth, double tree_depth_ratio,
              int total_data_set_size, int rank, int world_size, string input_path, string output_path);

        void grow_trees(float density,bool use_locality_optimization);

        std::map<int, vector<DataPoint>> gather_nns(int nn);


    };
}


#endif //DISTRIBUTED_MRPT_MDRPT_H
