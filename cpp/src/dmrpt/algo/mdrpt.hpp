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
#include <set>
#include "../utils/drpt_timer.h"

namespace dmrpt {
    class MDRPT {

    private:
        int ntrees;
        int tree_depth;
        double tree_depth_ratio;
        int data_dimension;
        int starting_data_index;
        int rank;
        int world_size;
        int global_data_set_size;
        int local_data_set_size;
        string input_path;
        string output_path;
        vector <vector<vector < DataPoint>>>trees_leaf_all;
        vector<set<int>> index_distribution;
        int local_tree_offset;


    private:
        std::map<int, vector<dmrpt::DataPoint>>
        communicate_nns(std::map<int, vector<dmrpt::DataPoint> > &local_nns,set<int> &keys, int nn);

        void calculate_nns(std::map<int, vector<dmrpt::DataPoint> > &local_nns, set<int> &keys, int tree, int nn);


    public:
        MDRPT(int ntrees, int tree_depth, double tree_depth_ratio, int local_tree_offset,
              int total_data_set_size,int local_data_set_size, int dimension, int rank, int world_size, string input_path, string output_path);

        void grow_trees(vector <vector<VALUE_TYPE>> &original_data, float density,
                        bool use_locality_optimization, int nn, ofstream  &fout);

        std::map<int, vector<DataPoint>> gather_nns(int nn, ofstream  &fout);


    };
}


#endif //DISTRIBUTED_MRPT_MDRPT_H
