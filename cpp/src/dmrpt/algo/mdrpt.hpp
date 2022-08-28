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
namespace dmrpt {
    class MDRPT {

    private:
        int ntrees;
        int tree_depth;
        double tree_depth_ratio;
        int data_dimension;
        dmrpt::StorageFormat storage_format;
        vector<vector<VALUE_TYPE>> original_data;
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


    public:
        MDRPT(int ntrees, int algo,vector<vector<VALUE_TYPE>> original_data,int tree_depth,double tree_depth_ratio, int total_data_set_size,int donate_per, int transfer_threshold,
              dmrpt::StorageFormat storageFormat, int rank, int world_size, string input_path,string output_path);
        void grow_trees(float density);
        vector<vector<dmrpt::DataPoint>> batch_query(int batch_size, VALUE_TYPE distance_threshold, int nn);
        vector<vector<dmrpt::DataPoint>> get_filtered_results(vector<vector<dmrpt::DataPoint>> results, int nn);
        vector<vector<dmrpt::DataPoint>> get_knn(int nn);



    };
}


#endif //DISTRIBUTED_MRPT_MDRPT_H
