#include "mdrpt.hpp"
#include <cblas.h>
#include <stdio.h>
#include "drpt.hpp"
#include "../math/matrix_multiply.hpp"
#include <vector>
#include <random>
#include <mpi.h>
#include <string>
#include <iostream>
#include <omp.h>
#include <map>
#include <unordered_map>
#include <fstream>
#include "../algo/drpt_global.hpp"
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <limits.h>


using namespace std;
using namespace std::chrono;

dmrpt::MDRPT::MDRPT(int ntrees, int algo, vector <vector<VALUE_TYPE>> original_data, int tree_depth,
                    int total_data_set_size,
                    int donate_per,
                    int transfer_threshold,
                    dmrpt::StorageFormat storage_format, int rank, int world_size, string input_path, string output_path) {
    this->data_dimension = original_data[0].size();
    this->tree_depth = tree_depth;
    this->original_data = original_data;
    this->total_data_set_size = total_data_set_size;
    this->storage_format = storage_format;
    this->donate_per = donate_per;
    this->rank = rank;
    this->world_size = world_size;
    this->ntrees = ntrees;
    this->transfer_threshold = transfer_threshold;
    this->algo = algo;
    this->input_path = input_path;
    this->output_path = output_path;
}

template<typename T> vector <T> slice(vector < T >
const &v,
int m,
int n
) {
auto first = v.cbegin() + m;
auto last = v.cbegin() + n + 1;

std::vector <T> vec(first, last);
return
vec;
}


vector <vector<dmrpt::DataPoint>>
dmrpt::MDRPT::get_filtered_results(vector <vector<dmrpt::DataPoint>> results, int nn) {

    vector <vector<dmrpt::DataPoint>> final_results(results.size());

#pragma omp parallel for
//    {
        for (int i = 0; i < results.size(); i++) {

            if (!results[i].empty()) {

                sort(results[i].begin(), results[i].end(),
                     [](const dmrpt::DataPoint &lhs, const dmrpt::DataPoint &rhs) {
                         return lhs.distance < rhs.distance;
                     });


                results[i].erase(unique(results[i].begin(), results[i].end(),
                                        [](const dmrpt::DataPoint &lhs,
                                           const dmrpt::DataPoint &rhs) {
                                            return lhs.index == rhs.index;
                                        }), results[i].end());

                vector <dmrpt::DataPoint> sub_vec = slice(results[i], 0, nn - 1);

                final_results[i].insert(final_results[i].end(), sub_vec.begin(), sub_vec.end());
            }

        }

//    }
    return final_results;
}

void dmrpt::MDRPT::grow_trees(float density) {

    dmrpt::MathOp mathOp;
    VALUE_TYPE *imdataArr = mathOp.convert_to_row_major_format(this->original_data);

    int rows = this->original_data[0].size();
    int cols = this->original_data.size();

    VALUE_TYPE *B = mathOp.build_sparse_projection_matrix(this->rank, this->world_size, this->data_dimension,
                                                          this->tree_depth * this->ntrees, density);

    char results[500];
    char hostname[HOST_NAME_MAX];
    int host = gethostname(hostname, HOST_NAME_MAX);
    string file_path_stat = output_path + "stats_divided.txt.";
    std::sprintf(results, file_path_stat.c_str());
    std::strcpy(results, hostname);

    ofstream fout(results, std::ios_base::app);

    auto start_matrix_index = high_resolution_clock::now();
    // P= X.R
    VALUE_TYPE *P = mathOp.multiply_mat(imdataArr, B, rows, this->tree_depth * this->ntrees, cols, 1.0);

    auto stop_matrix_index = high_resolution_clock::now();

    auto matrix_time = duration_cast<microseconds>(stop_matrix_index - start_matrix_index);

    int starting_index = this->rank * this->total_data_set_size / world_size;
    if (algo == 0) {
        this->drpt = dmrpt::DRPT(P, B, cols, this->tree_depth, this->original_data, this->ntrees, starting_index,
                                 this->storage_format, this->rank, this->world_size);
        this->drpt.grow_local_tree();
        cout << " rank " << rank << " completing growing trees" << endl;
    } else {

        this->drpt_global = dmrpt::DRPTGlobal(P, B, cols, this->tree_depth, this->original_data, this->ntrees,
                                              starting_index, this->total_data_set_size, this->donate_per,
                                              this->transfer_threshold, this->storage_format, this->rank,
                                              this->world_size,input_path,output_path);


        auto start_grow_index = high_resolution_clock::now();
        this->drpt_global.grow_global_tree();
        auto stop_grow_index = high_resolution_clock::now();
        auto index_time = duration_cast<microseconds>(stop_grow_index - start_grow_index);

        cout << " rank " << rank << " completing growing trees" << endl;

        cout << " rank " << rank << " running  datapoint collection " << endl;


        auto start_collect = high_resolution_clock::now();
        for (int i = 0; i < ntrees; i++) {
            this->drpt_global.collect_similar_data_points_for_all_tree_indices(i, 0, 0);
        }
        auto stop_collect = high_resolution_clock::now();
        auto collect_time = duration_cast<microseconds>(stop_collect - start_collect);


        fout << rank << " matrix  " << matrix_time.count() << " tree " << index_time.count() << " collecting " << collect_time.count()
             << endl;
        cout << " rank " << rank << " similar datapoint collection completed" << endl;
    }

}


vector <vector<dmrpt::DataPoint>>
dmrpt::MDRPT::batch_query(int batch_size, VALUE_TYPE distance_threshold,int nn) {
    vector <vector<dmrpt::DataPoint>> results(this->original_data.size());
    cout<<" rank "<< this->rank<< " runnnig batch query" <<endl;
    for (int j = 0; j < this->world_size; j++) {
        auto start = high_resolution_clock::now();
        vector <vector<dmrpt::DataPoint>> result = this->drpt.batch_query(this->original_data, batch_size,
                                                                                j, distance_threshold);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Time taken for batch_query tree  " << j << " duration" <<
             duration.count() << " microseconds" << endl;

        for (int k = 0; k < result.size(); k++) {
            results[k].insert(results[k].end(), result[k].begin(), result[k].end());
        }
    }


    return this->get_filtered_results(results, nn);

}


vector <vector<dmrpt::DataPoint>> dmrpt::MDRPT::get_knn(int nn) {
    return this->drpt_global.gather_nns(nn);

}





