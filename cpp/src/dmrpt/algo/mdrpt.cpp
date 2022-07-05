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


using namespace std;
using namespace std::chrono;

dmrpt::MDRPT::MDRPT(int ntrees, vector <vector<VALUE_TYPE>> original_data, int tree_depth, int total_data_set_size,
                    dmrpt::StorageFormat storageFormat, int rank, int world_size) {
    this->data_dimension = original_data[0].size();
    this->tree_depth = tree_depth;
    this->original_data = original_data;
    this->total_data_set_size = total_data_set_size;
    this->storageFormat = storageFormat;
    this->rank = rank;
    this->world_size = world_size;
    this->ntrees = ntrees;
}

template<typename T> vector <T> slice(vector < T >const &v,int m,int n) {
   auto first = v.cbegin() + m;
   auto last = v.cbegin() + n + 1;

    std::vector <T> vec(first, last);
   return vec;
}


vector <vector<dmrpt::DRPT::DataPoint>>
dmrpt::MDRPT::get_filtered_results(vector <vector<dmrpt::DRPT::DataPoint>> results, int vote_threshold, int nn) {

    vector <unordered_map<int, int>> voted_results(results.size());
    vector <vector<dmrpt::DRPT::DataPoint>> voted_selected(results.size());
    vector <vector<dmrpt::DRPT::DataPoint>> final_results(results.size());

#pragma omp parallel for
    {
        for (int i = 0; i < results.size(); i++) {

            unordered_map<int, int> voted_results;
            vector<dmrpt::DRPT::DataPoint> voted_selected;

            if (!results[i].empty()) {
//                for (int j = 0; j < results[i].size(); j++) {
//                    cout<<" runing loop"<<rank<<endl;
//                    unordered_map<int, int>::iterator it = voted_results.find(results[i][j].index);
//                    if (it != voted_results.end()) {
//                        ++it->second;// increment map's value for key `c`
//                        if (it->second >= vote_threshold) {
//                            voted_selected.push_back(results[i][j]);
//                        }
//                    } else {
//                        cout<<" running loop inserting "<<rank<<endl;
//                        voted_results.insert(std::make_pair(results[i][j].index, 1));
//                        if (vote_threshold == 1) {
//                            voted_selected.push_back(results[i][j]);
//                        }
//                    }
//                }

                sort(results[i].begin(), results[i].end(),
                     [](const dmrpt::DRPT::DataPoint &lhs, const dmrpt::DRPT::DataPoint &rhs) {
                         return lhs.distance < rhs.distance;
                     });


                results[i].erase(unique(results[i].begin(), results[i].end(),
                                               [](const dmrpt::DRPT::DataPoint &lhs,
                                                  const dmrpt::DRPT::DataPoint &rhs) {
                                                   return lhs.index == rhs.index;
                                               }), results[i].end());

                vector <dmrpt::DRPT::DataPoint> sub_vec = slice(results[i], 0, nn - 1);

                final_results[i].insert(final_results[i].end(), sub_vec.begin(), sub_vec.end());
            }

        }

    }
    return final_results;
}

void dmrpt::MDRPT::grow_trees(float density) {

    dmrpt::MathOp mathOp;
    VALUE_TYPE *imdataArr = mathOp.convert_to_row_major_format(this->original_data);

    int rows = this->original_data[0].size();
    int cols = this->original_data.size();

    VALUE_TYPE *B = mathOp.build_sparse_projection_matrix(this->rank, this->world_size, this->data_dimension,
                                                      this->tree_depth * this->ntrees, density);
    // P= X.R
    VALUE_TYPE *P = mathOp.multiply_mat(imdataArr, B, rows, this->tree_depth * this->ntrees, cols, 1.0);

    int starting_index = this->rank * this->total_data_set_size / world_size;
//    this->drpt = dmrpt::DRPT(P, B, cols, this->tree_depth, this->original_data, this->ntrees, starting_index,
//                                   this->storageFormat, this->rank, this->world_size);
//    this->drpt.grow_local_tree();

    dmrpt::DRPTGlobal global_drpt = dmrpt::DRPTGlobal(P, B, cols, this->tree_depth, this->original_data, this->ntrees, starting_index, this->total_data_set_size,
                                   this->storageFormat, this->rank, this->world_size);
    global_drpt.grow_global_tree();

}


vector <vector<dmrpt::DRPT::DataPoint>>
dmrpt::MDRPT::batch_query(int batch_size, VALUE_TYPE distance_threshold, int vote_threshold, int nn) {
    vector <vector<dmrpt::DRPT::DataPoint>> results(this->original_data.size());
        for (int j = 0; j < this->world_size; j++) {
            auto start = high_resolution_clock::now();
            vector <vector<dmrpt::DRPT::DataPoint>> result = this->drpt.batch_query(this->original_data, batch_size,
                                                                                        j, distance_threshold);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
            cout << "Time taken for batch_query tree  " << j << " duration" <<
                 duration.count() << " microseconds" << endl;

            for (int k = 0; k < result.size(); k++) {
                results[k].insert(results[k].end(), result[k].begin(), result[k].end());
            }
    }


    return this->get_filtered_results(results, vote_threshold, nn);

}





