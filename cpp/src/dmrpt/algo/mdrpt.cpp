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
#include <cstring>


using namespace std;
using namespace std::chrono;

dmrpt::MDRPT::MDRPT(int ntrees, int algo, vector <vector<VALUE_TYPE>> original_data, int tree_depth,
                    double tree_depth_ratio,
                    int total_data_set_size,
                    int rank, int world_size, string input_path, string output_path) {
    this->data_dimension = original_data[0].size();
    this->tree_depth = tree_depth;
    this->original_data = original_data;
    this->total_data_set_size = total_data_set_size;
    this->rank = rank;
    this->world_size = world_size;
    this->ntrees = ntrees;
    this->algo = algo;
    this->input_path = input_path;
    this->output_path = output_path;
    this->tree_depth_ratio = tree_depth_ratio;
    this->trees_leaf_all = vector < vector < vector < DataPoint > >>(ntrees);
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

template<typename T> bool allEqual(std::vector < T >
const &v) {
return
std::adjacent_find(v
.

begin(), v

.

end(), std::not_equal_to<T>()

) == v.

end();

}

void dmrpt::MDRPT::grow_trees(float density) {

    dmrpt::MathOp mathOp;
    VALUE_TYPE *imdataArr = mathOp.convert_to_row_major_format(this->original_data);

    int rows = this->original_data[0].size();
    int cols = this->original_data.size();

    int global_tree_depth = this->tree_depth * this->tree_depth_ratio;
    int local_tree_depth = this->tree_depth - global_tree_depth;


    VALUE_TYPE *B = mathOp.build_sparse_projection_matrix(this->rank, this->world_size, this->data_dimension,
                                                          global_tree_depth * this->ntrees, density);

    char results[500];
    char hostname[HOST_NAME_MAX];
    int host = gethostname(hostname, HOST_NAME_MAX);
    string file_path_stat = output_path + "stats_divided.txt.";
    std::strcpy(results, file_path_stat.c_str());
    std::strcpy(results + strlen(file_path_stat.c_str()), hostname);

    ofstream fout(results, std::ios_base::app);

    char data[500];
    string file_path_data = output_path + "data.txt.";
    std::strcpy(data, file_path_data.c_str());
    std::strcpy(data + strlen(file_path_data.c_str()), hostname);

    ofstream fout1(data, std::ios_base::app);


    auto start_matrix_index = high_resolution_clock::now();
    // P= X.R
    VALUE_TYPE *P = mathOp.multiply_mat(imdataArr, B, rows, this->tree_depth * this->ntrees, cols, 1.0);

    auto stop_matrix_index = high_resolution_clock::now();

    auto matrix_time = duration_cast<microseconds>(stop_matrix_index - start_matrix_index);

    int starting_index = (this->total_data_set_size / world_size) * this->rank;


    this->drpt_global = dmrpt::DRPTGlobal(P, B, cols, global_tree_depth, this->original_data, this->ntrees,
                                          starting_index,
                                          this->total_data_set_size, this->rank, this->world_size, input_path,
                                          output_path);


    auto start_grow_index = high_resolution_clock::now();
    cout << " rank " << rank << " starting growing trees" << endl;
    this->drpt_global.grow_global_tree();
    auto stop_grow_index = high_resolution_clock::now();
    auto index_time = duration_cast<microseconds>(stop_grow_index - start_grow_index);

    cout << " rank " << rank << " completing growing trees" << endl;

    cout << " rank " << rank << " running  datapoint collection " << endl;


    auto start_collect = high_resolution_clock::now();

    vector < vector < vector < DataPoint>>> leaf_nodes_of_trees(ntrees);
    int total_child_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

    for (int i = 0; i < ntrees; i++) {
        this->trees_leaf_all[i] = vector < vector < dmrpt::DataPoint >> (total_child_size);
        leaf_nodes_of_trees[i] = this->drpt_global.collect_similar_data_points(i);
    }
    auto stop_collect = high_resolution_clock::now();
    auto collect_time = duration_cast<microseconds>(stop_collect - start_collect);


    fout << rank << " matrix  " << matrix_time.count() << " tree " << index_time.count() << " collecting "
         << collect_time.count()
         << endl;
    cout << " rank " << rank << " similar datapoint collection completed" << endl;


    int total_leaf_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

    int leafs_per_node = total_leaf_size / this->world_size;

    cout << " leafs per node " << leafs_per_node << endl;

    int my_start_count = 0;
    int my_end_count = 0;

    //large trees
    if (total_leaf_size >= this->world_size) {
        my_start_count = leafs_per_node * this->rank;
        if (this->rank < this->world_size - 1) {
            my_end_count = leafs_per_node * (this->rank + 1);
        } else {
            my_end_count = total_leaf_size;
        }
    }

    cout << " start count " << my_start_count << " end count " << my_end_count << endl;

    for (int i = 0; i < ntrees; i++) {
        vector <vector<DataPoint>> leafs = leaf_nodes_of_trees[i];
        VALUE_TYPE *C = mathOp.build_sparse_projection_matrix(this->rank, this->world_size, this->data_dimension,
                                                              local_tree_depth , density);

        cout<<" tree "<<i<< " projection matrix completed and leafs size "<<leafs.size()<<endl;

        for (int j = 0; j < leafs.size(); j++) {
            cout<< " creating leaf " <<j<<endl;
            vector <vector<VALUE_TYPE>> local_data(leafs[j].size());
            for (int k = 0; k < leafs[j].size(); k++) {
                local_data[k] = leafs[j][k].image_data;
            }
            cout<< " data filling complete for  leaf " <<j <<" size "<<local_data.size()<<endl;
            VALUE_TYPE *local_data_arr = mathOp.convert_to_row_major_format(local_data);
            cout<< " row major version completed " <<j<<endl;

            VALUE_TYPE *LP = mathOp.multiply_mat(local_data_arr, C, this->data_dimension,
                                                 local_tree_depth,
                                                 leafs[j].size(), 1.0);
            cout<<" creating drpt "<< j <<leafs.size()<<endl;
            DRPT drpt1 = dmrpt::DRPT(LP, C, leafs[j].size(),
                                     local_tree_depth, local_data, 1, starting_index,
                                     this->storage_format, this->rank, this->world_size);

            drpt1.grow_local_tree();
            cout<<" creating drpt "<< j <<" tree growing completed"<<endl;

            vector <vector<int>> final_clustered_data = drpt1.get_all_leaf_node_indices(i);

            for (int l = 0; l < final_clustered_data.size(); l++) {
                vector <DataPoint> data_vec(final_clustered_data[l].size());
                for (int m = 0; m < final_clustered_data[l].size(); m++) {
                    int index = final_clustered_data[l][m];
                    std::vector<DataPoint>::iterator it = std::find_if(leafs[j].begin(),
                                                                       leafs[j].end(),
                                                                       [index](DataPoint const &n) {
                                                                           return n.index == index;
                                                                       });
                    data_vec.push_back(*it);

                }
                int id = my_start_count + l;
                this->trees_leaf_all[i][id] = data_vec;

            }

            free(local_data_arr);
            free(LP);

        }
        free(C);

    }
}


vector <vector<dmrpt::DataPoint>>
dmrpt::MDRPT::batch_query(int batch_size, VALUE_TYPE distance_threshold, int nn) {
    vector <vector<dmrpt::DataPoint>> results(this->original_data.size());
    cout << " rank " << this->rank << " runnnig batch query" << endl;
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





