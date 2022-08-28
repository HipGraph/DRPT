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
    VALUE_TYPE *P = mathOp.multiply_mat(imdataArr, B, rows, global_tree_depth * this->ntrees, cols, 1.0);

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
//            cout<< " creating leaf " <<j<<endl;
            vector <vector<VALUE_TYPE>> local_data(leafs[j].size());
            for (int k = 0; k < leafs[j].size(); k++) {
                local_data[k] = leafs[j][k].image_data;
            }
//            cout<< " data filling complete for  leaf " <<j <<" size "<<local_data.size()<<endl;
            VALUE_TYPE *local_data_arr = mathOp.convert_to_row_major_format(local_data);
//            cout<< " row major version completed " <<j<<endl;

            VALUE_TYPE *LP = mathOp.multiply_mat(local_data_arr, C, this->data_dimension,
                                                 local_tree_depth,
                                                 leafs[j].size(), 1.0);
//            cout<<" creating drpt "<< j <<leafs.size()<<endl;
            DRPT drpt1 = dmrpt::DRPT(LP, C, leafs[j].size(),
                                     local_tree_depth, local_data, 1, starting_index, this->rank, this->world_size);

            drpt1.grow_local_tree();
//            cout<<" creating drpt "<< j <<" tree growing completed"<<endl;

            vector <vector<int>> final_clustered_data = drpt1.get_all_leaf_node_indices(0);
            cout<<" final_clustered_data size for leaf "<<j<<  final_clustered_data.size()<<endl;

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
                cout <<" setting "<<id<<" i "<<i<<" id "<<id<< " datavec "<<this->trees_leaf_all[i][id].size()<<endl;

            }

            free(local_data_arr);
            free(LP);


        }
        free(C);

    }
}

vector <vector<dmrpt::DataPoint>> dmrpt::MDRPT::calculate_nns(int tree, int nn) {

    dmrpt::MathOp mathOp;

    int total_leaf_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

    int leafs_per_node = total_leaf_size / this->world_size;

    int my_start_count = 0;
    int end_count = 0;

    //large trees
    if (total_leaf_size >= this->world_size) {
        my_start_count = leafs_per_node * this->rank;
        if (this->rank < this->world_size - 1) {
            end_count = leafs_per_node * (this->rank + 1);
        } else {
            end_count = total_leaf_size;
        }
    }

    cout << " my start " << my_start_count << " my end " << end_count << "  rank " << rank << endl;
    vector <vector<DataPoint>> final_results(total_data_set_size);

    char results[500];

    char hostname[HOST_NAME_MAX];

    gethostname(hostname, HOST_NAME_MAX);
    string file_path_stat = output_path + "stats_divided.txt.";
    std::strcpy(results, file_path_stat.c_str());
    std::strcpy(results + strlen(file_path_stat.c_str()), hostname);

    ofstream fout(results, std::ios_base::app);

    auto start_distance = high_resolution_clock::now();

    for (int i = my_start_count; i < end_count; i++) {
        vector <DataPoint> data_points = this->trees_leaf_all[tree][i];
        cout <<" obtaining "<<id<<" i "<<tree<<" id "<<i<< " datavec "<<data_points.size()<<endl;

        for (int k = 0; k < data_points.size(); k++) {
            vector <DataPoint> vec(data_points.size());
#pragma omp parallel for
            for (int j = 0; j < data_points.size(); j++) {

                VALUE_TYPE distance = mathOp.calculate_distance(data_points[k].image_data,
                                                                data_points[j].image_data);

                DataPoint dataPoint;
                dataPoint.src_index = data_points[k].index;
                dataPoint.index = data_points[j].index;
                dataPoint.distance = distance;
                vec[j] = dataPoint;

            }


            sort(vec.begin(), vec.end(),
                 [](const DataPoint &lhs, const DataPoint &rhs) {
                     return lhs.distance < rhs.distance;
                 });

            vector <DataPoint> sub_vec;
            if (vec.size() > nn) {
                sub_vec = slice(vec, 0, nn - 1);
            } else {
                sub_vec = vec;
            }
            final_results[vec[0].src_index].insert(final_results[vec[0].src_index].end(), sub_vec.begin(),
                                                   sub_vec.end());
        }
    }
    auto end_distance = high_resolution_clock::now();
    auto distance_time = duration_cast<microseconds>(end_distance - start_distance);

    cout << rank << " distance calc returinng " << endl;
    fout << rank << " distance calc " << distance_time.count() << endl;
    cout << rank << " distance priting  done " << endl;

    return final_results;
}

vector <vector<dmrpt::DataPoint>> dmrpt::MDRPT::gather_nns(int nn) {

    cout << "gathering started " << endl;
    char results[500];

    char hostname[HOST_NAME_MAX];

    gethostname(hostname, HOST_NAME_MAX);
    string file_path_stat = output_path + "stats_divided.txt.";
    std::strcpy(results, file_path_stat.c_str());
    std::strcpy(results + strlen(file_path_stat.c_str()), hostname);

    ofstream fout(results, std::ios_base::app);


    int my_starting_index = this->rank * (this->total_data_set_size / world_size);

    int end_index = 0;
    if (this->rank < this->world_size - 1) {
        end_index = (this->rank + 1) * (this->total_data_set_size / world_size);
    } else {
        end_index = this->total_data_set_size;
    }


    vector <vector<DataPoint>> final_data(this->total_data_set_size);
    vector <vector<DataPoint>> collected_nns(this->total_data_set_size);

    auto start_distance = high_resolution_clock::now();

    for (int i = 0; i < ntrees; i++) {
        vector <vector<DataPoint>> data = this->calculate_nns(i, 2 * nn);

        cout << " rank " << rank << " inserting started" << data.size() << endl;
#pragma omp parallel for
        for (int j = 0; j < total_data_set_size; j++) {
            if (!data[j].empty()) {

                final_data[j].insert(final_data[j].end(), data[j].begin(),
                                     data[j].end());
            }
        }
    }

    cout << " rank " << rank << " distance calculation completed " << endl;

    auto stop_distance = high_resolution_clock::now();
    auto distance_time = duration_cast<microseconds>(stop_distance - start_distance);


    auto start_query = high_resolution_clock::now();

    int chunk_size = this->total_data_set_size / this->world_size;

    int remain = this->total_data_set_size - chunk_size * (this->world_size - 1);


    int count = 0;


    int sending_size = 0;
    int feasible_size = 0;
    while (count < this->total_data_set_size) {

        int sending_rank = -1;
        for (int g = 0; g < this->world_size; g++) {
            if (count >= (g * (this->total_data_set_size / this->world_size)) &&
                count < ((g + 1) * (this->total_data_set_size / this->world_size))) {
                sending_rank = g;
                break;
            }
        }

        feasible_size = chunk_size;

        if (count + remain == this->total_data_set_size) {
            feasible_size = remain;
        }
        sending_size = 0;
//        cout<< " rank "<<this->rank << " current count "<< count + feasible_size<<endl;
        for (int i = count; i < count + feasible_size; i++) {
            if (!final_data[i].empty()) {
                sending_size++;
            }
        }


        int tot_indices_size = sending_size * 2 * nn;
        int *source_indices = new int[sending_size];
        int *nn_indices = new int[tot_indices_size];
        int *process_counts = new int[this->world_size];
        int *process_counts_nns = new int[this->world_size];
        int *my_count = new int[1];
        my_count[0] = sending_size;
        VALUE_TYPE *nn_distances = new VALUE_TYPE[tot_indices_size];
        int co = 0;
        for (int l = count; l < count + feasible_size; l++) {
            if (!final_data[l].empty()) {
                source_indices[co] = final_data[l][0].src_index;
                sort(final_data[l].begin(), final_data[l].end(),
                     [](const DataPoint &lhs, const DataPoint &rhs) {
                         return lhs.distance < rhs.distance;
                     });

                for (int j = 0; j < 2 * nn; j++) {
                    nn_indices[co * 2 * nn + j] = final_data[l][j].index;
                    nn_distances[co * 2 * nn + j] = final_data[l][j].distance;
                }
                co++;
            }
        }

        if (count >= my_starting_index && count < end_index) {

            MPI_Gather(my_count, 1, MPI_INT, process_counts, 1, MPI_INT, this->rank, MPI_COMM_WORLD);


            int *disps = new int[this->world_size];
            int *disps_nns = new int[this->world_size];

            // Displacement for the first chunk of data - 0
            int tot = 0;
            for (int i = 0; i < this->world_size; i++) {
                tot = tot + process_counts[i];
                process_counts_nns[i] = process_counts[i] * 2 * nn;
                disps[i] = (i > 0) ? (disps[i - 1] + process_counts[i - 1]) : 0;
                disps_nns[i] = (i > 0) ? (disps_nns[i - 1] + process_counts_nns[i - 1]) : 0;
            }
            int *total_source_indices = new int[tot];
            int *total_nn_indices = new int[tot * 2 * nn];
            VALUE_TYPE *total_nn_distances = new VALUE_TYPE[tot * 2 * nn];

            MPI_Gatherv(source_indices, sending_size, MPI_INT, total_source_indices, process_counts, disps, MPI_INT,
                        this->rank, MPI_COMM_WORLD);
            MPI_Gatherv(nn_indices, tot_indices_size, MPI_INT, total_nn_indices, process_counts_nns, disps_nns,
                        MPI_INT,
                        this->rank, MPI_COMM_WORLD);
            MPI_Gatherv(nn_distances, tot_indices_size, MPI_VALUE_TYPE, total_nn_distances, process_counts_nns,
                        disps_nns, MPI_VALUE_TYPE, this->rank, MPI_COMM_WORLD);


            for (int m = 0; m < this->world_size; m++) {
                int my_index_start = disps[m];
                int my_start = disps_nns[m];
                for (int h = 0; h < process_counts[m]; h++) {
                    int source = total_source_indices[my_index_start + h];
                    vector <DataPoint> gathred_knns(2 * nn);

                    for (int y = 0; y < 2 * nn; y++) {
                        DataPoint dataPoint;
                        dataPoint.src_index = source;
                        int get_index = my_start + 2 * nn * h + y;
                        dataPoint.index = total_nn_indices[get_index];
                        dataPoint.distance = total_nn_distances[get_index];
                        gathred_knns[y] = dataPoint;
                    }

                    if (collected_nns[source].empty()) {
                        collected_nns[source] = gathred_knns;

                    } else {
                        std::vector <DataPoint> v3;
                        std::merge(collected_nns[source].begin(), collected_nns[source].end(),
                                   gathred_knns.begin(), gathred_knns.end(),
                                   std::back_inserter(v3), [](const DataPoint &lhs, const DataPoint &rhs) {
                                    return lhs.distance < rhs.distance;
                                });


                        collected_nns[source] = v3;
                    }
                    collected_nns[source].erase(unique(collected_nns[source].begin(), collected_nns[source].end(),
                                                       [](const DataPoint &lhs,
                                                          const DataPoint &rhs) {
                                                           return lhs.index == rhs.index;
                                                       }), collected_nns[source].end());


                }
            }

            free(total_source_indices);
            free(total_nn_indices);
            free(total_nn_distances);
            free(disps);
            free(disps_nns);

        } else {

            MPI_Gather(my_count, 1, MPI_INT, NULL, 1, MPI_INT, sending_rank, MPI_COMM_WORLD);
            MPI_Gatherv(source_indices, sending_size, MPI_INT, NULL, NULL, NULL, MPI_INT, sending_rank,
                        MPI_COMM_WORLD);
            MPI_Gatherv(nn_indices, tot_indices_size, MPI_INT, NULL, NULL, NULL, MPI_INT, sending_rank,
                        MPI_COMM_WORLD);
            MPI_Gatherv(nn_distances, tot_indices_size, MPI_VALUE_TYPE, NULL, NULL, NULL, MPI_VALUE_TYPE,
                        sending_rank,
                        MPI_COMM_WORLD);
        }

        free(source_indices);
        free(nn_indices);
        free(nn_distances);
        free(my_count);
        free(process_counts_nns);
        free(process_counts);

//        cout<< " rank "<<this->rank << " completinng count "<< count<<endl;

        count = count + feasible_size;


    }
    auto end_query = high_resolution_clock::now();
    auto query_time = duration_cast<microseconds>(end_query - start_query);

    cout << rank << " distance  " << distance_time.count() << " query " << query_time.count() << endl;

    return collected_nns;
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


//vector <vector<dmrpt::DataPoint>> dmrpt::MDRPT::get_knn(int nn) {
//    return this->drpt_global.gather_nns(nn);
//}





