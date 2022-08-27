#include "drpt_global.hpp"
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
#include <fstream>
#include <map>
#include <unordered_map>
#include "mdrpt.hpp"
#include <chrono>
#include <algorithm>
#include <limits.h>
#include <unistd.h>
#include <cstring>


using namespace std;
using namespace std::chrono;

dmrpt::DRPTGlobal::DRPTGlobal() {

}

dmrpt::DRPTGlobal::DRPTGlobal(VALUE_TYPE *projected_matrix, VALUE_TYPE *projection_matrix, int no_of_data_points,
                              int tree_depth,
                              vector <vector<VALUE_TYPE>> original_data, int ntrees,
                              int starting_index, int total_data_set_size, int donate_per, int transfer_threshold,
                              dmrpt::StorageFormat storage_format, int rank, int world_size, string input_path,
                              string output_path) {
    this->tree_depth = tree_depth;
    this->intial_no_of_data_points = no_of_data_points;
    this->storage_format = storage_format;
    this->projected_matrix = projected_matrix;
    this->projection_matrix = projection_matrix;
    this->total_data_set_size = total_data_set_size;
    this->data_dimension = original_data[0].size();
    this->transfer_threshold = transfer_threshold;


    this->ntrees = ntrees;

    this->trees_data = vector < vector < vector < DataPoint>>>(ntrees);
    this->trees_splits = vector < vector < VALUE_TYPE >> (ntrees);
    this->trees_leaf_first_indices = vector < vector < vector < DataPoint > >> (ntrees);
    this->trees_leaf_first_indices_all = vector < vector < vector < DataPoint > >> (ntrees);

    this->starting_data_index = starting_index;
    this->rank = rank;
    this->world_size = world_size;
    this->leaf_data = vector < vector < DataPoint >> (this->ntrees);
    this->donate_per = donate_per;
    this->original_data_processed = vector<ImageDataPoint>(intial_no_of_data_points);


    this->input_path = input_path;
    this->output_path = output_path;


#pragma omp parallel for
    for (int i = 0; i < this->intial_no_of_data_points; i++) {
        ImageDataPoint imageDataPoint;
        imageDataPoint.index = i + this->starting_data_index;
        imageDataPoint.value = original_data[i];
        this->original_data_processed[i] = imageDataPoint;
    }
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


void dmrpt::DRPTGlobal::grow_global_tree() {
    if (this->tree_depth <= 0 || this->tree_depth > log2(this->intial_no_of_data_points)) {
        throw std::out_of_range(" depth should be in range [1,....,log2(rows)]");
    }

    if (this->ntrees <= 0) {
        throw std::out_of_range(" no of trees should be greater than zero");
    }

    int total_split_size = 1 << (this->tree_depth + 1);
    int total_child_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

//    cout << " tree size " << total_child_size << " tree depth" << tree_depth << " rank " << this->rank << endl;

//    //TODO:remove
//    for (int i = 0; i <  this->original_data_processed.size(); i++) {
//        if (allEqual(this->original_data_processed[i].value) || this->original_data_processed[i].value.size() == 0) {
//            cout << "  original data zero for index ######" << this->original_data_processed[i].index << endl;
//        }
//    }



    if (dmrpt::StorageFormat::RAW == storage_format) {

        for (int k = 0; k < this->ntrees; k++) {
            this->trees_splits[k] = vector<VALUE_TYPE>(total_split_size);
            this->trees_data[k] = vector < vector < DataPoint >> (this->tree_depth);
            this->trees_leaf_first_indices[k] = vector < vector < DataPoint >> (total_child_size);
            this->trees_leaf_first_indices_all[k] = vector < vector < dmrpt::DataPoint >> (total_child_size);

            for (int i = 0; i < this->tree_depth; i++) {
                this->trees_data[k][i] = vector<DataPoint>(this->intial_no_of_data_points);
#pragma  omp parallel for
//                {
                for (int j = 0; j < this->intial_no_of_data_points; j++) {
                    int index = this->tree_depth * k + i + j * this->tree_depth * this->ntrees;
                    DataPoint dataPoint;
                    dataPoint.value = this->projected_matrix[index];
                    dataPoint.index = j + this->starting_data_index;
                    dataPoint.image_data = this->original_data_processed[j].value;
                    this->trees_data[k][i][j] = dataPoint;
                }
//                }
            }


            vector <vector<DataPoint>> child_data_tracker(total_split_size);
            vector<int> total_size_vector(total_split_size);
            child_data_tracker[0] = this->trees_data[k][0];
            total_size_vector[0] = this->total_data_set_size;
            for (int i = 0; i < this->tree_depth - 1; i++) {
                cout << " working on tree depth" << i << endl;
                this->grow_global_subtree(child_data_tracker, total_size_vector, i, k);
            }
        }
    }
}


void
dmrpt::DRPTGlobal::grow_global_subtree(vector <vector<DataPoint>> &child_data_tracker, vector<int> &total_size_vector,
                                       int depth, int tree) {

    int current_nodes = (1 << (depth));
    int number_of_childs = (1 << (depth + 1));
    int split_starting_index = (1 << (depth)) - 1;
    int next_split = (1 << (depth + 1)) - 1;

    if (depth == 0) {
        split_starting_index = 0;
    }

    MathOp mathOp;
    for (int i = 0; i < current_nodes; i++) {
        vector <DataPoint> data_vector = child_data_tracker[split_starting_index + i];
        int data_vec_size = data_vector.size();
        VALUE_TYPE *data = new VALUE_TYPE[data_vec_size];

#pragma omp parallel for
        for (int j = 0; j < data_vector.size(); j++) {
            data[j] = data_vector[j].value;
        }

        int no_of_bins = 1 + (3.322 * log2(data_vec_size));


        VALUE_TYPE *result = mathOp.distributed_median(data, data_vec_size, 1,
                                                       total_size_vector[split_starting_index + i],
                                                       28, dmrpt::StorageFormat::RAW, this->rank);

        VALUE_TYPE median = result[0];

        this->trees_splits[tree][split_starting_index + i] = median;


        vector <DataPoint> left_childs_global;
        vector <DataPoint> right_childs_global;
#pragma omp parallel
        {
            vector <DataPoint> left_childs;
            vector <DataPoint> right_childs;
#pragma omp for  nowait
            for (int k = 0; k < data_vector.size(); k++) {
                int index = data_vector[k].index;
                std::vector<DataPoint>::iterator it = std::find_if(this->trees_data[tree][depth + 1].begin(),
                                                                   this->trees_data[tree][depth + 1].end(),
                                                                   [index](DataPoint const &n) {
                                                                       return n.index == index;
                                                                   });
                if (data_vector[k].value <= median) {
                    left_childs.push_back(*it);
                } else {
                    right_childs.push_back(*it);
                }
            }
#pragma omp critical
            {
                left_childs_global.insert(left_childs_global.end(), left_childs.begin(), left_childs.end());
                right_childs_global.insert(right_childs_global.end(), right_childs.begin(), right_childs.end());

            }
        }


        int left_index = (next_split + 2 * i);
        int right_index = left_index + 1;
        child_data_tracker[left_index] = left_childs_global;
        child_data_tracker[right_index] = right_childs_global;
        if (depth == this->tree_depth - 2) {
            int selected_leaf_left = left_index - (1 << (this->tree_depth - 1)) + 1;
            this->trees_leaf_first_indices[tree][selected_leaf_left] = left_childs_global;
            this->trees_leaf_first_indices[tree][selected_leaf_left + 1] = right_childs_global;
        }
        free(data);
    }

    if (depth == this->tree_depth - 2) {
        return;
    }


// Displacements in the receive buffer for MPI_GATHERV
    int *disps = new int[this->world_size];

// Displacement for the first chunk of data - 0
    for (int i = 0; i < this->world_size; i++) {
        disps[i] = (i > 0) ? (disps[i - 1] + 2 * current_nodes) : 0;
    }

    int *total_counts = new int[2 * this->world_size * current_nodes];

    int *process_counts = new int[this->world_size];

    for (int k = 0; k < this->world_size; k++) {
        process_counts[k] = 2 * current_nodes;
    }
    for (int j = 0; j < current_nodes; j++) {
        int id = (next_split + 2 * j);
        total_counts[2 * j + this->rank * current_nodes * 2] = child_data_tracker[id].size();
        total_counts[2 * j + 1 + this->rank * current_nodes * 2] = child_data_tracker[id + 1].size();
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_INT, total_counts, process_counts, disps, MPI_INT, MPI_COMM_WORLD);

//    left_childs_global = this->send_receive_data_points_if_zero(left_childs_global, total_counts, process_counts,
//                                                                disps,
//                                                                depth, 0, tree);
//    right_childs_global = this->send_receive_data_points_if_zero(right_childs_global, total_counts, process_counts,
//                                                                 disps,
//                                                                 depth, 1, tree);


    for (int j = 0; j < current_nodes; j++) {
        int left_totol = 0;
        int right_total = 0;
        int id = (next_split + 2 * j);
        for (int k = 0; k < this->world_size; k++) {

            left_totol = left_totol + total_counts[2 * j + k * current_nodes * 2];
            right_total = right_total + total_counts[2 * j + 1 + k * current_nodes * 2];
        }

        total_size_vector[id] = left_totol;
        total_size_vector[id + 1] = right_total;
    }

    free(process_counts);
    free(total_counts);
    free(disps);
}


vector <vector<dmrpt::DataPoint>>
dmrpt::DRPTGlobal::collect_similar_data_points(int tree) {

    dmrpt::MathOp mathOp;

    int total_leaf_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

    int leafs_per_node = total_leaf_size / this->world_size;
    int my_start_count = 0;
    int end_count = 0;
    int sending_rank = -1;

    int *send_counts = new int[total_leaf_size];
    int *recv_counts = new int[total_leaf_size];

    cout << "total leaf size" << total_leaf_size << endl;

    int sum_per_node = 0;
    int process = 0;
    int *send_indices_count = new int[this->world_size];
    int *disps_indices_count = new int[this->world_size];

    int *send_values_count = new int[this->world_size];
    int *disps_values_count = new int[this->world_size];


    int my_total = 0;
    for (int i = 0; i < total_leaf_size; i++) {
        if (i > 0 && i % leafs_per_node == 0) {
            send_indices_count[process] = sum_per_node;
            send_values_count[process] = sum_per_node * this->data_dimension;
            sum_per_node = 0;
            process++;
        }
        vector <DataPoint> all_points = this->trees_leaf_first_indices[tree][i];
        send_counts[i] = all_points.size();
        sum_per_node += send_counts[i];
        my_total += send_counts[i];
    }

    send_indices_count[process] = sum_per_node;
    send_values_count[process] = sum_per_node * this->data_dimension;


    MPI_Alltoall(send_counts, leafs_per_node, MPI_INT, recv_counts, leafs_per_node,
                 MPI_INT, MPI_COMM_WORLD);


    int *total_leaf_count = new int[leafs_per_node];

    int *recev_indices_count = new int[this->world_size];
    int *recev_values_count = new int[this->world_size];
    int *recev_disps_count = new int[this->world_size];
    int *recev_disps_values_count = new int[this->world_size];


    int total_sum = 0;
    for (int j = 0; j < leafs_per_node; j++) {
        int count = 0;
        for (int i = 0; i < this->world_size; i++) {
            count += recv_counts[j + i * leafs_per_node];
        }
        total_leaf_count[j] = count;
        total_sum += count;
    }

    for (int i = 0; i < this->world_size; i++) {
        int count = 0;
        for (int j = 0; j < leafs_per_node; j++) {
            count += recv_counts[j + i * leafs_per_node];
        }
        recev_indices_count[i] = count;
        recev_values_count[i] = count * this->data_dimension;

    }

    for (int i = 0; i < this->world_size; i++) {
        disps_indices_count[i] = (i > 0) ? (disps_indices_count[i - 1] + send_indices_count[i - 1]) : 0;
        recev_disps_count[i] = (i > 0) ? (recev_disps_count[i - 1] + recev_indices_count[i - 1]) : 0;
        disps_values_count[i] = (i > 0) ? (disps_values_count[i - 1] + send_values_count[i - 1]) : 0;
        recev_disps_values_count[i] = (i > 0) ? (recev_disps_values_count[i - 1] + recev_values_count[i - 1]) : 0;
    }


    cout << " total sum" << total_sum << endl;


    int *receive_indices = new int[total_sum];

    VALUE_TYPE *receive_values = new VALUE_TYPE[total_sum * this->data_dimension];

    int *send_indices = new int[my_total];
    VALUE_TYPE *send_values = new VALUE_TYPE[my_total * this->data_dimension];


    int co = 0;
    for (int i = 0; i < total_leaf_size; i++) {
        vector <DataPoint> all_points = this->trees_leaf_first_indices[tree][i];
        for (int j = 0; j < all_points.size(); j++) {
            send_indices[co] = all_points[j].index;
#pragma omp parallel for
            for (int k = 0; k < this->data_dimension; k++) {
                send_values[co*this->data_dimension + k] = all_points[j].image_data[k];
            }
            co++;
        }
    }


    MPI_Alltoallv(send_indices, send_indices_count, disps_indices_count, MPI_INT, receive_indices,
                  recev_indices_count, recev_disps_count, MPI_INT, MPI_COMM_WORLD);

    MPI_Alltoallv(send_values, send_values_count, disps_values_count, MPI_VALUE_TYPE, receive_values,
                  recev_values_count, recev_disps_values_count, MPI_VALUE_TYPE, MPI_COMM_WORLD);

    char results[500];

    char hostname[HOST_NAME_MAX];

    gethostname(hostname, HOST_NAME_MAX);
    string file_path_stat = output_path + "data_received.txt.";
    std::strcpy(results, file_path_stat.c_str());
    std::strcpy(results + strlen(file_path_stat.c_str()), hostname);
    ofstream fout(results, std::ios_base::app);

    for(int k=0; k< total_sum * this->data_dimension;k++){
       if(receive_values[k] != send_values[k]){
           cout<<" values mismacth "<<receive_values[k]<<" "<<send_values[k]<<endl;
       }
    }

    my_start_count = leafs_per_node * this->rank;
    if (this->rank < this->world_size - 1) {
        end_count = leafs_per_node * (this->rank + 1);
    } else {
        end_count = total_leaf_size;
    }

    vector<int> process_read_offsets(this->world_size);
    vector<int> process_read_offsets_value(this->world_size);
    vector<vector<DataPoint>> all_leaf_nodes(leafs_per_node);

    for (int i = 0; i < leafs_per_node; i++) {
        vector <DataPoint> datavec(total_leaf_count[i]);
        for (int j = 0; j < this->world_size; j++) {
            int count_per_leaf_per_node = recv_counts[i + j * leafs_per_node];
            int read_offset = recev_disps_count[j];
            int read_offset_data = recev_disps_values_count[j];

            if (i == 0) {
                process_read_offsets[j] = read_offset + count_per_leaf_per_node;
                process_read_offsets_value[j] = read_offset_data + count_per_leaf_per_node * this->data_dimension;
            } else {
                read_offset = process_read_offsets[j];
                process_read_offsets[j] = read_offset + count_per_leaf_per_node;
                read_offset_data = process_read_offsets_value[j];
                process_read_offsets_value[j] = read_offset_data + count_per_leaf_per_node * this->data_dimension;
            }

            int value_read_count = read_offset_data;
            for (int k = read_offset; k < process_read_offsets[j]; k++) {
                DataPoint dataPoint;
                dataPoint.index = receive_indices[k];
                if (dataPoint.index <= 0 || dataPoint.index >= 60000){
                    cout<< " index zero for k "<< read_offset<<endl;
                }
                dataPoint.image_data = vector<VALUE_TYPE>(this->data_dimension);
                for (int m = value_read_count; m < value_read_count + this->data_dimension; m++) {
                    dataPoint.image_data[m - value_read_count] = receive_values[m];
                    if ( dataPoint.image_data[m - value_read_count] < 0 ||  dataPoint.image_data[m - value_read_count] > 255){
                        cout<< " value zero for k "<< read_offset<<" m"<<m <<endl;
                    }
                }
                datavec.push_back(dataPoint);
                value_read_count += this->data_dimension;

            }
        }
        this->trees_leaf_first_indices_all[tree][i + my_start_count] = datavec;
        all_leaf_nodes[i]=datavec;
    }

    return all_leaf_nodes;

}


vector <vector<dmrpt::DataPoint>> dmrpt::DRPTGlobal::calculate_nns(int tree, int nn) {

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
    } else {
        my_start_count = this->rank % total_leaf_size;
        end_count = my_start_count + 1;

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
        vector <DataPoint> data_points = this->trees_leaf_first_indices_all[tree][i];
        cout<< " data point of size "<< data_points.size() <<" tree "<< tree << i<<endl;

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
                fout<<dataPoint.src_index << ' '<<dataPoint.index<<' '<<dataPoint.distance<<endl;
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
    fout << rank << " distance calc " << distance_time.count() << endl;

    return final_results;
}

vector <vector<dmrpt::DataPoint>> dmrpt::DRPTGlobal::gather_nns(int nn) {

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

    fout << rank << " distance  " << distance_time.count() << " query " << query_time.count() << endl;

    return collected_nns;
}