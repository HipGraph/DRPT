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


using namespace std;
using namespace std::chrono;

dmrpt::DRPTGlobal::DRPTGlobal() {

}

dmrpt::DRPTGlobal::DRPTGlobal(VALUE_TYPE *projected_matrix, VALUE_TYPE *projection_matrix, int no_of_data_points,
                              int tree_depth,
                              vector <vector<VALUE_TYPE>> original_data, int ntrees,
                              int starting_index, int total_data_set_size, int donate_per, int transfer_threshold,
                              dmrpt::StorageFormat storage_format, int rank, int world_size) {
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
    this->trees_indices = vector < vector < int >> (ntrees);
    this->trees_leaf_first_indices = vector < vector < vector < DataPoint > >> (ntrees);

    this->starting_data_index = starting_index;
    this->rank = rank;
    this->world_size = world_size;
    this->leaf_data = vector < vector < DataPoint >> (this->ntrees);
    this->donate_per = donate_per;
    this->original_data_processed = vector<ImageDataPoint>(intial_no_of_data_points);


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

    cout << " tree size " << total_child_size << " tree depth" << tree_depth << " rank " << this->rank << endl;
    this->trees_leaf_first_indices_all = vector < vector < dmrpt::DataPoint >> (total_child_size);
    if (dmrpt::StorageFormat::RAW == storage_format) {

        for (int k = 0; k < this->ntrees; k++) {
            this->trees_splits[k] = vector<VALUE_TYPE>(total_split_size);
            this->trees_data[k] = vector < vector < DataPoint >> (this->tree_depth);
            this->trees_indices[k] = vector<int>(this->intial_no_of_data_points);
            this->trees_leaf_first_indices[k] = vector < vector < DataPoint >> (total_child_size);


            for (int i = 0; i < this->tree_depth; i++) {
                this->trees_data[k][i] = vector<DataPoint>(this->intial_no_of_data_points);
#pragma  omp parallel for
                {
                    for (int j = 0; j < this->intial_no_of_data_points; j++) {
                        int index = this->tree_depth * k + i + j * this->tree_depth * this->ntrees;
                        DataPoint dataPoint;
                        dataPoint.value = this->projected_matrix[index];
                        dataPoint.index = j + this->starting_data_index;
                        this->trees_data[k][i][j] = dataPoint;
                    }
                }
            }

//            iota(this->trees_indices[k].begin(), this->trees_indices[k].end(), 0);
            this->grow_global_subtree(this->trees_data[k][0], this->total_data_set_size, 0, 0, k);

        }

    }

}


void dmrpt::DRPTGlobal::grow_global_subtree(std::vector<DataPoint> data_vector,
                                            int total_data_set_size, int depth, int index, int tree) {

    int id_left = 2 * index + 1;
    int id_right = id_left + 1;

    if (depth == this->tree_depth - 1) {
        int selected_leaf = index - (1 << (this->tree_depth - 1)) + 1;
        this->trees_leaf_first_indices[tree][selected_leaf] = data_vector;
        return;
    }

    MathOp mathOp;

    VALUE_TYPE *data = new VALUE_TYPE[data_vector.size()];

#pragma omp parallel for
    for (int i = 0; i < data_vector.size(); i++) {
        data[i] = data_vector[i].value;
    }

    int no_of_bins = 1 + (3.322 * log2(data_vector.size()));

//    cout << " calling distirbuted mean calc rank " << this->rank << endl;
    VALUE_TYPE *result = mathOp.distributed_median(data, data_vector.size(), 1, total_data_set_size,
                                                   28, dmrpt::StorageFormat::RAW, this->rank);
//    cout << " exiting distirbuted mean calc rank " << this->rank << endl;

    VALUE_TYPE median = result[0];

//    cout<<" rank "<< this->rank<<" median  "<<median<<" depth  "<<depth<<endl;

    this->trees_splits[tree][index] = median;
    vector <DataPoint> left_childs_global;
    vector <DataPoint> right_childs_global;
#pragma omp parallel
    {
        vector <DataPoint> left_childs;
        vector <DataPoint> right_childs;
#pragma omp for  nowait
        for (int i = 0; i < data_vector.size(); i++) {
            int index = data_vector[i].index;


            std::vector<DataPoint>::iterator it = std::find_if(this->trees_data[tree][depth + 1].begin(),
                                                               this->trees_data[tree][depth + 1].end(),
                                                               [index](DataPoint const &n) {
                                                                   return n.index == index;
                                                               });
//            if (it == this->trees_data[tree][depth+1].end()){
//                cout<<" rank "<<this->rank<<" depth "<< depth<<" cannot find "<<(*it).value<<endl;
//            }
            if (data_vector[i].value <= median) {

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





    // Displacements in the receive buffer for MPI_GATHERV
    int *disps = new int[this->world_size];


    // Displacement for the first chunk of data - 0
    for (int i = 0; i < this->world_size; i++) {
        disps[i] = (i > 0) ? (disps[i - 1] + 2) : 0;
    }


    int *total_counts = new int[2 * this->world_size];


    int *process_counts = new int[this->world_size];
    for (int k = 0; k < this->world_size; k++) {
        process_counts[k] = 2;
        int id = this->rank * 2;
        total_counts[id] = left_childs_global.size();
        total_counts[id + 1] = right_childs_global.size();
    }

//   cout << " Calling API gatherV mean calc rank" << this->rank << endl;
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_INT, total_counts, process_counts, disps, MPI_INT, MPI_COMM_WORLD);
//    cout << " Ending API gatherV mean calc rank " << this->rank << endl;

//    int left_totol = 0, right_total = 0;
//    for (int i = 0; i < this->world_size; i++) {
//        int id = i * 2;
//        left_totol = left_totol + total_counts[id];
//        right_total = right_total + total_counts[id + 1];
//    }
//
//    if (left_totol == 0 || right_total == 0) {
//        cout << " Rank " << rank << " Depth " << depth << " left child size "
//             << left_childs_global.size() << " right child size " << right_childs_global.size() << " left total "
//             << left_totol << " right total " << right_total<< " median "<< median<<" no of bins"<<no_of_bins << endl;
//
//    }
    left_childs_global = this->send_receive_data_points_if_zero(left_childs_global, total_counts, process_counts, disps,
                                                                depth, 0, tree);
    right_childs_global = this->send_receive_data_points_if_zero(right_childs_global, total_counts, process_counts,
                                                                 disps,
                                                                 depth, 1, tree);

    int left_totol = 0;
    int right_total = 0;
    for (int i = 0; i < this->world_size; i++) {
        int id = i * 2;
        left_totol = left_totol + total_counts[id];
        right_total = right_total + total_counts[id + 1];
    }
//    if (left_totol == 0 || right_total == 0) {
//        cout << " Rank ###" << rank << " Depth " << depth << " left child size "
//             << left_childs_global.size() << " right child size " << right_childs_global.size() << " left total "
//             << left_totol << " right total " << right_total << " median " << median << endl;
//
//    }

    free(process_counts);
    free(total_counts);
    free(disps);
    free(data);

    grow_global_subtree(left_childs_global, left_totol, depth + 1, id_left, tree);
    grow_global_subtree(right_childs_global, right_total, depth + 1, id_right, tree);
}


int dmrpt::DRPTGlobal::detect_max_rank(int *total_counts, int direction) {
    int max = -1;
    int select_rank = -1;
    for (int i = 0; i < this->world_size; i++) {
        int id = 0;

        if (direction == 0) {
            id = i * 2;
        } else {
            id = i * 2 + 1;
        }

        if (total_counts[id] > max) {
            max = total_counts[id];
            select_rank = i;
        }

    }
    return select_rank;
}

int dmrpt::DRPTGlobal::detect_min_rank(int *total_counts, int direction) {
    int max = INT_MAX;
    int select_rank = -1;
    for (int i = 0; i < this->world_size; i++) {
        int id = 0;

        if (direction == 0) {
            id = i * 2;
        } else {
            id = i * 2 + 1;
        }

        if (total_counts[id] < max) {
            max = total_counts[id];
            select_rank = i;
        }

    }
    return select_rank;


}

vector <dmrpt::DataPoint>
dmrpt::DRPTGlobal::send_receive_data_points_if_zero(vector <DataPoint> data_points, int *total_counts, int current_rank,
                                                    int direction, int depth, int tree) {
    dmrpt::MathOp mathOp;
    int max_rank = this->detect_max_rank(total_counts, direction);
    int search_index = direction == 0 ? max_rank * 2 : max_rank * 2 + 1;
    int count = total_counts[search_index];
    int send_count = ceil(count * this->donate_per * 1.0 / 100);
    int remain = count - send_count;

    if (max_rank != this->rank && remain > 0 && current_rank == this->rank) {
//        cout << " rank " << rank << " receiving " << send_count << endl;
        VALUE_TYPE *receive = new VALUE_TYPE[send_count * this->data_dimension];
        int *receving_indexes = new int[send_count];
        MPI_Recv(receving_indexes, send_count, MPI_INT, max_rank, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        int tot = send_count * this->data_dimension;
        MPI_Recv(receive, tot, MPI_VALUE_TYPE, max_rank, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        VALUE_TYPE *recevied_projected = mathOp.multiply_mat(receive, this->projection_matrix, send_count,
                                                             this->tree_depth, this->data_dimension, 1.0);

        DataPoint selected_datapoint;
        for (int dep = 0; dep < this->tree_depth; dep++) {
            int current_size = this->trees_data[tree][dep].size();
            this->trees_data[tree][dep].resize(current_size + send_count);
            for (int j = 0; j < send_count; j++) {
                int ind = this->tree_depth * tree + dep + j * this->tree_depth * this->ntrees;
                DataPoint dataPoint;
                dataPoint.value = recevied_projected[ind];
                dataPoint.index = receving_indexes[j];
                this->trees_data[tree][dep][current_size + j] = dataPoint;
                if (dep == depth + 1) {
//                    cout<<" Adding to rank "<< this->rank<<" value "<<dataPoint.index<<endl;
                    data_points.push_back(dataPoint);
                }

            }
        }

        int original_data_count = this->original_data_processed.size();
        this->original_data_processed.resize(original_data_count + send_count);
        for (int j = 0; j < send_count; j++) {
            ImageDataPoint imageDataPoint;
            imageDataPoint.index = receving_indexes[j];
            vector<VALUE_TYPE> array_val(this->data_dimension);
#pragma omp parallel for
            {
                for (int k = 0; k < this->data_dimension; k++) {
                    array_val[k] = receive[k * send_count + j];
                }
            }
            imageDataPoint.value = array_val;
            if (allEqual(array_val) || array_val.size() == 0) {
                cout << " recevied data zero for index ######" << imageDataPoint.index << endl;
            }
            this->original_data_processed[original_data_count + j] = imageDataPoint;
        }
        int count_index = direction == 0 ? this->rank * 2 : this->rank * 2 + 1;
        total_counts[count_index] = total_counts[count_index] + send_count;
        free(receive);
        free(receving_indexes);
        free(recevied_projected);

        return data_points;

    } else if (max_rank == this->rank && current_rank != this->rank && remain > 0) {
//        cout << " rank " << rank << " send_count sending " << send_count << endl;
        int *receving_indexes = new int[send_count];
        vector <vector<VALUE_TYPE>> sendVector(send_count);
        for (int j = 0; j < send_count; j++) {
            auto val = data_points.back();
            int index = val.index;
//            cout << " rank " << rank << "searc index " << index << endl;
            vector<ImageDataPoint>::iterator it = std::find_if(this->original_data_processed.begin(),
                                                               this->original_data_processed.end(),
                                                               [index](ImageDataPoint const &n) {
                                                                   return n.index == index;
                                                               });

            vector<VALUE_TYPE> dataP = (*it).value;
            receving_indexes[j] = (*it).index;
            sendVector.push_back(dataP);
            data_points.pop_back();

            //remove these values from rest of the projected matrix

            for (int dep = 0; dep < this->tree_depth; dep++) {
                vector<DataPoint>::iterator dit = std::find_if(this->trees_data[tree][dep].begin(),
                                                               this->trees_data[tree][dep].end(),
                                                               [index](DataPoint const &n) {
                                                                   return n.index == index;
                                                               });

                this->trees_data[tree][dep].erase(dit);

            }
        }

        VALUE_TYPE *sendVec = mathOp.convert_to_row_major_format(sendVector);
//        cout << " rank " << this->rank << " sending " << send_count << " max rank " << max_rank << endl;
        MPI_Send(receving_indexes, send_count, MPI_INT, current_rank, 0, MPI_COMM_WORLD);
        int tot = send_count * this->data_dimension;
        MPI_Send(sendVec, tot, MPI_VALUE_TYPE, current_rank, 1, MPI_COMM_WORLD);

        int count_index = direction == 0 ? this->rank * 2 : this->rank * 2 + 1;
        total_counts[count_index] = total_counts[count_index] - send_count;

//        cout << " rank " << this->rank << " sending " << send_count << " max rank completed" << max_rank << endl;
        free(receving_indexes);
        free(sendVec);

        return data_points;
    }

    return data_points;
}

vector <dmrpt::DataPoint> dmrpt::DRPTGlobal::send_receive_data_points_if_zero(vector <DataPoint> data_points,
                                                                              int *total_counts, int *process_counts,
                                                                              int *disps, int depth,
                                                                              int direction, int tree) {

    for (int i = 0; i < this->world_size; i++) {

        if (this->is_transfer_needed(total_counts, direction)) {
            vector <DataPoint> selected_data_points;

            selected_data_points = this->send_receive_data_points_if_zero(data_points, total_counts, i, direction,
                                                                          depth, tree);
//            cout<<"Calling MPIGatherV left "<<this->rank<<endl;
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_INT, total_counts, process_counts, disps, MPI_INT, MPI_COMM_WORLD);
//            cout<<"Exit MPIGatherV left "<<this->rank<<endl;
            return selected_data_points;
        }
    }
    return data_points;
}


bool dmrpt::DRPTGlobal::is_transfer_needed(int *total_counts, int direction) {

    int max_rank = this->detect_max_rank(total_counts, direction);
    int min_rank = this->detect_min_rank(total_counts, direction);
    int id_max = 0;
    int id_min = 0;
    if (direction == 0) {
        id_max = max_rank * 2;
        id_min = min_rank * 2;
    } else {
        id_max = max_rank * 2 + 1;
        id_min = min_rank * 2 + 1;
    }

    int total = 0;
    for (int i = 0; i < this->world_size; i++) {
        if (direction == 0) {
            int id = i * 2;
            total = total + total_counts[id];
        } else {
            int id = i * 2 + 1;
            total = total + total_counts[id];
        }
    }


    int max_value = total_counts[id_max];
    int min_value = total_counts[id_min];

    int diff_precentage = (max_value - min_value) * 100.0 / total;

    if (diff_precentage >= this->transfer_threshold) {
        return true;
    }

    return false;
}


vector <dmrpt::DataPoint>
dmrpt::DRPTGlobal::collect_similar_data_points_for_given_tree_index(int index) {

    dmrpt::MathOp mathOp;
    int selected_leaf = index - (1 << (this->tree_depth - 1)) + 1;

    int total_leaf_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

    int leafs_per_node = total_leaf_size / this->world_size;


    int my_start_count = leafs_per_node * this->rank;
    int end_count = 0;
    int sending_rank = -1;
    if (this->rank < this->world_size - 1) {
        end_count = leafs_per_node * (this->rank + 1);
    } else {
        end_count = total_leaf_size;
    }

    vector <DataPoint> all_points;


    for (int i = 0; i < this->ntrees; i++) {
        vector <DataPoint> dp_vecs = this->trees_leaf_first_indices[i][selected_leaf];
        all_points.insert(all_points.end(), dp_vecs.begin(), dp_vecs.end());
    }

    all_points.erase(unique(all_points.begin(), all_points.end(),
                            [](const DataPoint &lhs,
                               const DataPoint &rhs) {
                                return lhs.index == rhs.index;
                            }), all_points.end());

    if (selected_leaf >= my_start_count && selected_leaf < end_count) {
        vector <DataPoint> dps = this->request_data_points_for_given_index(all_points);
        return dps;
    } else {

        for (int ra = 0; ra < this->world_size; ra++) {
            if (ra < (this->world_size - 1) && selected_leaf >= leafs_per_node * ra &&
                selected_leaf < leafs_per_node * (ra + 1)) {
                sending_rank = ra;
                break;
            } else if (ra == (this->world_size - 1) && selected_leaf >= leafs_per_node * ra) {
                sending_rank = ra;
                break;
            }
        }
        return this->send_data_points_for_requested_node(all_points, sending_rank);
    }

}


void dmrpt::DRPTGlobal::collect_similar_data_points_for_all_tree_indices(int index, int depth) {

    int id_left = 2 * index + 1;
    int id_right = id_left + 1;

    if (depth == this->tree_depth - 1) {
        int selected_leaf = index - (1 << (this->tree_depth - 1)) + 1;
        this->trees_leaf_first_indices_all[selected_leaf] = this->collect_similar_data_points_for_given_tree_index(
                index);
        return;
    }

    collect_similar_data_points_for_all_tree_indices(id_left, depth + 1);
    collect_similar_data_points_for_all_tree_indices(id_right, depth + 1);

}


vector <dmrpt::DataPoint>
dmrpt::DRPTGlobal::request_data_points_for_given_index(vector <DataPoint> all_my_points) {

    dmrpt::MathOp mathOp;

    int *counts = new int[1];
    counts[0] = all_my_points.size();

    int *process_counts = new int[this->world_size];

//    MPI_Bcast(&index, 1, MPI_INT, this->rank, MPI_COMM_WORLD);

    MPI_Gather(counts, 1, MPI_INT, process_counts, 1, MPI_INT, this->rank, MPI_COMM_WORLD);


    int sum = 0;
    // Displacements in the receive buffer for MPI_GATHERV
    int *disps = new int[this->world_size];


    int *process_counts_queries = new int[this->world_size];
    int *disps_queries = new int[this->world_size];

    // Displacement for the first chunk of data - 0
    for (int i = 0; i < this->world_size; i++) {
        sum = sum + process_counts[i];
        process_counts_queries[i] = process_counts[i] * this->data_dimension;
        disps[i] = (i > 0) ? (disps[i - 1] + process_counts[i - 1]) : 0;
        disps_queries[i] = (i > 0) ? (disps_queries[i - 1] + process_counts_queries[i - 1]) : 0;
    }

    int *total_recev_indexes = new int[sum];
    int *my_send = new int[process_counts[this->rank]];
    VALUE_TYPE *total_recev_queries = new VALUE_TYPE[sum * this->data_dimension];
    vector <vector<VALUE_TYPE>> send_vector(all_my_points.size());

#pragma omp parallel for
    for (int g = 0; g < all_my_points.size(); g++) {
        int src_index = all_my_points[g].index;
        my_send[g] = src_index;
        vector<ImageDataPoint>::iterator src_it = std::find_if(this->original_data_processed.begin(),
                                                               this->original_data_processed.end(),
                                                               [src_index](ImageDataPoint const &n) {
                                                                   return n.index == src_index;
                                                               });
        send_vector[g] = ((*src_it).value);
//        this->original_data_processed.erase(src_it);

    }

//    cout << "my rank" << this->rank << " send vector size " << send_vector.size() << endl;
//    cout << "my rank" << this->rank << " total vector size " << sum << endl;
    VALUE_TYPE *my_queries = mathOp.convert_to_row_major_format(send_vector);


    //send indices of selected nodes
    MPI_Gatherv(my_send, process_counts[this->rank], MPI_INT, total_recev_indexes, process_counts, disps, MPI_INT,
                this->rank,
                MPI_COMM_WORLD);

    //gather queries
    MPI_Gatherv(my_queries, process_counts_queries[this->rank], MPI_VALUE_TYPE, total_recev_queries,
                process_counts_queries,
                disps_queries,
                MPI_VALUE_TYPE, this->rank,
                MPI_COMM_WORLD);

    int original_data_count = this->original_data_processed.size();

//    this->original_data_processed.resize(original_data_count + sum);

    int co = original_data_count;
    all_my_points.clear();
//    vector <DataPoint> results(sum);
    for (int m = 0; m < this->world_size; m++) {
        int my_index_start = disps[m];
        int my_start = disps_queries[m];
        int my_end = my_start + process_counts_queries[m];
        int process_sum = process_counts_queries[m];
        for (int h = 0; h < process_counts[m]; h++) {
//            ImageDataPoint imageDataPoint;
//            imageDataPoint.value = vector<VALUE_TYPE>(this->data_dimension);
//            imageDataPoint.index = total_recev_indexes[my_index_start + h];
//            this->original_data_processed[co] = imageDataPoint;

            DataPoint dataPoint;
            dataPoint.index = total_recev_indexes[my_index_start + h];
            co++;
            vector<VALUE_TYPE> im_data(this->data_dimension);
#pragma omp parallel for
            {
                for (int y = 0; y < this->data_dimension; y++) {
                    int get_index = my_start + h + process_counts[m] * y;
                    im_data[y] = total_recev_queries[get_index];
                }
            }

            dataPoint.image_data = im_data;
            if (dataPoint.image_data.size() == 0) {
                cout << " index " << dataPoint.index << " size " << dataPoint.image_data.size() << endl;
            }

            all_my_points.push_back(dataPoint);
        }
    }

    for (int h = 0; h < all_my_points.size(); h++) {
        if (all_my_points[h].image_data.size() == 0) {
            cout << "total size " << all_my_points.size() << "  EMpty for index"
                 << all_my_points[h].index << endl;

        }
    }


    free(counts);
    free(process_counts);
    free(process_counts_queries);
    free(disps_queries);
    free(total_recev_indexes);
    free(total_recev_queries);
    free(my_send);
    free(my_queries);
    free(disps);
    return all_my_points;
}


vector <dmrpt::DataPoint>
dmrpt::DRPTGlobal::send_data_points_for_requested_node(vector <DataPoint> all_my_points, int sending_rank) {

    dmrpt::MathOp mathOp;
    int *counts = new int[1];
    counts[0] = all_my_points.size();


    MPI_Gather(counts, 1, MPI_INT, NULL, 1, MPI_INT, sending_rank, MPI_COMM_WORLD);

    int sum = 0;
    // Displacements in the receive buffer for MPI_GATHERV

    int *my_send = new int[all_my_points.size()];

    vector <vector<VALUE_TYPE>> send_vector(all_my_points.size());

#pragma omp parallel for
    for (int g = 0; g < all_my_points.size(); g++) {
        int src_index = all_my_points[g].index;
        my_send[g] = src_index;
        vector<ImageDataPoint>::iterator src_it = std::find_if(this->original_data_processed.begin(),
                                                               this->original_data_processed.end(),
                                                               [src_index](ImageDataPoint const &n) {
                                                                   return n.index == src_index;
                                                               });
        send_vector[g] = ((*src_it).value);
//      this->original_data_processed.erase(src_it);
        if (allEqual(send_vector[g]) || send_vector[g].size() == 0) {
            cout << " all equal befor seding index  " << src_index << endl;
        }

    }
    VALUE_TYPE *my_queries = mathOp.convert_to_row_major_format(send_vector);


    //send indices of selected nodes
    MPI_Gatherv(my_send, all_my_points.size(), MPI_INT, NULL, NULL, NULL, MPI_INT,
                sending_rank,
                MPI_COMM_WORLD);

    int cf = send_vector.size() * this->data_dimension;
//    cout << " my rank " << this->rank << " sending rank " << sending_rank << " count " << send_vector.size() << endl;
    //gather queries
    MPI_Gatherv(my_queries, cf, MPI_VALUE_TYPE, NULL, NULL,
                NULL, MPI_VALUE_TYPE, sending_rank, MPI_COMM_WORLD);

    free(counts);
    free(my_send);
    free(my_queries);

    all_my_points.clear();
    return all_my_points;
}

vector <dmrpt::DataPoint> dmrpt::DRPTGlobal::get_nns(int nn) {

    dmrpt::MathOp mathOp;

    int total_leaf_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

    int leafs_per_node = total_leaf_size / this->world_size;

    int my_start_count = leafs_per_node * this->rank;
    int end_count = 0;

    if (this->rank < this->world_size - 1) {
        end_count = leafs_per_node * (this->rank + 1);
    } else {
        end_count = total_leaf_size;
    }
    vector <DataPoint> final_results;

//#pragma omp parallel
////#pragma omp for collapse(3)
//    {
#pragma omp parallel for
    for (int i = my_start_count; i < end_count; i++) {

        vector <DataPoint> data_points = this->trees_leaf_first_indices_all[i];
#pragma omp parallel for
        for (int j = 0; j < data_points.size(); j++) {
            vector <DataPoint> vec = vector<DataPoint>(data_points.size());

#pragma omp parallel for
            for (int k = 0; k < data_points.size(); k++) {

                VALUE_TYPE distance = mathOp.calculate_distance(data_points[j].image_data, data_points[k].image_data);
                DataPoint dataPoint;
                dataPoint.src_index = data_points[j].index;
                dataPoint.index = data_points[k].index;
                dataPoint.distance = distance;
                vec[k] = dataPoint;
            }

            sort(vec.begin(), vec.end(),
                 [](const DataPoint &lhs, const DataPoint &rhs) {
                     return lhs.distance < rhs.distance;
                 });
            vector <DataPoint> sub_vec = slice(vec, 0, nn - 1);
#pragma omp critical
            {
                final_results.insert(final_results.end(), sub_vec.begin(), sub_vec.end());
            }
        }
    }

//    }
    return final_results;
}



