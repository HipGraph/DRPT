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
//            {
            for (int k = 0; k < this->data_dimension; k++) {
                array_val[k] = receive[k * send_count + j];
            }
//            }
            imageDataPoint.value = array_val;
//            if (allEqual(array_val) || array_val.size() == 0) {
//                cout << " recevied data zero for index ######" << imageDataPoint.index << endl;
//            }
            this->original_data_processed[original_data_count + j] = imageDataPoint;
        }
        int count_index = direction == 0 ? this->rank * 2 : this->rank * 2 + 1;
        total_counts[count_index] = total_counts[count_index] + send_count;
        free(receive);
        free(receving_indexes);
        free(recevied_projected);

        return data_points;

    } else if (max_rank == this->rank && current_rank != this->rank && remain > 0) {

        int *receving_indexes = new int[send_count];
        vector <vector<VALUE_TYPE>> sendVector(send_count);
        for (int j = 0; j < send_count; j++) {
            auto val = data_points.back();
            int index = val.index;

            vector<ImageDataPoint>::iterator it = std::find_if(this->original_data_processed.begin(),
                                                               this->original_data_processed.end(),
                                                               [index](ImageDataPoint const &n) {
                                                                   return n.index == index;
                                                               });

            vector<VALUE_TYPE> dataP = (*it).value;
            receving_indexes[j] = (*it).index;

//            if (allEqual(dataP) || dataP.size() == 0) {
//                cout << "  sending data zero for index ######" << receving_indexes[j] << endl;
//            }


            sendVector[j] = dataP;
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

        MPI_Send(receving_indexes, send_count, MPI_INT, current_rank, 0, MPI_COMM_WORLD);
        int tot = send_count * this->data_dimension;


        MPI_Send(sendVec, tot, MPI_VALUE_TYPE, current_rank, 1, MPI_COMM_WORLD);

        int count_index = direction == 0 ? this->rank * 2 : this->rank * 2 + 1;
        total_counts[count_index] = total_counts[count_index] - send_count;

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

            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_INT, total_counts, process_counts, disps, MPI_INT, MPI_COMM_WORLD);

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
dmrpt::DRPTGlobal::collect_similar_data_points(int tree) {

    dmrpt::MathOp mathOp;
//    int selected_leaf = index - (1 << (this->tree_depth - 1)) + 1;

    int total_leaf_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

    int leafs_per_node = total_leaf_size / this->world_size;
    int my_start_count = 0;
    int end_count = 0;
    int sending_rank = -1;

    int *send_counts = new int[total_leaf_size];
    int *recv_counts = new int[total_leaf_size];

    cout << "total leaf size" << total_leaf_size << endl;

    int sum_per_node=0;
    int process=0;
    int *send_indices_count = new int[this->world_size];
    int *disps_indices_count = new int[this->world_size];


    int my_total=0;
    for (int i = 0; i < total_leaf_size; i++) {
        if (i>0 && i % leafs_per_node == 0){
            send_indices_count[process]=sum_per_node;
            sum_per_node=0;
            process++;
        }
        vector <DataPoint> all_points = this->trees_leaf_first_indices[tree][i];
        send_counts[i] = all_points.size();
        sum_per_node +=send_counts[i];
        my_total +=send_counts[i];
    }

    MPI_Alltoall(send_counts, leafs_per_node, MPI_INT, recv_counts, leafs_per_node,
                 MPI_INT, MPI_COMM_WORLD);



    int *total_leaf_count = new int[leafs_per_node];
    int *disps_indices = new int[this->world_size];
    int *recev_indices_count = new int[this->world_size];
    int *recev_disps_count = new int[this->world_size];
    int total_sum=0;
    for (int j = 0; j < leafs_per_node; j++) {
        int count = 0;
        for (int i = 0; i < this->world_size; i++) {
            count += recv_counts[j + i * leafs_per_node];
        }
        total_leaf_count[j] = count;
        total_sum +=count;
    }

    for (int i = 0; i < this->world_size; i++) {
        int count = 0;
        for (int j = 0; j < leafs_per_node; j++) {
            count += recv_counts[j + i * leafs_per_node];
        }
        recev_indices_count[i] = count;
        cout<<" rank "<<rank <<" count "<< recev_indices_count[i]<< endl;
    }

    for (int i = 0; i < this->world_size; i++) {
        disps_indices_count[i] = (i > 0) ? (disps_indices_count[i - 1] + send_indices_count[i - 1]) : 0;
        recev_disps_count[i]=(i > 0) ? (recev_disps_count[i - 1] + recev_indices_count[i - 1]) : 0;
    }


    cout<<" total sum"<< total_sum << endl;


    int *receive_indices = new int[total_sum];

    int *send_indices = new int[my_total];


    int co=0;
    for (int i = 0; i < total_leaf_size; i++) {
        vector <DataPoint> all_points = this->trees_leaf_first_indices[tree][i];
        for(int j=0; j<all_points.size();j++){
            send_indices[co]=all_points[j].index;
            co++;
        }
    }

    int *send_ind  = new int[2];
    send_ind[0]=1000;
    send_ind[1]=1001;

    int *send_ind_count = new int[1];
    send_ind_count[0]=2;

    int *disps_ind_count = new int[1];
    disps_ind_count[0]=0;

    int *receive_ind = new int[2];
    int *recev_ind_count = new int[1];
    recev_ind_count[0]=2;

    int *recev_disp_count = new int[1];
    recev_disp_count[0]=0;

//    MPI_Alltoallv(send_indices,send_indices_count,disps_indices_count,MPI_INT,receive_indices,
//                  recev_indices_count,recev_disps_count,MPI_INT,MPI_COMM_WORLD);

    MPI_Alltoallv(send_ind,send_ind_count,disps_ind_count,MPI_INT,receive_ind,
                  recev_ind_count,recev_disp_count,MPI_INT,MPI_COMM_WORLD);


     if(this->rank==0)    {
        for(int i=0;i<total_sum;i++){
            cout<<receive_indices[i]<<' '<<endl;
        }
     }



    cout << " completed for loop" << rank << endl;







//    //large trees
//    if (total_leaf_size >= this->world_size) {
//        my_start_count = leafs_per_node * this->rank;
//        if (this->rank < this->world_size - 1) {
//            end_count = leafs_per_node * (this->rank + 1);
//        } else {
//            end_count = total_leaf_size;
//        }
//
//        if (selected_leaf >= my_start_count && selected_leaf < end_count) {
//            vector <DataPoint> dps = this->request_data_points_for_given_index(all_points);
//            return dps;
//        } else {
//
//            for (int ra = 0; ra < this->world_size; ra++) {
//                if (ra < (this->world_size - 1) && selected_leaf >= leafs_per_node * ra &&
//                    selected_leaf < leafs_per_node * (ra + 1)) {
//                    sending_rank = ra;
//                    break;
//                } else if (ra == (this->world_size - 1) && selected_leaf >= leafs_per_node * ra) {
//                    sending_rank = ra;
//                    break;
//                }
//            }
////            cout<<" rank "<<this->rank<<" sending data to "<<sending_rank <<" size "<<all_points.size()
////                << " tree "<<tree<<" leaf "<<selected_leaf <<endl;
//            return this->send_data_points_for_requested_node(all_points, sending_rank);
//        }
//    }

}


void dmrpt::DRPTGlobal::collect_similar_data_points_for_all_tree_indices(int tree, int index, int depth) {

    int id_left = 2 * index + 1;
    int id_right = id_left + 1;

//    if (depth == this->tree_depth - 1) {
//        int selected_leaf = index - (1 << (this->tree_depth - 1)) + 1;
//        this->trees_leaf_first_indices_all[tree][selected_leaf] = this->collect_similar_data_points_for_given_tree_index(
//                tree, index);
//        return;
//    }

    collect_similar_data_points_for_all_tree_indices(tree, id_left, depth + 1);
    collect_similar_data_points_for_all_tree_indices(tree, id_right, depth + 1);

}


vector <dmrpt::DataPoint>
dmrpt::DRPTGlobal::request_data_points_for_given_index(vector <DataPoint> all_my_points) {

    dmrpt::MathOp mathOp;

    int *counts = new int[1];
    counts[0] = all_my_points.size();

    int *process_counts = new int[this->world_size];

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

//        if (src_it == this->original_data_processed.end()) {
//            cout << " couldn't find " << src_index << endl;
//        }
        send_vector[g] = ((*src_it).value);

//        this->original_data_processed.erase(src_it);

    }

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

            DataPoint dataPoint;
            dataPoint.index = total_recev_indexes[my_index_start + h];
            co++;
            vector<VALUE_TYPE> im_data(this->data_dimension);
#pragma omp parallel for
//            {
            for (int y = 0; y < this->data_dimension; y++) {
                int get_index = my_start + h + process_counts[m] * y;
//                if (total_recev_queries[get_index] > 255 || total_recev_queries[get_index] < 0) {
//                    cout << " index " << dataPoint.index << " calculated index " << get_index << endl;
//                }
                im_data[y] = total_recev_queries[get_index];
            }
//            }

            dataPoint.image_data = im_data;

            all_my_points.push_back(dataPoint);
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

    }
//    if (send_vector.empty()) {
//        cout << " send vector empty **********" << endl;
//    }

    VALUE_TYPE *my_queries = mathOp.convert_to_row_major_format(send_vector);


    //send indices of selected nodes
    MPI_Gatherv(my_send, all_my_points.size(), MPI_INT, NULL, NULL, NULL, MPI_INT,
                sending_rank,
                MPI_COMM_WORLD);

    int cf = send_vector.size() * this->data_dimension;

    //gather queries
    MPI_Gatherv(my_queries, cf, MPI_VALUE_TYPE, NULL, NULL,
                NULL, MPI_VALUE_TYPE, sending_rank, MPI_COMM_WORLD);

    free(counts);
    free(my_send);
    free(my_queries);

    all_my_points.clear();
    return all_my_points;
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
//        cout<< " data point of size "<< data_points.size() <<" tree "<< tree << i<<endl;

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