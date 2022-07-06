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


using namespace std;
using namespace std::chrono;

dmrpt::DRPTGlobal::DRPTGlobal() {

}

dmrpt::DRPTGlobal::DRPTGlobal(VALUE_TYPE *projected_matrix, VALUE_TYPE *projection_matrix, int no_of_data_points,
                              int tree_depth,
                              vector <vector<VALUE_TYPE>> original_data, int ntrees,
                              int starting_index, int total_data_set_size, int donate_per,
                              dmrpt::StorageFormat storage_format, int rank, int world_size) {
    this->tree_depth = tree_depth;
    this->intial_no_of_data_points = no_of_data_points;
    this->storage_format = storage_format;
    this->projected_matrix = projected_matrix;
    this->projection_matrix = projection_matrix;
    this->total_data_set_size = total_data_set_size;
    this->data_dimension = original_data[0].size();


    this->ntrees = ntrees;

    this->trees_data = vector < vector < vector < DataPoint>>>(ntrees);
    this->trees_splits = vector < vector < VALUE_TYPE >> (ntrees);
    this->trees_indices = vector < vector < int >> (ntrees);
    this->trees_leaf_first_indices_all = vector < vector < vector < int>>>(ntrees);
    this->trees_leaf_first_indices = vector < vector < int >> (ntrees);

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
    cout << "completed processing" << endl;
}


void dmrpt::DRPTGlobal::grow_global_tree() {
    if (this->tree_depth <= 0 || this->tree_depth > log2(this->intial_no_of_data_points)) {
        throw std::out_of_range(" depth should be in range [1,....,log2(rows)]");
    }

    if (this->ntrees <= 0) {
        throw std::out_of_range(" no of trees should be greater than zero");
    }

    int total_split_size = 1 << (this->tree_depth + 1);

    if (dmrpt::StorageFormat::RAW == storage_format) {

        for (int k = 0; k < this->ntrees; k++) {
            this->trees_splits[k] = vector<VALUE_TYPE>(total_split_size);
            this->trees_data[k] = vector < vector < DataPoint >> (this->tree_depth);
            this->trees_indices[k] = vector<int>(this->intial_no_of_data_points);


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
            grow_global_subtree(this->trees_data[k][0], this->total_data_set_size, 0, 0, k);

        }
    }

}


void dmrpt::DRPTGlobal::grow_global_subtree(std::vector <DataPoint> data_vector,
                                            int total_data_set_size, int depth, int index, int tree) {

    int id_left = 2 * index + 1;
    int id_right = id_left + 1;

    MathOp mathOp;

    VALUE_TYPE *data = new VALUE_TYPE[data_vector.size()];

#pragma omp parallel for
    for (int i = 0; i < data_vector.size(); i++) {
        data[i] = data_vector[i].value;
    }


//    cout << " calling distirbuted mean calc rank " << this->rank << endl;
    VALUE_TYPE *result = mathOp.distributed_median(data, data_vector.size(), 1, total_data_set_size,
                                                   7, dmrpt::StorageFormat::RAW, this->rank);
//    cout << " exiting distirbuted mean calc rank " << this->rank << endl;

    VALUE_TYPE median = result[0];


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


    int left_totol = 0, right_total = 0;
    for (int i = 0; i < this->world_size; i++) {
        int id = i * 2;
        left_totol = left_totol + total_counts[id];
        right_total = right_total + total_counts[id + 1];
    }

//    cout << " Rank " << rank << " Depth " << depth << " left child size "
//         << left_childs_global.size() << " right child size " << right_childs_global.size() << " left total "
//         << left_totol << " right total " << right_total << endl;



    this->send_receive_data_points_if_zero(left_childs_global, right_childs_global, total_counts, process_counts, disps, tree);

    free(process_counts);
    free(total_counts);
    free(disps);

    if (depth == this->tree_depth - 2) {
//        cout<<" returing "<<this->rank<<endl;
        return;
    }

    grow_global_subtree(left_childs_global, left_totol, depth + 1, id_left, tree);
    grow_global_subtree(right_childs_global, right_total, depth + 1, id_right, tree);
}


int dmrpt::DRPTGlobal::detect_max_rank(int *total_counts, int direction) {
    int max = 0;
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

void dmrpt::DRPTGlobal::send_receive_data_points_if_zero(vector<DataPoint> data_points, int* total_counts,int current_rank, int direction, int tree) {
     dmrpt::MathOp mathOp;
    int max_rank = this->detect_max_rank(total_counts, direction);
    int search_index = direction==0?max_rank*2:max_rank*2+1;
    int count = total_counts[search_index];
    int send_count = ceil(count * this->donate_per * 1.0 / 100);
    int remain = count - send_count;
    vector <vector<VALUE_TYPE>> sendVector(send_count);

    if (max_rank != this->rank && remain > 0 && current_rank == this->rank) {
        VALUE_TYPE *receive = new VALUE_TYPE[send_count * this->data_dimension];
        int *receving_indexes = new int[send_count];
        MPI_Recv(receving_indexes, send_count, MPI_INT, max_rank, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(receive, send_count * this->data_dimension, MPI_VALUE_TYPE, max_rank, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        VALUE_TYPE *recevied_projected = mathOp.multiply_mat(receive, this->projection_matrix, send_count,
                                                             this->tree_depth, this->data_dimension, 1.0);
        cout << " rank " << this->rank << " projection completed " << endl;
        for (int dep = 0; dep < this->tree_depth; dep++) {
            int current_size = this->trees_data[tree][dep].size();
            this->trees_data[tree][dep].resize(current_size + send_count);
            for (int j = 0; j < send_count; j++) {
                int ind = this->tree_depth * tree + dep + j * this->tree_depth * this->ntrees;
                DataPoint dataPoint;
                dataPoint.value = recevied_projected[ind];
                dataPoint.index = receving_indexes[j];
                this->trees_data[tree][dep][current_size + j] = dataPoint;
                data_points.push_back(dataPoint);
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
                    array_val[k] = receive[j * this->data_dimension + k];
                }
            }
            imageDataPoint.value = array_val;
            this->original_data_processed[original_data_count + j] = imageDataPoint;
        }
        int count_index = direction==0?this->rank*2:this->rank*2+1;
        total_counts[count_index]=total_counts[count_index]+send_count;
        free(receive);
        free(receving_indexes);
        free(recevied_projected);

    } else if (max_rank == this->rank && current_rank != this->rank && remain > 0) {
        int *receving_indexes = new int[send_count];
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
            sendVector.push_back(dataP);
            data_points.pop_back();

            //remove these values from rest of the projected matrix

            for (int dep = 0; dep < this->tree_depth; dep++) {
                vector<DataPoint>::iterator dit = std::find_if(this->trees_data[tree][dep].begin(),
                                                               this->trees_data[tree][dep].end(),
                                                               [index](DataPoint const &n) {
                                                                   return n.index == index;
                                                               });
                if (dit != this->trees_data[tree][dep].end()) {
                    cout<<"deleting "<<endl;
                    this->trees_data[tree][dep].erase(dit);
                }else {
                    cout << "deleting ommiting rank" << this->rank << endl;
                }
            }
        }

        VALUE_TYPE *sendVec = mathOp.convert_to_row_major_format(sendVector);
        cout << " rank " << this->rank << " sending " << send_count << " max rank " << max_rank << endl;
        MPI_Send(receving_indexes, send_count, MPI_INT, current_rank, 0, MPI_COMM_WORLD);
        MPI_Send(sendVec, send_count * this->data_dimension, MPI_VALUE_TYPE, current_rank, 1, MPI_COMM_WORLD);

        int count_index = direction==0?this->rank*2:this->rank*2+1;
        total_counts[count_index]=total_counts[count_index]-send_count;

        cout << " rank " << this->rank << " sending " << send_count << " max rank completed" << max_rank << endl;
        free(receving_indexes);
        free(sendVec);
    }


}

void dmrpt::DRPTGlobal::send_receive_data_points_if_zero(vector <DataPoint> left_data_points,
                                                                              vector <DataPoint> right_data_points,
                                                                              int *total_counts, int *process_counts, int *disps, int tree) {

    MathOp mathOp;
    for (int i = 0; i < this->world_size; i++) {
        int id = i * 2;
        int left = total_counts[id];
        int right = total_counts[id + 1];

        if (left == 0) {
            cout << "calling left rank" << this->rank << endl;
            this->send_receive_data_points_if_zero(left_data_points, total_counts, i, 0, tree);
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_INT, total_counts, process_counts, disps, MPI_INT, MPI_COMM_WORLD);
        } else if (right == 0) {
            cout << "calling right rank" << this->rank << endl;
            this->send_receive_data_points_if_zero(right_data_points, total_counts, i, 1, tree);
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_INT, total_counts, process_counts, disps, MPI_INT, MPI_COMM_WORLD);
        }
    }
}



