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
                              vector <vector<VALUE_TYPE>> original_data,  int ntrees,
                              int starting_index,int total_data_set_size, dmrpt::StorageFormat storage_format, int rank, int world_size) {
    this->tree_depth = tree_depth;
    this->no_of_data_points = no_of_data_points;
    this->storage_format = storage_format;
    this->projected_matrix = projected_matrix;
    this->projection_matrix = projection_matrix;
    this->total_data_set_size = total_data_set_size;

    this->original_data = original_data;

    this->ntrees = ntrees;

    this->trees_data = vector < vector < vector < DataPoint>>>(ntrees);
    this->trees_splits = vector < vector < VALUE_TYPE >> (ntrees);
    this->trees_indices = vector < vector < int >> (ntrees);
    this->trees_leaf_first_indices_all = vector < vector < vector < int>>>(ntrees);
    this->trees_leaf_first_indices = vector < vector < int >> (ntrees);

    this->starting_data_index = starting_index;
    this->rank = rank;
    this->world_size = world_size;
    this->leaf_data= vector<vector<DataPoint>>(this->ntrees);

}


void dmrpt::DRPTGlobal::grow_global_tree() {
    if (this->tree_depth <= 0 || this->tree_depth > log2(this->no_of_data_points)) {
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
            this->trees_indices[k] = vector<int>(this->no_of_data_points);


            for (int i = 0; i < this->tree_depth; i++) {
                this->trees_data[k][i] = vector<DataPoint>(this->no_of_data_points);
#pragma  omp parallel for
                {
                    for (int j = 0; j < this->no_of_data_points; j++) {
                        int index = this->tree_depth * k + i + j * this->tree_depth * this->ntrees;
                          DataPoint dataPoint;
                          dataPoint.value =  this->projected_matrix[index];
                          dataPoint.index=j + this->starting_data_index;
                        this->trees_data[k][i][j] = dataPoint;
                    }
                }
            }

//            iota(this->trees_indices[k].begin(), this->trees_indices[k].end(), 0);
                grow_global_subtree(this->trees_data[k][0], this->total_data_set_size, 0, 0,k);

        }
    }

}


void dmrpt::DRPTGlobal::grow_global_subtree(std::vector <DataPoint> data_vector, int total_data_set_size, int depth, int index, int tree) {

    int id_left = 2 * index + 1;
    int id_right = id_left + 1;

    if (depth == this->tree_depth) {
        return;
    }

    MathOp mathOp;


    VALUE_TYPE *data = new VALUE_TYPE[data_vector.size()];

    #pragma omp parallel for
    for(int i=0;i<data_vector.size();i++){
        data[i]=data_vector[i].value;
    }


   VALUE_TYPE * result = mathOp.distributed_median(data,data_vector.size(),1,total_data_set_size,
                                                   7,dmrpt::StorageFormat::RAW,this->rank);

   VALUE_TYPE median =  result[0];


   this->trees_splits[tree][index]=median;
    vector<DataPoint> left_childs_global;
    vector<DataPoint> right_childs_global;
#pragma omp parallel
    {
        vector<DataPoint> left_childs;
        vector<DataPoint> right_childs;
       #pragma omp for  nowait
        for (int i = 0; i < data_vector.size(); i++) {
            if (data_vector[i].value <= median) {
                left_childs.push_back(data_vector[i]);
            } else {
                right_childs.push_back(data_vector[i]);
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


    int *total_counts = new int [2*this->world_size];




    int *process_counts = new int[this->world_size];
    for(int k=0;k<this->world_size;k++){
        process_counts[k]=2;
        int id = this->rank*2;
        total_counts[id]= left_childs_global.size();
        total_counts[id+1]= right_childs_global.size();
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0,MPI_INT, total_counts, process_counts, disps, MPI_INT,MPI_COMM_WORLD);


    int left_totol=0, right_total =0;
    for(int i=0;i<this->world_size;i++){
        int id = i*2;
        left_totol = left_totol+ total_counts[id];
        right_total = right_total + total_counts[id+1];
    }
    cout<<" Rank "<<rank<<" Depth "<< depth << " left child size "
        <<left_childs_global.size() << " right child size "<<  right_childs_global.size()<<" left total "<<left_totol<<" right total "<<right_total<<endl;

     free(process_counts);
     free(total_counts);
     free(disps);
    grow_global_subtree(left_childs_global,left_totol,depth+1,id_left,tree);
    grow_global_subtree(right_childs_global,right_total,depth+1,id_right,tree);
}

