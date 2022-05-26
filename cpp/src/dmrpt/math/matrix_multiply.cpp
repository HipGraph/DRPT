#include <cblas.h>
#include <stdio.h>
#include "matrix_multiply.hpp"
#include <vector>
#include <random>
#include <mpi.h>
#include <string>
#include <iostream>
#include <omp.h>

using namespace std;


double *dmrpt::MathOp::multiply_mat(double *A, double *B, int A_rows, int B_cols,int A_cols,int alpha) {
    int result_size = A_cols*B_cols;
    double *result = (double *) malloc(sizeof (double )*result_size);
    for(int k=0;k<result_size;k++){
        result[k]=1;
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A_cols, B_cols, A_rows, alpha, A, A_cols, B, B_cols, 0.0, result, B_cols);

   return result;
}

double *dmrpt::MathOp::build_sparse_local_random_matrix(int rows, int cols, float density) {
    double *A;
    int size = rows * cols;
    A = (double *) malloc(sizeof(double) * size);

    int seed = 0;
    std::random_device rd;
    int s = seed ? seed : rd();
    std::mt19937 gen(s);
    std::uniform_real_distribution<float> uni_dist(0, 1);
    std::normal_distribution<float> norm_dist(0, 1);

    //follow row major order
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            if (uni_dist(gen) > density) {
                A[i+ j*cols] = 0.0;
            } else {
                A[i+ j*cols] = (double) norm_dist(gen);
            }
        }
    }
    return A;
}

double *dmrpt::MathOp::build_sparse_projection_matrix(int rank, int world_size, int total_dimension, int levels,
                                                      float density) {

    double *global_project_matrix;
    int local_rows;
    int length = total_dimension / world_size;
    if (rank < world_size - 1) {
        local_rows = length;
    } else if (rank == world_size - 1) {
        local_rows = total_dimension - rank * (length);
    }
    double *local_sparse_matrix = this->build_sparse_local_random_matrix(local_rows, levels, density);

    global_project_matrix = (double *) malloc(sizeof(double) * total_dimension * levels);


    int my_start, my_end;
    if (rank < world_size - 1) {
        my_start = rank * length * levels;
        my_end = my_start + length * levels - 1;
    } else if (rank == world_size - 1) {
        my_start = rank * length * levels;
        my_end = total_dimension * levels - 1;
    }

    for (int i = my_start; i <= my_end; i++) {
        global_project_matrix[i] = local_sparse_matrix[i - my_start];
    }

    int my_total = local_rows * levels;

    int *counts = new int[world_size];
    int *disps = new int[world_size];

    for (int i = 0; i < world_size - 1; i++)
        counts[i] = length * levels;
    counts[world_size - 1] = total_dimension * levels - length * levels * (world_size - 1);

    disps[0] = 0;
    for (int i = 1; i < world_size; i++)
        disps[i] = disps[i - 1] + counts[i - 1];

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE,
                   global_project_matrix, counts, disps, MPI_DOUBLE, MPI_COMM_WORLD);

    delete[] disps;
    delete[] counts;
    free(local_sparse_matrix);

    return global_project_matrix;

}

double *dmrpt::MathOp::convert_to_row_major_format(vector <vector<double>> data) {
    int cols = data.size();
    int rows = data[0].size();
    int total_size = cols * rows;

    double *arr = (double *) malloc(sizeof(double) * total_size);
    for (int i = 0; i < rows;  i++) {
        for (int j = 0; j < cols;  j++) {
            arr[j + i * cols] = 0.0;
        }
    }

    for (int i = 0; i < rows;  i++) {
        for (int j = 0; j < cols;  j++) {
           arr[j + i * cols] = data[j][i];
        }

    }
    return arr;
}


