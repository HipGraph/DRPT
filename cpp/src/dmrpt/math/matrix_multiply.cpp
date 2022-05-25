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


double *dmrpt::MathOp::multiply_mat(vector <vector<double>> A, vector <vector<double>> B) {
    int i = 0;
    double E[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
    double G[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
    double K[9] = {.5, .5, .5, .5, .5, .5, .5, .5, .5};
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 3, 3, 2, 1, E, 3, G, 3, 2, K, 3);

    for (i = 0; i < 9; i++)
        printf("%lf ", K[i]);
    printf("\n");
    double *F;
    return F;
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
                A[j * cols + i] = 0.0;
            } else {
                A[j * cols + i] = (double) norm_dist(gen);
            }
        }
    }
    return A;
}

double *dmrpt::MathOp::build_sparse_projection_matrix(int rank, int world_size, int total_dimension, int levels, float density) {

    double *global_project_matrix;
    int local_rows;
    int length = total_dimension / world_size;
    if (rank < world_size - 1) {
        local_rows = length;
    } else if (rank == world_size - 1) {
        local_rows = total_dimension - rank * (length);
    }
    double *local_sparse_matrix = this->build_sparse_local_random_matrix(local_rows, levels, density);

    global_project_matrix = (double *)malloc(sizeof(double) * total_dimension * levels);


    int my_start, my_end;
    if (rank < world_size - 1) {
        my_start = rank* length*levels;
        my_end = my_start+ length*levels - 1;
    } else if (rank == world_size - 1) {
        my_start = rank* length*levels;
        my_end = total_dimension*levels - 1;
    }

    for(int i=my_start;i<=my_end;i++){
        global_project_matrix[i] =local_sparse_matrix[i-my_start];
    }

    int my_total = local_rows*levels;

    int *counts = new int[world_size];
    int *disps  = new int[world_size];

    for (int i=0; i<world_size-1; i++)
        counts[i] = length*levels;
    counts[world_size-1] = total_dimension*levels - length*levels*(world_size-1);

    disps[0] = 0;
    for (int i=1; i<world_size; i++)
        disps[i] = disps[i-1] + counts[i-1];

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE,
                   global_project_matrix, counts, disps, MPI_DOUBLE, MPI_COMM_WORLD);

    delete [] disps;
    delete [] counts;
    free(local_sparse_matrix);

    return global_project_matrix;

}


