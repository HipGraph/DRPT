#include <cblas.h>
#include <stdio.h>
#include "matrix_multiply.hpp"
#include <vector>
#include <random>
#include <mpi.h>
#include <string>
#include <iostream>
#include <omp.h>
#include <fstream>

using namespace std;


VALUE_TYPE *dmrpt::MathOp::multiply_mat(VALUE_TYPE *A, VALUE_TYPE *B, int A_rows, int B_cols, int A_cols, int alpha) {
    int result_size = A_cols * B_cols;
    VALUE_TYPE *result = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * result_size);
    for (int k = 0; k < result_size; k++) {
        result[k] = 1;
    }

//#ifdef   DOUBLE_VALUE_TYPE
//    double *Acast = reinterpret_cast<double *>(A);
//    double *Bcast = reinterpret_cast<double *>(B);
//    double *resultCast = reinterpret_cast<double *>(result);
//
//    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A_cols, B_cols, A_rows, alpha, Acast, A_cols, Bcast,
//                B_cols,
//                0.0,
//                resultCast, B_cols);
//#elifdef FLOAT_VALUE_TYPE
    float *AcastFloat = reinterpret_cast<float *>(A);
    float *BcastFloat = reinterpret_cast<float *>(B);
    float *resultCastFloat = reinterpret_cast<float *>(result);

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A_cols, B_cols, A_rows, alpha, AcastFloat, A_cols,
                BcastFloat, B_cols, 0.0,
                resultCastFloat, B_cols);
//#endif
    return result;
}

VALUE_TYPE *dmrpt::MathOp::build_sparse_local_random_matrix(int rows, int cols, float density) {
    VALUE_TYPE *A;
    int size = rows * cols;
    A = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * size);

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
                A[i + j * cols] = 0.0;
            } else {
                A[i + j * cols] = (VALUE_TYPE) norm_dist(gen);
            }
        }
    }
    return A;
}

VALUE_TYPE *dmrpt::MathOp::build_sparse_projection_matrix(int rank, int world_size, int total_dimension, int levels,
                                                          float density) {

    VALUE_TYPE *global_project_matrix;
    int local_rows;
    int length = total_dimension / world_size;
    if (rank < world_size - 1) {
        local_rows = length;
    } else if (rank == world_size - 1) {
        local_rows = total_dimension - rank * (length);
    }
    VALUE_TYPE *local_sparse_matrix = this->build_sparse_local_random_matrix(local_rows, levels, density);

    global_project_matrix = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * total_dimension * levels);


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


    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_VALUE_TYPE,
                   global_project_matrix, counts, disps, MPI_VALUE_TYPE, MPI_COMM_WORLD);


    delete[] disps;
    delete[] counts;
    free(local_sparse_matrix);

    return global_project_matrix;

}

VALUE_TYPE *dmrpt::MathOp::convert_to_row_major_format(vector <vector<VALUE_TYPE>> data) {
    if (data.empty()) {
        return (VALUE_TYPE *) malloc(0);
    }

    int cols = data.size();
    int rows = data[0].size();
    int total_size = cols * rows;

    VALUE_TYPE *arr = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * total_size);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            arr[j + i * cols] = 0.0;
        }
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            arr[j + i * cols] = data[j][i];
        }

    }
    return arr;
}

VALUE_TYPE *dmrpt::MathOp::distributed_mean(VALUE_TYPE *data, int local_rows, int local_cols, int total_elements,
                                            dmrpt::StorageFormat format, int rank) {
    int size = local_rows * local_cols;
    VALUE_TYPE *sums = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * local_cols);
    VALUE_TYPE *gsums = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * local_cols);
    for (int i = 0; i < local_cols; i++) {
        sums[i] = 0.0;
    }
    if (format == dmrpt::StorageFormat::RAW) {
        for (int i = 0; i < local_cols; i++) {
            VALUE_TYPE sum = 0.0;
            for (int j = 0; j < local_rows; j++) {
                sum = sum + data[i + j * local_cols];
            }
            sums[i] = sum;
        }
    }
    MPI_Allreduce(sums, gsums, local_cols, MPI_VALUE_TYPE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < local_cols; i++) {
        gsums[i] = gsums[i] / total_elements;
    }
    free(sums);
    return gsums;
}

VALUE_TYPE *dmrpt::MathOp::distributed_variance(VALUE_TYPE *data, int local_rows, int local_cols, int total_elements,
                                                dmrpt::StorageFormat format, int rank) {
    VALUE_TYPE *means = this->distributed_mean(data, local_rows, local_cols, total_elements, format, rank);
    int size = local_rows * local_cols;
    VALUE_TYPE *var = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * local_cols);
    VALUE_TYPE *gvariance = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * local_cols);
    for (int i = 0; i < local_cols; i++) {
        var[i] = 0.0;
    }
    if (format == dmrpt::StorageFormat::RAW) {
        for (int i = 0; i < local_cols; i++) {
            VALUE_TYPE sum = 0.0;
            for (int j = 0; j < local_rows; j++) {
                VALUE_TYPE diff = (data[i + j * local_cols] - means[i]);
                sum = sum + (diff * diff);
            }
            var[i] = sum;
        }
    }
    MPI_Allreduce(var, gvariance, local_cols, MPI_VALUE_TYPE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < local_cols; i++) {
        gvariance[i] = gvariance[i] / (total_elements - 1);
//        cout<<"Rank "<<rank<<"Variance "<< gvariance[i]<<" "<<endl;
    }
    free(means);
    return gvariance;
}


VALUE_TYPE *
dmrpt::MathOp::distributed_median(VALUE_TYPE *data, int local_rows, int local_cols, int total_elements, int no_of_bins,
                                  dmrpt::StorageFormat format, int rank) {
    VALUE_TYPE *means = this->distributed_mean(data, local_rows, local_cols, total_elements, format, rank);
    VALUE_TYPE *variance = this->distributed_variance(data, local_rows, local_cols, total_elements, format, rank);
    VALUE_TYPE *medians = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * local_cols);
    int size = local_rows * local_cols;

    int std1 = 4, std2 = 2, std3 = 1;

    for (int k = 0; k < local_cols; k++) {
        medians[k] = INFINITY;
    }

    if (format == dmrpt::StorageFormat::RAW) {

        for (int i = 0; i < local_cols; i++) {
            VALUE_TYPE mu = means[i];
            VALUE_TYPE sigma = variance[i];
            sigma = sqrt(sigma);
//            cout << "Col " << i << " mean " << mu << " variance " << sigma << endl;
            VALUE_TYPE val = 0.0;
            int factor = (int) ceil(no_of_bins * 1.0 / (std1 + std2 + std3));

            vector<VALUE_TYPE> distribution(2 * factor * (std1 + std2 + std3) + 2, 0);

            vector<int> frequency(2 * factor * (std1 + std2 + std3) + 2, 0);

            int start1 = factor * (std1 + std2 + std3);
            VALUE_TYPE step1 = sigma / (2 * pow(2, std1 * factor) - 2);

            for (int k = start1, j = 1; k < start1 + std1 * factor; k++, j++) {
                VALUE_TYPE rate = j * step1;
                distribution[k] = mu + rate;
                distribution[start1 - j] = mu - rate;
            }

            int start2 = start1 + std1 * factor;
            int rstart2 = start1 - std1 * factor;
            VALUE_TYPE step2 = sigma / (2 * pow(2, std2 * factor) - 2);

            for (int k = start2, j = 1; k < start2 + std2 * factor; k++, j++) {
                VALUE_TYPE rate = sigma + j * step2;
                distribution[k] = mu + rate;
                distribution[rstart2 - j] = mu - rate;
            }

            int start3 = start2 + std2 * factor;
            int rstart3 = rstart2 - std2 * factor;
            VALUE_TYPE step3 = sigma / (2 * pow(2, std3 * factor) - 2);

            for (int k = start3, j = 1; k < start3 + std3 * factor; k++, j++) {
                VALUE_TYPE rate = 2 * sigma + j * step3;
                distribution[k] = mu + rate;
                distribution[rstart3 - j] = mu - rate;
            }


            for (int k = 0; k < local_rows; k++) {
                int flag = 1;
                for (int j = 1; j < 2 * no_of_bins + 2; j++) {
                    VALUE_TYPE dval = data[i + k * local_cols];
                    if (distribution[j - 1] < dval && distribution[j] >= dval) {
                        flag = 0;
                        frequency[j] += 1;
                    }
                }
                if (flag) {
                    frequency[0] += 1;
                }
            }

            int *gfrequency = (int *) malloc(sizeof(int) * distribution.size());
            int *freqarray = (int *) malloc(sizeof(int) * distribution.size());
            for (int k = 0; k < distribution.size(); k++) {
                freqarray[k] = frequency[k];
                gfrequency[k] = 0;
            }


            MPI_Allreduce(freqarray, gfrequency, distribution.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);

//            for(int k=0;k<distribution.size();k++){
//                if (rank==0) {
//                    cout << "k "<<k<<" distribution " << distribution[k] << " Frequency " << gfrequency[k] << endl;
//                }
//            }

            VALUE_TYPE cfreq = 0;
            VALUE_TYPE cper = 0;
            int selected_index = -1;
            for (int k = 1; k < distribution.size(); k++) {
                cfreq += gfrequency[k];
                cper += gfrequency[k] * 100 / total_elements;
                if (cper > 50) {
                    selected_index = k;
                    break;
                }
            }

            int count = gfrequency[selected_index];


            VALUE_TYPE median = distribution[selected_index - 1] +
                                ((total_elements / 2 - (cfreq - count)) / count) *
                                (distribution[selected_index] - distribution[selected_index - 1]);
            medians[i] = median;

            free(gfrequency);
            free(freqarray);

        }
    }

    return medians;
}

VALUE_TYPE dmrpt::MathOp::calculate_distance(vector<VALUE_TYPE> data, vector<VALUE_TYPE> query) {
//    std::vector<VALUE_TYPE> auxiliary(data.size());
//
//    std::transform(data.begin(), data.end(), query.begin(), std::back_inserter(auxiliary),//
//                   [](VALUE_TYPE element1, VALUE_TYPE element2) { return pow((element1 - element2), 2); });
//
//    VALUE_TYPE value = sqrt(std::accumulate(auxiliary.begin(), auxiliary.end(), 0.0));
//    data.clear();
//    query.clear();
//    auxiliary.clear();
    if (data.size() != query.size()) {
        cout << " wrong length vector data size" << data.size() << " query size" << query.size() << endl;
    }

    int sum = 0;
//#pragma omp parallel for reduction(+ :sum)
    for (int n = 0; n < query.size(); n++) {

        sum += pow((data[n] - query[n]), 2);
    }

    return sqrt(sum);
}


