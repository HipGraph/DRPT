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

dmrpt::DRPT::DRPT() {

}

dmrpt::DRPT::DRPT(double *projected_matrix, double *projection_matrix, int rows, int cols,
                  vector <vector<double>> original_data,
                  int starting_index, dmrpt::StorageFormat storageFormat, int rank, int world_size) {
    this->tree_depth = cols;
    this->rows = rows;
    this->cols = cols;
    this->storageFormat = storageFormat;
    this->projected_matrix = projected_matrix;
    this->projection_matrix = projection_matrix;
    this->data = vector < vector < double >> (cols);
    this->indices = vector<int>(rows);
    this->original_data = original_data;
    this->starting_data_index = starting_index;
    this->rank = rank;
    this->world_size = world_size;

}


void dmrpt::DRPT::count_leaf_sizes(int datasize, int level, int depth, std::vector<int> &out_leaf_sizes) {
    if (level == depth) {
        out_leaf_sizes.push_back(datasize);
        return;
    }

    this->count_leaf_sizes(datasize - datasize / 2, level + 1, depth, out_leaf_sizes);
    this->count_leaf_sizes(datasize / 2, level + 1, depth, out_leaf_sizes);
}


void dmrpt::DRPT::count_first_leaf_indices(std::vector<int> &indices, int datasize, int depth) {
    std::vector<int> leaf_sizes;
    this->count_leaf_sizes(datasize, 0, depth, leaf_sizes);
    indices = std::vector<int>(leaf_sizes.size() + 1);
    indices[0] = 0;
    for (int i = 0; i < (int) leaf_sizes.size(); ++i)
        indices[i + 1] = indices[i] + leaf_sizes[i];
}

void dmrpt::DRPT::count_first_leaf_indices_all(std::vector <std::vector<int>> &indices, int datasize, int depth_max) {
    for (int d = 0; d <= depth_max; ++d) {
        std::vector<int> idx;
        this->count_first_leaf_indices(idx, datasize, d);
        indices.push_back(idx);
    }
}


void dmrpt::DRPT::grow_local_tree() {

    if (tree_depth <= 0 || tree_depth > log2(rows)) {
        throw std::out_of_range(" depth should be in range [1,....,log2(rows)]");
    }
    int total_split_size = 1 << (tree_depth + 1);
    this->splits = vector<double>(total_split_size);

    if (dmrpt::StorageFormat::RAW == storageFormat) {

        this->count_first_leaf_indices_all(this->leaf_first_indices_all, rows, this->tree_depth);
        this->leaf_first_indices = this->leaf_first_indices_all[this->tree_depth];

#pragma omp parallel for shared(this->data)
        for (int i = 0; i < cols; i++) {
            this->data[i] = vector<double>(rows);
            for (int j = 0; j < rows; j++) {
                this->data[i][j] = this->projected_matrix[i + j * cols];
            }
        }

        iota(this->indices.begin(), this->indices.end(), 0);
        grow_local_subtree(this->indices.begin(), this->indices.end(), 0, 0);

    }

}

void dmrpt::DRPT::grow_local_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                                     int depth, int i) {
    int datasize = end - begin;
    int id_left = 2 * i + 1;
    int id_right = id_left + 1;

    if (depth == tree_depth) {
        return;
    }

    std::nth_element(begin, begin + datasize / 2, end, [this, depth](int a, int b) -> bool {
        return this->data[depth][a] < this->data[depth][b];
    });

    auto mid = end - datasize / 2;


    if (datasize % 2) {
        this->splits[i] = this->data[depth][*(mid - 1)];

    } else {

        auto left = std::max_element(begin, mid, [this, depth](int a, int b) -> bool {
            return this->data[depth][a] < this->data[depth][b];
        });

        this->splits[i] = (this->data[depth][*mid] + this->data[depth][*left]) / 2.0;
    }

    grow_local_subtree(begin, mid, depth + 1, id_left);

    grow_local_subtree(mid, end, depth + 1, id_right);

}

vector <vector<int>>
dmrpt::DRPT::query(double *queryP, int no_datapoints, dmrpt::StorageFormat storageFormat) {


    vector <vector<int>> vec(no_datapoints);

    if (storageFormat == dmrpt::StorageFormat::RAW) {
#pragma  omp parallel for
        for (int j = 0; j < no_datapoints; ++j) {
            int idx = 0;
            for (int i = 0; i < this->tree_depth; ++i) {

                int id_left = 2 * idx + 1;
                int id_right = id_left + 1;
                double split_point = this->splits[idx];
                if (queryP[i + j * this->tree_depth] <= split_point) {
                    idx = id_left;
                } else {
                    idx = id_right;
                }

            }

            int selected_leaf = idx - (1 << this->tree_depth) + 1;

            int leaf_begin = this->leaf_first_indices[selected_leaf];
            int leaf_end = this->leaf_first_indices[selected_leaf + 1];
            vec[j] = vector<int>();
            for (int k = leaf_begin; k < leaf_end; ++k) {
                int orginal_data_index = this->indices[k];
                int reconstrcutedIndex = orginal_data_index + this->starting_data_index;
                vec[j].push_back(reconstrcutedIndex);
            }
        }
    }

    return vec;
}


vector <vector<dmrpt::DRPT::DataPoint>>
dmrpt::DRPT::batch_query(vector <vector<double>> queries, int batch_size, int current_master,
                         double distance_threshold) {
    int total_data_size;
    vector <vector<DataPoint>> all_results;
    if (this->rank == current_master) {
        int batchCount = 0;
        total_data_size = queries.size();
        int rounded = total_data_size / batch_size;
        int remain = total_data_size - rounded * batch_size;
        dmrpt::MathOp mathOp;
        vector <vector<double>> queryBatch;
        MPI_Bcast(&total_data_size, 1, MPI_INT, current_master, MPI_COMM_WORLD);
        int count = 0;

        while (count < total_data_size) {
            int sending_count = 0;
            if (count <= (rounded - 1) * batch_size) {
                sending_count = batch_size;
                queryBatch.insert(queryBatch.end(), std::make_move_iterator(queries.begin() + count),
                                  std::make_move_iterator(queries.begin() + count + batch_size));
//                queries.erase(queries.begin() + count, queries.begin() + count + batch_size);
            } else if (count > (rounded - 1) * batch_size) {
                sending_count = remain;
                queryBatch.insert(queryBatch.end(), std::make_move_iterator(queries.begin() + count),
                                  std::make_move_iterator(queries.begin() + count + remain));
//                queries.erase(queries.begin() + count, queries.begin() + count + remain);

            }
            vector <vector<DataPoint>> results = this->send_query_and_receive_results(queryBatch, sending_count,
                                                                                      queryBatch[0].size(),
                                                                                      distance_threshold);
            count = count + sending_count;

            for (int j = 0; j < results.size(); j++) {
                all_results.push_back(results[j]);
            }
            queryBatch.clear();
        }

//        //TODO: improve using chunk copying
//        for (int k = 1; k <= total_data_size; k++) {
//            if (k % batch_size == 0 and k <= rounded * batch_size) {
//                queryBatch.push_back(queries[k - 1]);
//                vector <vector<DataPoint>> results = this->send_query_and_receive_results(queryBatch,
//                                                                                          batch_size,
//                                                                                          queries[0].size(),
//                                                                                          distance_threshold);
//                for (int j = 0; j < results.size(); j++) {
//                    all_results.push_back(results[j]);
//                }
//
//                queryBatch.clear();
//            } else if (k == total_data_size && remain > 0) {
//                vector <vector<DataPoint>> results = this->send_query_and_receive_results(queryBatch,
//                                                                                          remain,
//                                                                                          queries[0].size(),
//                                                                                          distance_threshold);
//                #pragma  omp parallel for
//                for (int j = 0; j < results.size(); j++) {
//                    all_results.push_back(results[j]);
//                }
//                queryBatch.clear();
//            } else {
//                queryBatch.push_back(queries[k - 1]);
//            }
//        }

    } else {
        this->receive_queries_and_evaluate_results(current_master, queries[0].size(), distance_threshold);

    }
    return all_results;
}


vector <vector<dmrpt::DRPT::DataPoint>>
dmrpt::DRPT::send_query_and_receive_results(vector <vector<double>> query_batch, int batch_size,
                                            int query_dimension, double distance_threshold) {

    dmrpt::MathOp mathOp;
    vector <vector<DataPoint>> results(batch_size);

    double *querArr = mathOp.convert_to_row_major_format(query_batch);

    // P= X.R
    double *querP = mathOp.multiply_mat(querArr, this->projection_matrix, query_dimension, this->tree_depth,
                                        batch_size, 1.0);


    vector <vector<int>> selectedNodes = this->query(querP, batch_size, this->storageFormat);


    int* buffer= new int[batch_size * this->world_size];
    int* counts = new int[selectedNodes.size()];

    //count selected nodes for each query locally
    vector <vector<int>> selec(selectedNodes.size());
    vector <vector<double>> selecDistances(selectedNodes.size());

    //calculate distances for each selected node
#pragma omp parallel for shared(this->original_data, query_batch, selec, selecDistances,counts,selectedNodes)
    {
        for (int m = 0; m < selectedNodes.size(); m++) {
            int cf = 0;
            selec[m] = vector<int>(selectedNodes[m].size());
            selecDistances[m] = vector<double>(selectedNodes[m].size());
            for (int w = 0; w < selectedNodes[m].size(); w++) {
                int ind = selectedNodes[m][w];
                double dist = mathOp.calculate_distance(
                        this->original_data[ind - this->rank * this->starting_data_index], query_batch[m]);
                selec[m][w] = ind;
                selecDistances[m][w] = dist;

            }
            counts[m] = selectedNodes[m].size();
        }
    }


    MPI_Bcast(&batch_size, 1, MPI_INT, this->rank, MPI_COMM_WORLD);

    int totalQ = batch_size * this->tree_depth;


    MPI_Bcast(querP, totalQ, MPI_DOUBLE, this->rank, MPI_COMM_WORLD);

    int totaArr = batch_size * query_dimension;


    MPI_Bcast(querArr, totaArr, MPI_DOUBLE, this->rank, MPI_COMM_WORLD);

    MPI_Gather(counts, batch_size, MPI_INT, buffer, batch_size, MPI_INT, this->rank, MPI_COMM_WORLD);

    int sum = 0;
    //each processor selected nodes counts
    int *process_counts = new int[this->world_size];
    for (int n = 0; n < this->world_size; n++) {
        int process_count = 0;
        for (int m = 0; m < batch_size; m++) {
            sum = sum + buffer[m + n * batch_size];
            process_count = process_count + buffer[m + n * batch_size];;
        }
        process_counts[n] = process_count;
    }

    // Displacements in the receive buffer for MPI_GATHERV
    int *disps  = new int[this->world_size];
    // Displacement for the first chunk of data - 0
    for (int i = 0; i < this->world_size; i++) {
        disps[i] = (i > 0) ? (disps[i - 1] + process_counts[i - 1]) : 0;
    }


    int* total_recev= new int[sum];
    int* my_send = new int[process_counts[this->rank]];
    double* my_send_dis= new double[process_counts[this->rank]];
    double* total_recev_dis= new double[sum];

    int co = 0;
    for (int g = 0; g < selec.size(); g++) {
        for (int w = 0; w < selec[g].size(); w++) {
            int ind = selec[g][w];
            double dis = selecDistances[g][w];
            my_send[co] = ind;
            my_send_dis[co] = dis;
            co++;
        }
    }



    //send indices of selected nodes
    MPI_Gatherv(my_send, process_counts[this->rank], MPI_INT, total_recev, process_counts, disps, MPI_INT,
                this->rank,
                MPI_COMM_WORLD);

    //gather distances
    MPI_Gatherv(my_send_dis, process_counts[this->rank], MPI_DOUBLE, total_recev_dis, process_counts, disps,
                MPI_DOUBLE, this->rank,
                MPI_COMM_WORLD);


    //reconstructed received nns from MPI calls
    int last_process_count[this->world_size];
    for (int p = 0; p < batch_size; ++p) {
        results[p] = vector<DataPoint>();
        for (int u = 0; u < this->world_size; u++) {
            if (p == 0) {
                last_process_count[u] = 0;
            }
            int localtot = buffer[p + u * batch_size];
            int start = disps[u] + last_process_count[u];
            int end = start + localtot;
            for (int b = start; b < end; b++) {
                DataPoint dataPoint;
                dataPoint.index = total_recev[b];
                dataPoint.distance = total_recev_dis[b];
                results[p].push_back(dataPoint);
            }
            last_process_count[u] = last_process_count[u] + localtot;
        }
    }
    selec.clear();
    free(querArr);
    free(querP);
    free(buffer);
    free(counts);
    free(disps);
    free(process_counts);
    free(total_recev);
    free(my_send);
    return results;
}


void dmrpt::DRPT::receive_queries_and_evaluate_results(int sending_rank, int query_dimension,
                                                       double distance_threshold) {
    int total_data_size;
    MPI_Bcast(&total_data_size, 1, MPI_INT, sending_rank, MPI_COMM_WORLD);
    int count = 0;
    dmrpt::MathOp mathOp;
//    double *recev = (double *) malloc(sizeof(double) * total_data_size * this->tree_depth);
//    double *originalQ = (double *) malloc(sizeof(double) * total_data_size * query_dimension);
//    int *counts = (int *) malloc(sizeof(int) * total_data_size);
//    int *my_send = (int *) malloc(sizeof(int) * total_data_size);
//    double *my_send_dis = (double *) malloc(sizeof(double) * total_data_size);
    while (count < total_data_size) {

        int batch_size;
        MPI_Bcast(&batch_size, 1, MPI_INT, sending_rank, MPI_COMM_WORLD);
//        double *recev;
//        double *originalQ;
//        int *counts;
//        int *my_send;
//        double *my_send_dis;

        vector <vector<double>> receivedOrgQ(batch_size);

//#pragma omp single
//        {
//            recev = (double *) malloc(sizeof(double) * batch_size * this->tree_depth);
//
//            originalQ = (double *) malloc(sizeof(double) * batch_size * query_dimension);

        double* recev = new double[batch_size * this->tree_depth];



        double* originalQ = new double [batch_size * query_dimension];
//            recev = recevArray;
//            originalQ = originalQArray;

//        }

//        vector<double> originalQ(batch_size * query_dimension);

        int totalQ = batch_size * this->tree_depth;


        MPI_Bcast(recev, totalQ, MPI_DOUBLE, sending_rank, MPI_COMM_WORLD);

        vector <vector<int>> selectedNodes = this->query(recev, batch_size, this->storageFormat);
        int len_originalQ = batch_size * query_dimension;
        MPI_Bcast(originalQ, len_originalQ, MPI_DOUBLE, sending_rank, MPI_COMM_WORLD);

//#pragma omp single
//        {
//        counts = (int *) malloc(sizeof(int) * selectedNodes.size());
        int counts[selectedNodes.size()];
//           counts = countsArray;
//        }
        int mytotal = 0;


#pragma omp parallel for shared(originalQ, receivedOrgQ)
        {
            for (int h = 0; h < batch_size; h++) {
                receivedOrgQ[h] = vector<double>(query_dimension);
                for (int e = 0; e < query_dimension; e++) {
                    receivedOrgQ[h][e] = originalQ[h + e * batch_size];
                }
            }
        }

        vector <vector<int>> selec(selectedNodes.size());
        vector <vector<double>> selecDistances(selectedNodes.size());

#pragma omp parallel for shared(this->original_data, receivedOrgQ, selec, selecDistances, counts, mytotal)
        {
            for (int m = 0; m < selectedNodes.size(); m++) {
                selec[m] = vector<int>(selectedNodes[m].size());
                selecDistances[m] = vector<double>(selectedNodes[m].size());
                for (int w = 0; w < selectedNodes[m].size(); w++) {
                    int ind = selectedNodes[m][w];
                    double dist = mathOp.calculate_distance(
                            this->original_data[ind - this->rank * this->starting_data_index], receivedOrgQ[m]);
                    selec[m][w] = ind;
                    selecDistances[m][w] = dist;

                }
                counts[m] = selectedNodes[m].size();

            }
        }

        for(int b=0;b<selectedNodes.size();b++){
            mytotal = mytotal+counts[b];
        }


//        cout << "Time taken for count calculation rank " << this->rank << " duration" <<
//             duration.count() << " count " << count << " microseconds" << endl;



        MPI_Gather(&counts, batch_size, MPI_INT, NULL, 1, MPI_INT, sending_rank, MPI_COMM_WORLD);

//#pragma omp single
//        {
//            my_send = (int *) malloc(sizeof(int) * mytotal);
//
//            my_send_dis = (double *) malloc(sizeof(double) * mytotal);
        int*  my_send = new int[mytotal];
//            my_send = my_sendArray;

        double* my_send_dis = new double[mytotal];
//            my_send_dis = my_send_disArray;
//        }


        int co = 0;
        for (int g = 0; g < selec.size(); g++) {
            for (int w = 0; w < selec[g].size(); w++) {
                int ind = selec[g][w];
                double dis = selecDistances[g][w];
                my_send[co] = ind;
                my_send_dis[co] = dis;
                co++;

            }
        }


        MPI_Gatherv(my_send, mytotal, MPI_INT, NULL, NULL, NULL, MPI_INT, sending_rank, MPI_COMM_WORLD);

        //gather distances
        MPI_Gatherv(my_send_dis, mytotal, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, sending_rank,
                    MPI_COMM_WORLD);
//#pragma omp single
//        {
        count = count + batch_size;
        receivedOrgQ.clear();
        selectedNodes.clear();
        free(originalQ);
        free(recev);
        free(my_send);
        free(my_send_dis);
//            free(originalQ);
//            free(recev);
//            delete counts;
//            delete my_send;
//            delete my_send_dis;
//            delete originalQ;
//            delete recev;

//        }


    }


}

