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
#include "drpt_global.hpp"
#include <chrono>
#include <algorithm>


using namespace std;
using namespace std::chrono;

dmrpt::DRPT::DRPT() {

}

dmrpt::DRPT::DRPT(VALUE_TYPE *projected_matrix, VALUE_TYPE *projection_matrix, int no_of_data_points, int tree_depth,
                  vector <vector<VALUE_TYPE>> original_data, int ntrees,
                  int starting_index, dmrpt::StorageFormat storageFormat, int rank, int world_size) {
    this->tree_depth = tree_depth;
    this->no_of_data_points = no_of_data_points;
    this->storageFormat = storageFormat;
    this->projected_matrix = projected_matrix;
    this->projection_matrix = projection_matrix;
    this->data = vector < vector < VALUE_TYPE >> (tree_depth);
    this->indices = vector<int>(no_of_data_points);
    this->original_data = original_data;

    this->ntrees = ntrees;

    this->trees_data = vector < vector < vector < VALUE_TYPE>>>(ntrees);
    this->trees_splits = vector < vector < VALUE_TYPE >> (ntrees);
    this->trees_indices = vector < vector < int >> (ntrees);
    this->trees_leaf_first_indices_all = vector < vector < vector < int>>>(ntrees);
    this->trees_leaf_first_indices = vector < vector < int >> (ntrees);

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

    if (this->tree_depth <= 0 || this->tree_depth > log2(this->no_of_data_points)) {
        throw std::out_of_range(" depth should be in range [1,....,log2(rows)]");
    }

    if (this->ntrees <= 0) {
        throw std::out_of_range(" no of trees should be greater than zero");
    }

    int total_split_size = 1 << (this->tree_depth + 1);

    if (dmrpt::StorageFormat::RAW == storageFormat) {


#pragma omp parallel for
//        {
        for (int k = 0; k < this->ntrees; k++) {
            this->count_first_leaf_indices_all(this->trees_leaf_first_indices_all[k], this->no_of_data_points,
                                               this->tree_depth);
            this->trees_leaf_first_indices[k] = this->trees_leaf_first_indices_all[k][this->tree_depth];
            cout<<"  calcualted leaf size "<< this->trees_leaf_first_indices[k].size()<<" depth "
            << this->tree_depth <<endl;
            this->trees_splits[k] = vector<VALUE_TYPE>(total_split_size);
            this->trees_data[k] = vector < vector < VALUE_TYPE >> (this->tree_depth);
            this->trees_indices[k] = vector<int>(this->no_of_data_points);;
            for (int i = 0; i < this->tree_depth; i++) {
                this->trees_data[k][i] = vector<VALUE_TYPE>(this->no_of_data_points);
                for (int j = 0; j < this->no_of_data_points; j++) {
                    int index = this->tree_depth * k + i + j * this->tree_depth * this->ntrees;
                    this->trees_data[k][i][j] = this->projected_matrix[index];
                }
            }


            iota(this->trees_indices[k].begin(), this->trees_indices[k].end(), 0);
            grow_local_subtree(this->trees_indices[k].begin(), this->trees_indices[k].end(), 0, 0, k);

        }
    }
//    }

}

void dmrpt::DRPT::grow_local_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                                     int depth, int i, const int tree) {
    int datasize = end - begin;
    int id_left = 2 * i + 1;
    int id_right = id_left + 1;

    if (depth == this->tree_depth) {
        return;
    }

    std::nth_element(begin, begin + datasize / 2, end, [this, tree, depth](int a, int b) -> bool {
        return this->trees_data[tree][depth][a] < this->trees_data[tree][depth][b];
    });

    auto mid = end - datasize / 2;


    if (datasize % 2) {
        this->trees_splits[tree][i] = this->trees_data[tree][depth][*(mid - 1)];

    } else {

        auto left = std::max_element(begin, mid, [this, tree, depth](int a, int b) -> bool {
            return this->trees_data[tree][depth][a] < this->trees_data[tree][depth][b];
        });

        this->trees_splits[tree][i] =
                (this->trees_data[tree][depth][*mid] + this->trees_data[tree][depth][*left]) / 2.0;
    }

    grow_local_subtree(begin, mid, depth + 1, id_left, tree);

    grow_local_subtree(mid, end, depth + 1, id_right, tree);

}


vector <vector<int>> dmrpt::DRPT::get_all_leaf_node_indices(int tree) {

    int leaf_size = this->trees_leaf_first_indices[tree].size();
    cout<<" leaf size for tree "<<leaf_size<<endl;
    vector<vector<int>> nodes(leaf_size);
#pragma omp parallel for
    for(int i=0;i<leaf_size;i++){
        int leaf_begin = this->trees_leaf_first_indices[tree][i];
        int leaf_end = this->trees_leaf_first_indices[tree][i + 1];
        for (int k = leaf_begin; k < leaf_end; ++k) {
            int orginal_data_index = this->trees_indices[tree][k];
            int reconstrcutedIndex = orginal_data_index + this->starting_data_index;
            if (reconstrcutedIndex < 0) {
                cout << " error in reconstructing index" << reconstrcutedIndex <<" i"<<i<< endl;
            }
            nodes[i].push_back(reconstrcutedIndex);
        }
    }
    return nodes;
}



vector <vector<int>>
dmrpt::DRPT::query(VALUE_TYPE *queryP, int no_data_points, dmrpt::StorageFormat storageFormat) {


    vector <vector<int>> vec(no_data_points);
#pragma omp parallel for
    for (int i = 0; i < no_data_points; i++) {
        vec[i] = vector<int>();
    }

    for (int m = 0; m < this->ntrees; m++) {
        for (int j = 0; j < no_data_points; ++j) {
            int idx = 0;
            for (int i = 0; i < this->tree_depth; ++i) {

                int id_left = 2 * idx + 1;
                int id_right = id_left + 1;
                VALUE_TYPE split_point = this->trees_splits[m][idx];
                int index = this->tree_depth * m + i + j * this->tree_depth * this->ntrees;
                if (queryP[index] <= split_point) {
                    idx = id_left;
                } else {
                    idx = id_right;
                }

            }

            int selected_leaf = idx - (1 << this->tree_depth) + 1;

            int leaf_begin = this->trees_leaf_first_indices[m][selected_leaf];
            int leaf_end = this->trees_leaf_first_indices[m][selected_leaf + 1];
            for (int k = leaf_begin; k < leaf_end; ++k) {
                int orginal_data_index = this->trees_indices[m][k];
                int reconstrcutedIndex = orginal_data_index + this->starting_data_index;
                if (reconstrcutedIndex < 0) {
                    cout << " error in reconstructing index" << reconstrcutedIndex
                         << " selected index " << selected_leaf << endl;
                }
                vec[j].push_back(reconstrcutedIndex);
            }


        }
    }

    return vec;
}


vector <vector<dmrpt::DataPoint>>
dmrpt::DRPT::batch_query(vector <vector<VALUE_TYPE>> queries, int batch_size, int current_master,
                         VALUE_TYPE distance_threshold) {
    int total_data_size;
    vector <vector<DataPoint>> all_results;
    if (this->rank == current_master) {
        int batchCount = 0;
        total_data_size = queries.size();
        int rounded = total_data_size / batch_size;
        int remain = total_data_size - rounded * batch_size;
        dmrpt::MathOp mathOp;
        vector <vector<VALUE_TYPE>> queryBatch;
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

    } else {

        this->receive_queries_and_evaluate_results(current_master, queries[0].size(), distance_threshold);

    }
    return all_results;
}


vector <vector<dmrpt::DataPoint>>
dmrpt::DRPT::send_query_and_receive_results(vector <vector<VALUE_TYPE>> query_batch, int batch_size,
                                            int query_dimension, VALUE_TYPE distance_threshold) {

    dmrpt::MathOp mathOp;
    vector <vector<DataPoint>> results(batch_size);

    VALUE_TYPE *querArr = mathOp.convert_to_row_major_format(query_batch);

    // P= X.R
    VALUE_TYPE *querP = mathOp.multiply_mat(querArr, this->projection_matrix, query_dimension,
                                            this->tree_depth * this->ntrees,
                                            batch_size, 1.0);

    vector <vector<int>> selectedNodes = this->query(querP, batch_size, this->storageFormat);

    int *buffer = new int[batch_size * this->world_size];
    int *counts = new int[selectedNodes.size()];

    //count selected nodes for each query locally
    vector <vector<int>> selec(selectedNodes.size());
    vector <vector<VALUE_TYPE>> selecDistances(selectedNodes.size());

    cout << " rank " << this->rank << " start sending " << endl;

    //calculate distances for each selected node
#pragma omp parallel for
//    {
    for (int m = 0; m < selectedNodes.size(); m++) {
        int cf = 0;
        selec[m] = vector<int>(selectedNodes[m].size());
        selecDistances[m] = vector<VALUE_TYPE>(selectedNodes[m].size());
        for (int w = 0; w < selectedNodes[m].size(); w++) {
            int ind = selectedNodes[m][w];
            VALUE_TYPE dist = mathOp.calculate_distance(
                    this->original_data[ind - this->starting_data_index], query_batch[m]);
            selec[m][w] = ind;
            selecDistances[m][w] = dist;

        }
        counts[m] = selectedNodes[m].size();
//        }
    }


    MPI_Bcast(&batch_size, 1, MPI_INT, this->rank, MPI_COMM_WORLD);

    int totalQ = batch_size * this->tree_depth * this->ntrees;


    MPI_Bcast(querP, totalQ, MPI_VALUE_TYPE, this->rank, MPI_COMM_WORLD);

    int totaArr = batch_size * query_dimension;

    MPI_Bcast(querArr, totaArr, MPI_VALUE_TYPE, this->rank, MPI_COMM_WORLD);

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
    int *disps = new int[this->world_size];
    // Displacement for the first chunk of data - 0
    for (int i = 0; i < this->world_size; i++) {
        disps[i] = (i > 0) ? (disps[i - 1] + process_counts[i - 1]) : 0;
    }


    int *total_recev = new int[sum];
    int *my_send = new int[process_counts[this->rank]];
    VALUE_TYPE *my_send_dis = new VALUE_TYPE[process_counts[this->rank]];
    VALUE_TYPE *total_recev_dis = new VALUE_TYPE[sum];

    int co = 0;
    for (int g = 0; g < selec.size(); g++) {
        for (int w = 0; w < selec[g].size(); w++) {
            int ind = selec[g][w];
            VALUE_TYPE dis = selecDistances[g][w];
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
    MPI_Gatherv(my_send_dis, process_counts[this->rank], MPI_VALUE_TYPE, total_recev_dis, process_counts, disps,
                MPI_VALUE_TYPE, this->rank,
                MPI_COMM_WORLD);

    cout << " rank " << this->rank << " gathering completed " << endl;

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
                                                       VALUE_TYPE distance_threshold) {
    cout << " rank " << this->rank << " start receving " << endl;
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

        vector <vector<VALUE_TYPE>> receivedOrgQ(batch_size);

//#pragma omp single
//        {
//            recev = (double *) malloc(sizeof(double) * batch_size * this->tree_depth);
//
//            originalQ = (double *) malloc(sizeof(double) * batch_size * query_dimension);

        VALUE_TYPE *recev = new VALUE_TYPE[batch_size * this->tree_depth * this->ntrees];


        VALUE_TYPE *originalQ = new VALUE_TYPE[batch_size * query_dimension];
//            recev = recevArray;
//            originalQ = originalQArray;

//        }

//        vector<double> originalQ(batch_size * query_dimension);

        int totalQ = batch_size * this->tree_depth * this->ntrees;


        MPI_Bcast(recev, totalQ, MPI_VALUE_TYPE, sending_rank, MPI_COMM_WORLD);

        vector <vector<int>> selectedNodes = this->query(recev, batch_size, this->storageFormat);


        int len_originalQ = batch_size * query_dimension;
        MPI_Bcast(originalQ, len_originalQ, MPI_VALUE_TYPE, sending_rank, MPI_COMM_WORLD);

//#pragma omp single
//        {
//        counts = (int *) malloc(sizeof(int) * selectedNodes.size());
        int counts[selectedNodes.size()];
//           counts = countsArray;
//        }
        int mytotal = 0;

        cout << " rank " << this->rank << " broadcating completed " << " count " << count << endl;
#pragma omp parallel for shared(originalQ, receivedOrgQ)
//        {
        for (int h = 0; h < batch_size; h++) {
            receivedOrgQ[h] = vector<VALUE_TYPE>(query_dimension);
            for (int e = 0; e < query_dimension; e++) {
                receivedOrgQ[h][e] = originalQ[h + e * batch_size];
            }
        }
//        }

        vector <vector<int>> selec(selectedNodes.size());
        vector <vector<VALUE_TYPE>> selecDistances(selectedNodes.size());

        cout << " rank " << this->rank << " reformation completed " << " count " << count << endl;
#pragma omp parallel for
//        {
        for (int m = 0; m < selectedNodes.size(); m++) {
            selec[m] = vector<int>(selectedNodes[m].size());
            selecDistances[m] = vector<VALUE_TYPE>(selectedNodes[m].size());
            for (int y = 0; y < selectedNodes[m].size(); y++) {
                if (selectedNodes[m][y] < 0) {
                    cout << " empty vec return " << this->rank << " index " << selectedNodes[m][y] << endl;
                }
            }
            for (int w = 0; w < selectedNodes[m].size(); w++) {
                int ind = selectedNodes[m][w];
                if (this->original_data[ind - this->starting_data_index].size() == 0) {
                    cout << " rank" << this->rank << " error prone " << " index " <<

                         (ind - this->starting_data_index) << "original index" << ind << endl;
                }
                VALUE_TYPE dist = mathOp.calculate_distance(
                        this->original_data[ind - this->starting_data_index], receivedOrgQ[m]);
                selec[m][w] = ind;
                selecDistances[m][w] = dist;

            }
            counts[m] = selectedNodes[m].size();

        }
//        }

        cout << " rank " << this->rank << " distance calculation completed " << " count " << count << endl;

        for (int b = 0; b < selectedNodes.size(); b++) {
            mytotal = mytotal + counts[b];
        }


//        cout << "Time taken for count calculation rank " << this->rank << " duration" <<
//             duration.count() << " count " << count << " microseconds" << endl;



        MPI_Gather(&counts, batch_size, MPI_INT, NULL, 1, MPI_INT, sending_rank, MPI_COMM_WORLD);

//#pragma omp single
//        {
//            my_send = (int *) malloc(sizeof(int) * mytotal);
//
//            my_send_dis = (double *) malloc(sizeof(double) * mytotal);
        int *my_send = new int[mytotal];
//            my_send = my_sendArray;

        VALUE_TYPE *my_send_dis = new VALUE_TYPE[mytotal];
//            my_send_dis = my_send_disArray;
//        }


        int co = 0;
        for (int g = 0; g < selec.size(); g++) {
            for (int w = 0; w < selec[g].size(); w++) {
                int ind = selec[g][w];
                VALUE_TYPE dis = selecDistances[g][w];
                my_send[co] = ind;
                my_send_dis[co] = dis;
                co++;

            }
        }


        MPI_Gatherv(my_send, mytotal, MPI_INT, NULL, NULL, NULL, MPI_INT, sending_rank, MPI_COMM_WORLD);

        //gather distances
        MPI_Gatherv(my_send_dis, mytotal, MPI_VALUE_TYPE, NULL, NULL, NULL, MPI_VALUE_TYPE, sending_rank,
                    MPI_COMM_WORLD);
//#pragma omp single
//        {

        cout << " rank " << this->rank << " releasing completed " << " count " << count << endl;
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

