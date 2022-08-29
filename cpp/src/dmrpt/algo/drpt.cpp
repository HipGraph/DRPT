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
                  int starting_index, int rank, int world_size) {
    this->tree_depth = tree_depth;
    this->no_of_data_points = no_of_data_points;
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

    for (int k = 0; k < this->ntrees; k++) {
        this->count_first_leaf_indices_all(this->trees_leaf_first_indices_all[k], this->no_of_data_points,
                                           this->tree_depth);
        this->trees_leaf_first_indices[k] = this->trees_leaf_first_indices_all[k][this->tree_depth];
        this->trees_splits[k] = vector<VALUE_TYPE>(total_split_size);
        this->trees_data[k] = vector < vector < VALUE_TYPE >> (this->tree_depth);
        this->trees_indices[k] = vector<int>(this->no_of_data_points);;
        for (int i = 0; i < this->tree_depth; i++) {
            this->trees_data[k][i] = vector<VALUE_TYPE>(this->no_of_data_points);
#pragma omp parallel for
            for (int j = 0; j < this->no_of_data_points; j++) {
                int index = this->tree_depth * k + i + j * this->tree_depth * this->ntrees;
                this->trees_data[k][i][j] = this->projected_matrix[index];
            }
        }


        iota(this->trees_indices[k].begin(), this->trees_indices[k].end(), 0);
        grow_local_subtree(this->trees_indices[k].begin(), this->trees_indices[k].end(), 0, 0, k);

    }


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
    vector <vector<int>> nodes(leaf_size - 1);
#pragma omp parallel for
    for (int i = 0; i < leaf_size - 1; i++) {
        int leaf_begin = this->trees_leaf_first_indices[tree][i];
        int leaf_end = this->trees_leaf_first_indices[tree][i + 1];
        for (int k = leaf_begin; k < leaf_end; ++k) {
            int orginal_data_index = this->trees_indices[tree][k];
            nodes[i].push_back(orginal_data_index);
        }
    }
    return nodes;
}

