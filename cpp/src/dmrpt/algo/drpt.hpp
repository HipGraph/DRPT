#ifndef DISTRIBUTED_MRPT_MRPT_H
#define DISTRIBUTED_MRPT_MRPT_H

#include <cblas.h>
#include <vector>
#include "drpt.hpp"
#include "../math/matrix_multiply.hpp"
#include <mpi.h>
#include <string>
#include <omp.h>

namespace dmrpt {
    class DRPT {
    private:
        int tree_depth;
        double *projected_matrix;
        int rows;
        int cols;
        dmrpt::StorageFormat storageFormat;
        vector<vector<double>> data;
        vector<double> splits;
        vector<int> indices;
        vector<std::vector<int>> leaf_first_indices_all;
        vector<int> leaf_first_indices;
    public:
        DRPT(double *projected_matrix, int rows, int cols, dmrpt::StorageFormat storageFormat);

        void grow_local_tree(int rank);

        void grow_local_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                                int depth, int i, int rank);

        vector<vector<double>> query(double *queryP, int rank);

        void count_leaf_sizes(int datasize, int level,  int depth,std::vector<int> &out_leaf_sizes);

        void count_first_leaf_indices(std::vector<int> &indices, int datasize, int depth);

        void count_first_leaf_indices_all(std::vector<std::vector<int>> &indices, int datasize, int depth_max);


        int getTreeDepth();
    };
}


#endif //DISTRIBUTED_MRPT_MRPT_H
