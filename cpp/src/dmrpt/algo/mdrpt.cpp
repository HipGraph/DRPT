#include "mdrpt.hpp"
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
#include <map>
#include <unordered_map>
#include <fstream>


using namespace std;

dmrpt::MDRPT::MDRPT(int ntrees, vector<vector<double>> original_data,int tree_depth,dmrpt::StorageFormat storageFormat, int rank, int world_size) {
    this->data_dimension= original_data[0].size();
    this->tree_depth=tree_depth;
    this->original_data=original_data;
    this->storageFormat=storageFormat;
    this->rank=rank;
    this->world_size=world_size;
    this->ntrees=ntrees;
    this->trees= vector<dmrpt::DRPT>(ntrees);
}

void dmrpt::MDRPT::grow_trees(float density) {

    dmrpt::MathOp mathOp;
    double *imdataArr = mathOp.convert_to_row_major_format(this->original_data);

    int rows = this->original_data[0].size();
    int cols = this->original_data.size();

    this->trees.clear();


    for(int i=0; i<this->ntrees; i++) {

       double *B = mathOp.build_sparse_projection_matrix(this->rank, this->world_size, this->data_dimension, this->tree_depth, density);
        // P= X.R
        double *P = mathOp.multiply_mat(imdataArr, B, rows, this->tree_depth, cols, 1.0);

        int starting_index = this->rank * cols/world_size;
        this->trees[i]= dmrpt::DRPT(P,B, cols, this->tree_depth, this->original_data, starting_index,this->storageFormat, this->rank,this->world_size);
        this->trees[i].grow_local_tree(this->rank);
    }

}


vector <vector<dmrpt::DRPT::DataPoint>> dmrpt::MDRPT::batchQuery(int batch_size, double distance_threshold, int nn) {
    vector<vector<dmrpt::DRPT::DataPoint>> results(this->original_data.size());
    for(int i=0;i<ntrees;i++){
        for(int j=0;j<this->world_size;j++){
           vector<vector<dmrpt::DRPT::DataPoint>> result =  this->trees[i].batchQuery(this->original_data,batch_size,j,distance_threshold);
           for(int k=0;k<result.size();k++){
              results[k].insert(results[k].end(),result[k].begin(),result[k].end());
           }
        }
    }


   cout<< " rank "<< rank << " size "<<results.size()<<endl;

   return   this->get_vote_results(results,2,nn);

}


template<typename T>
vector<T> slice(vector<T> const &v, int m, int n)
{
auto first = v.cbegin() + m;
auto last = v.cbegin() + n + 1;

std::vector<T> vec(first, last);
return vec;
}




vector <vector<dmrpt::DRPT::DataPoint>>
dmrpt::MDRPT::get_vote_results(vector <vector<dmrpt::DRPT::DataPoint>> results, int vote_threshold, int nn) {

    vector<unordered_map<int, int>> voted_results(results.size());
    vector<vector<dmrpt::DRPT::DataPoint>> voted_selected(results.size());

    for(int i=0;i<results.size();i++){
        voted_results[i]= unordered_map<int,int>();
        voted_selected[i]= vector<dmrpt::DRPT::DataPoint>();

        vector<dmrpt::DRPT::DataPoint> local_selected = results[i];


        sort(local_selected.begin(), local_selected.end(), [](const dmrpt::DRPT::DataPoint& lhs, const dmrpt::DRPT::DataPoint& rhs) {
            return lhs.distance < rhs.distance;
        });

        for(int j=0;j<local_selected.size();j++){
            unordered_map<int, int>::iterator it = voted_results[i].find(local_selected[j].index);
//            // key already present on the map
            if (it !=  voted_results[i].end()) {
                it->second++;// increment map's value for key `c`
                if(it->second>= vote_threshold){
                    voted_selected[i].push_back(local_selected[j]);
                }
            }
//                // key not found
           else {
                voted_results[i].insert(std::make_pair(local_selected[j].index, 1));
            }

        }

        voted_selected[i].erase( unique( voted_selected[i].begin(), voted_selected[i].end(),[](const dmrpt::DRPT::DataPoint& lhs, const dmrpt::DRPT::DataPoint& rhs) {
            return lhs.index == rhs.index;
        }) , voted_selected[i].end());

//        cout<< " rank "<<rank<< " index "<<i<< " size "<< voted_selected[i].size() <<endl;
//        vector<dmrpt::DRPT::DataPoint> sub_vec = slice(voted_selected[i], 0, nn-1);
//
//        cout<< " rank "<<rank<< " index "<<i<< " sub size "<< sub_vec.size() <<endl;
//        voted_selected[i].insert(voted_selected[i].end(),sub_vec.begin(),sub_vec.end());

    }
    return voted_selected;
}


