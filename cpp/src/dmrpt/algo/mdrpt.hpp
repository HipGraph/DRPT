#ifndef DISTRIBUTED_MRPT_MDRPT_H
#define DISTRIBUTED_MRPT_MDRPT_H

#include "../math/matrix_multiply.hpp"
#include "../utils/drpt_timer.h"
#include "drpt.hpp"
#include "drpt_global.hpp"
#include <cblas.h>
#include <map>
#include <mpi.h>
#include <omp.h>
#include <set>
#include <string>
#include <vector>

namespace dmrpt
{
	class MDRPT
	{

	 private:
		int ntrees;
		int tree_depth;
		double tree_depth_ratio;
		int data_dimension;
		int starting_data_index;
		int rank;
		int world_size;
		int global_data_set_size;
		int local_data_set_size;
		string input_path;
		string output_path;
		vector<vector<vector < DataPoint>>>
		trees_leaf_all;
		vector<set<int>> index_distribution;
		int local_tree_offset;
		int total_leaf_size;
		int leafs_per_node;
		int my_leaf_start_index;
		int my_leaf_end_index;

	 private:
		std::map<int, vector<dmrpt::DataPoint>>
		communicate_nns(std::map<int, vector<dmrpt::DataPoint>>& local_nns,
				set<int>& keys, int nn);

		void calculate_nns(std::map<int, vector<dmrpt::DataPoint>>& local_nns,
				set<int>& keys, int tree, int nn);

		int* receive_random_seeds(int seed);

		int get_global_minimum_leaf_size(vector<vector<vector < DataPoint>>> &leaf_nodes_of_trees);

		void grow_local_trees(vector<vector< vector < DataPoint>>> &leaf_nodes_of_trees,
				int global_minimum,int nn,int global_tree_depth, int density);

		template<typename T>
		vector<T> slice(vector < T >const &v, int m, int n)
		{
			auto first = v.cbegin() + m;
			auto last = v.cbegin() + n + 1;
			std::vector<T> vec(first, last);
			return vec;
		}

		template<typename T>
		bool all_equal(std::vector<T> const& v)
		{
			return std::adjacent_find(v.begin(), v.end(), std::not_equal_to<T>()) == v.end();
		}

	 public:
		MDRPT(int ntrees, int tree_depth, double tree_depth_ratio,
				int local_tree_offset, int total_data_set_size, int local_data_set_size,
				int dimension, int rank, int world_size, string input_path,
				string output_path);

		void grow_trees(vector<vector<VALUE_TYPE>>& original_data, float density,
				bool use_locality_optimization, int nn, ofstream& fout);

		std::map<int, vector<DataPoint>> gather_nns(int nn, ofstream& fout);


	};
} // namespace dmrpt

#endif // DISTRIBUTED_MRPT_MDRPT_H
