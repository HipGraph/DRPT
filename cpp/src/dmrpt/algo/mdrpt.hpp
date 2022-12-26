#ifndef DISTRIBUTED_MRPT_MDRPT_H
#define DISTRIBUTED_MRPT_MDRPT_H

#include "../math/math_operations.hpp"
#include "../utils/drpt_timer.h"
#include "drpt_local.hpp"
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
		std::map<int, vector<VALUE_TYPE>> datamap;

	 private:

		struct index_distance_pair
		{
			float distance;
			int index;
		};

		std::map<int, vector<dmrpt::DataPoint>>
		communicate_nns(std::map<int, vector<dmrpt::DataPoint>>& local_nns,
				set<int>& keys, int nn);

		void calculate_nns(std::map<int, vector<dmrpt::DataPoint>>& local_nns,
				set<int>& keys, int tree, int nn);

		int* receive_random_seeds(int seed);

		int get_global_minimum_leaf_size(vector<vector<vector < DataPoint>>> &leaf_nodes_of_trees);

		void grow_local_trees(vector<vector< vector < DataPoint>>> &leaf_nodes_of_trees,
				int global_minimum,int nn,int global_tree_depth, int density);

		dmrpt::MDRPT::index_distance_pair* send_min_max_distance_to_data_owner(map<int, vector<dmrpt::DataPoint>>& local_nns,
				int* receiving_indices_count,int* disps_receiving_indices,
				int &send_count,int &total_receving, int nn);

		void finalize_final_dataowner(int *receiving_indices_count,int *disps_receiving_indices,
				index_distance_pair *out_index_dis,vector<index_distance_pair> &final_sent_indices_to_rank_map);

		vector<vector<index_distance_pair>> announce_final_dataowner(int total_receving, int *receiving_indices_count, int *disps_receiving_indices,
				dmrpt::MDRPT::index_distance_pair *out_index_dis, vector<index_distance_pair> &final_sent_indices_to_rank_map);

		void select_final_forwarding_nns(vector<vector<index_distance_pair>> &final_indices_allocation,
				map<int,vector<dmrpt::DataPoint>>& local_nns,
				map<int, vector<DataPoint>> &final_nn_sending_map,
				map<int, vector<DataPoint>>  &final_nn_map,
				int* sending_selected_indices_count,
				int* sending_selected_indices_nn_count);

		void send_nns(int *sending_selected_indices_count,int *sending_selected_indices_nn_count, int *receiving_selected_indices_count,
				std::map<int, vector<DataPoint>> &final_nn_map,std::map<int, vector<DataPoint>> &final_nn_sending_map,
				vector<vector<index_distance_pair>> &final_indices_allocation);


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
