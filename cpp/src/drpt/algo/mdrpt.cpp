#include "mdrpt.hpp"
#include <cblas.h>
#include <stdio.h>
#include "drpt_local.hpp"
#include "../math/math_operations.hpp"
#include <vector>
#include <random>
#include <mpi.h>
#include <string>
#include <iostream>
#include <omp.h>
#include <map>
#include <unordered_map>
#include <fstream>
#include "../algo/drpt_global.hpp"
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <limits.h>
#include <cstring>
#include <set>

using namespace std;
using namespace std::chrono;

drpt::MDRPT::MDRPT(int ntrees, int tree_depth,
		double tree_depth_ratio, int local_tree_offset,
		int total_data_set_size, int local_data_set_size, int dimension,
		int rank, int world_size, string input_path, string output_path)
{
	this->data_dimension = dimension;
	this->tree_depth = tree_depth;
	this->global_data_set_size = total_data_set_size;
	this->local_data_set_size = local_data_set_size;
	this->rank = rank;
	this->world_size = world_size;
	this->ntrees = ntrees;
	this->input_path = input_path;
	this->output_path = output_path;
	this->tree_depth_ratio = tree_depth_ratio;
	this->trees_leaf_all = vector < vector < vector < DataPoint > >>(ntrees);
	this->index_distribution = vector < set < int >> (world_size);
	this->local_tree_offset = local_tree_offset;
}

void drpt::MDRPT::grow_trees(vector<vector<VALUE_TYPE>>& original_data, float density, bool use_locality_optimization,
		int nn, ofstream& fout) {

	drpt::MathOp mathOp; //class uses for math operations
	VALUE_TYPE* row_data_array = mathOp.convert_to_row_major_format(original_data); // this algorithm assumes row major format for operations

	int global_tree_depth = this->tree_depth * this->tree_depth_ratio;

	// generate random seed at process 0 and broadcast it to multiple processes.
	int* receive = this->receive_random_seeds(1);

	// build global sparse random project matrix for all trees
	VALUE_TYPE* B = mathOp.build_sparse_projection_matrix(this->rank, this->world_size, this->data_dimension,
			global_tree_depth * this->ntrees, density, receive[0]);

	// get the matrix projection
	// P= X.R
	VALUE_TYPE* P = mathOp.multiply_mat(row_data_array, B, this->data_dimension,
			global_tree_depth * this->ntrees,
			this->local_data_set_size,
			1.0);

	//calculates starting data index
	int starting_index = (this->global_data_set_size / world_size) * this->rank;
	this->starting_data_index = starting_index;

	// creating DRPTGlobal class
	drpt::DRPTGlobal drpt_global = drpt::DRPTGlobal(P,
			B,
			this->local_data_set_size,
			this->data_dimension,
			global_tree_depth,
			this->ntrees, starting_index,
			this->global_data_set_size,
			this->rank,
			this->world_size);

	cout << " rank " << rank << " starting growing trees" << endl;
	// start growing global tree
	drpt_global.grow_global_tree(original_data);
	cout << " rank " << rank << " completing growing trees" << endl;


	//calculate locality optimization to improve data locality
	if (use_locality_optimization)
	{
		cout << " rank " << rank << " starting tree leaf correlation " << endl;
		drpt_global.calculate_tree_leaf_correlation();
		cout << " rank " << rank << "  tree leaf correlation completed " << endl;
	}

	vector<vector<vector<DataPoint>>> leaf_nodes_of_trees(ntrees);

	cout << " rank " << rank << " running  datapoint collection  "<< endl;
	// running the similar datapoint collection
	for (int i = 0; i < ntrees; i++)
	{
		leaf_nodes_of_trees[i] = drpt_global.collect_similar_data_points(i, use_locality_optimization,
				this->index_distribution,this->datamap);

	}

	// get the global minimum value of a leaf
	int global_minimum = this->get_global_minimum_leaf_size(leaf_nodes_of_trees);

        cout << " rank " << rank << " global_minimum  "<<global_minimum<< endl;
	//grow local trees for each leaf
	this->grow_local_trees(leaf_nodes_of_trees,global_minimum,nn,global_tree_depth, density);


	delete[] receive;
}

void drpt::MDRPT::calculate_nns(map<int, vector<drpt::DataPoint>>& local_nns, set<int>& keys, int tree, int nn)
{

	drpt::MathOp mathOp;

	// only process belonging indices
	for (int i = this->my_leaf_start_index; i < this->my_leaf_end_index; i++)
	{
		if (!this->trees_leaf_all[tree][i].empty())
		{
			vector<DataPoint> data_points = this->trees_leaf_all[tree][i];

			// vector to store source datapoint  and it's nns (kind of linked list data structure)
			vector<vector<DataPoint>> vec(data_points.size());

#pragma omp parallel for
			for (int k = 0; k < data_points.size(); k++)
			{
				vec[k] = vector<DataPoint>(data_points.size());
			}

#pragma omp parallel for
			for (int k = 0; k < data_points.size(); k++)
			{
				for (int j = 0; j < data_points.size(); j++)
				{
					VALUE_TYPE distance = mathOp.calculate_distance(this->datamap[data_points[k].index],
							this->datamap[data_points[j].index]);
					DataPoint dataPoint;
					dataPoint.src_index = data_points[k].index;
					dataPoint.index = data_points[j].index;
					dataPoint.distance = distance;
					vec[k][j] = dataPoint;
			}
			}

			for (int k = 0; k < data_points.size(); k++)
			{
				// sort all nearest neighbours
				sort(vec[k].begin(), vec[k].end(),
						[](const DataPoint& lhs, const DataPoint& rhs)
						{
						  return lhs.distance < rhs.distance;
						});

				vector<DataPoint> sub_vec;
				if (vec.size() > nn)
				{
					// select only closed proximity nearest neighbours
					sub_vec = slice(vec[k], 0, nn - 1);
				}
				else
				{
					sub_vec = vec[k];
				}

				int idx = sub_vec[0].src_index;
				if (local_nns.find(idx) != local_nns.end())
				{
					std::vector<DataPoint> dst;
					auto it = local_nns.find(idx);
					vector<DataPoint> ex_vec = it->second;
					// merge based on distances to the linked list
					std::merge(ex_vec.begin(), ex_vec.end(), sub_vec.begin(),
							sub_vec.end(), std::back_inserter(dst), [](const DataPoint& lhs, const DataPoint& rhs)
							{
							  return lhs.distance < rhs.distance;
							});
					// erase duplicates
					dst.erase(unique(dst.begin(), dst.end(),
							[](const DataPoint& lhs,
									const DataPoint& rhs)
							{
							  return lhs.index == rhs.index;
							}), dst.end());
					(it->second) = dst;

				} else {
						if (local_nns.find(idx) == local_nns.end())
						{
							//final linked list of source indices and their nearest neighbours
							local_nns.insert(pair < int, vector < drpt::DataPoint >> (idx, sub_vec));
							keys.insert(idx);
						}
					}
//				}
			}
		}
	}
}

std::map<int,vector<drpt::DataPoint>> drpt::MDRPT::communicate_nns(map<int, vector<drpt::DataPoint>> &local_nns,
		set<int>& keys,int nn) {

	int* receiving_indices_count = new int[this->world_size]();
	int* disps_receiving_indices = new int[this->world_size]();
	int send_count = 0;
	int total_receving = 0;

	//send distance threshold to original data owner
	index_distance_pair *out_index_dis = this->send_min_max_distance_to_data_owner(local_nns,
			receiving_indices_count,disps_receiving_indices,send_count,total_receving,nn);


	vector<index_distance_pair> final_sent_indices_to_rank_map(this->local_data_set_size);

	//finalize data owners based on data owner having minimum distance threshold.
	this->finalize_final_dataowner(receiving_indices_count,disps_receiving_indices,out_index_dis,final_sent_indices_to_rank_map);

	//announce the selected dataowner to all interesting data holders
	vector<vector<index_distance_pair>> final_indices_allocation =  this->announce_final_dataowner(total_receving,
			receiving_indices_count, disps_receiving_indices,out_index_dis,final_sent_indices_to_rank_map);


	std::map<int, vector<DataPoint>>final_nn_sending_map;
	std::map<int, vector<DataPoint>>final_nn_map;

	int* sending_selected_indices_count = new int[this->world_size]();
	int* sending_selected_indices_nn_count = new int[this->world_size]();

	int* receiving_selected_indices_count = new int[this->world_size]();
	int* receiving_selected_indices_nn_count = new int[this->world_size]();

	//select final nns to be forwared to dataowners
	this->select_final_forwarding_nns(final_indices_allocation,
			local_nns,
			final_nn_sending_map,final_nn_map,
			sending_selected_indices_count,
			sending_selected_indices_nn_count);


	this->send_nns(sending_selected_indices_count,sending_selected_indices_nn_count,
			receiving_selected_indices_count,final_nn_map,final_nn_sending_map,final_indices_allocation);

	delete[]
			receiving_indices_count;
	delete[]
			disps_receiving_indices;

	delete[]
			sending_selected_indices_count;
	delete[]
			sending_selected_indices_nn_count;
	delete[]
			receiving_selected_indices_count;
	delete[]
			receiving_selected_indices_nn_count;
	return final_nn_map;
}

std::map<int, vector<drpt::DataPoint>> drpt::MDRPT::gather_nns(int nn, ofstream& fout) {

	std::map<int, vector<DataPoint>> local_nn_map;

	set<int> keys;

	for (int i = 0; i < ntrees; i++)
	{
		// calculate nearest neighbours
		this->calculate_nns(local_nn_map, keys, i, 2 * nn);
	}

	cout << " rank " << rank << " distance calculation completed " << endl;

	std::map<int, vector<drpt::DataPoint>> final_map = communicate_nns(local_nn_map, keys, nn);


	return final_map;
}

int drpt::MDRPT::get_global_minimum_leaf_size(vector<vector < vector < DataPoint>>>& leaf_nodes_of_trees) {
	int my_minimum_count = INT32_MAX;
	for (int i = 0; i < ntrees; i++)
	{
		for (int j = 0; j < leaf_nodes_of_trees[i].size(); j++)
		{
			if (leaf_nodes_of_trees[i][j].size() > 0 and my_minimum_count > leaf_nodes_of_trees[i][j].size())
				my_minimum_count = leaf_nodes_of_trees[i][j].size();
		}
	}

	int* minimum_array = new int[1];
	minimum_array[0] = my_minimum_count;
	int* minimum_array_receive = new int[this->world_size]();

	MPI_Allgather(minimum_array, 1, MPI_INT, minimum_array_receive, 1, MPI_INT, MPI_COMM_WORLD);

	int global_minimum = INT32_MAX;
	for (int i = 0; i < this->world_size; i++)
	{
		if (global_minimum > minimum_array_receive[i])
		{
			global_minimum = minimum_array_receive[i];
		}
	}
	return global_minimum;
}

void drpt::MDRPT::grow_local_trees(vector<vector<vector<DataPoint>>> &leaf_nodes_of_trees, int global_minimum,
		int nn,int global_tree_depth, int density) {
	drpt::MathOp mathOp;
	int local_tree_depth = log2(global_minimum) - (log2(nn) + this->local_tree_offset);
	this->tree_depth = local_tree_depth + global_tree_depth;

	cout << "rank " << rank << " adjusted local tree height " << local_tree_depth << " adjusted global tree depth "
		 << global_tree_depth <<
		 endl;

	 this->total_leaf_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

	 this->leafs_per_node = total_leaf_size / this->world_size;

	this->my_leaf_start_index  = leafs_per_node * this->rank;
	 this->my_leaf_end_index  = leafs_per_node * (this->rank + 1);
	if (this->rank == this->world_size - 1)
	{
        this->my_leaf_end_index = total_leaf_size;
	}

// random seed generation for all local trees
	int* receive_ntrees = this->receive_random_seeds(this->ntrees);

	for (int i = 0;i < this->ntrees;i++)
	{
		this->trees_leaf_all[i] =
				vector < vector < drpt::DataPoint >> (total_leaf_size);

		VALUE_TYPE* C = mathOp.build_sparse_projection_matrix(this->rank,
				this->world_size,
				this->data_dimension,
				local_tree_depth, density,
				receive_ntrees[i]);

		int data_nodes_count_per_process = 0;

		for (int j = 0;j < leaf_nodes_of_trees[i].size();j++)
		{
			vector<vector<VALUE_TYPE>> local_data(leaf_nodes_of_trees[i][j].size());

			for (int k = 0;k < leaf_nodes_of_trees[i][j].size();k++)
			{
//				local_data[k] = leaf_nodes_of_trees[i][j][k].image_data;
                local_data[k] = this->datamap[leaf_nodes_of_trees[i][j][k].index];
			}
			VALUE_TYPE* local_data_arr = mathOp.convert_to_row_major_format(local_data);

			VALUE_TYPE* LP = mathOp.multiply_mat(local_data_arr, C, this->data_dimension,
					local_tree_depth,
					leaf_nodes_of_trees[i][j].size(), 1.0);

			DRPTLocal drpt_local = drpt::DRPTLocal(LP, C,
					leaf_nodes_of_trees[i][j].size(), local_tree_depth, local_data,
					1, this->starting_data_index,
					this->rank,
					this->world_size);

			drpt_local.grow_local_tree();

			vector<vector<int>> final_clustered_data = drpt_local.get_all_leaf_node_indices(0);

			for (int l = 0;l < final_clustered_data.size();l++)
			{
				vector<DataPoint> data_vec;
				for (int m = 0;m < final_clustered_data[l].size();m++)
				{
					int index = final_clustered_data[l][m];
					data_vec.push_back(leaf_nodes_of_trees[i][j][index]);
				}

				int id = this->my_leaf_start_index + (data_nodes_count_per_process % leafs_per_node);
				this->trees_leaf_all[i][id] =data_vec;
				data_nodes_count_per_process++;
			}

			free(local_data_arr);
			free(LP);
		}
		free(C);
	}
}

int* drpt::MDRPT::receive_random_seeds(int size) {
	int* receive = new int[size]();
	if (this->rank == 0) {
		for (int i = 0; i < size; i++)
		{
			std::random_device rd;
			int seed = rd();
			receive[i] = seed;
		}
		MPI_Bcast(receive, size, MPI_INT, this->rank, MPI_COMM_WORLD);
	} else {
		MPI_Bcast(receive, size, MPI_INT, NULL, MPI_COMM_WORLD);
	}
	return receive;
}

drpt::MDRPT::index_distance_pair* drpt::MDRPT::send_min_max_distance_to_data_owner(map<int, vector<drpt::DataPoint>>& local_nns,
		int* receiving_indices_count,int* disps_receiving_indices,
		int &send_count,int &total_receiving, int nn) {
	int* sending_indices_count = new int[this->world_size]();
	int* disps_sending_indices = new int[this->world_size]();

	for (int i = 0;i < this->world_size;i++)
	{
		sending_indices_count[i] = this->index_distribution[i].size();
		send_count += sending_indices_count[i];
		disps_sending_indices[i] = (i > 0) ? (disps_sending_indices[i - 1] + sending_indices_count[i - 1]) : 0;
	}

	//sending back received data during collect similar data points to original process
	MPI_Alltoall(sending_indices_count,1, MPI_INT, receiving_indices_count, 1, MPI_INT, MPI_COMM_WORLD);

	for (int i = 0;i < this->world_size;i++)
	{
		total_receiving += receiving_indices_count[i];
		disps_receiving_indices[i] = (i > 0) ? (disps_receiving_indices[i - 1] + receiving_indices_count[i - 1]) : 0;
	}

	index_distance_pair *in_index_dis = new index_distance_pair[send_count];
	index_distance_pair *out_index_dis =  new index_distance_pair[total_receiving];
	int co_process = 0;
	for (int i = 0;i < this->world_size;i++)
	{
		set<int> process_se_indexes = this->index_distribution[i];
		for (set<int>::iterator it = process_se_indexes.begin();it != process_se_indexes.end();it++)
		{
			in_index_dis[co_process].index = (*it);
			in_index_dis[co_process].distance = local_nns[(*it)][nn - 1].distance;
			co_process++;
		}
	}

	//distribute minimum maximum distance threshold (for k=nn)
	MPI_Alltoallv(in_index_dis, sending_indices_count, disps_sending_indices, MPI_FLOAT_INT,out_index_dis,
			receiving_indices_count, disps_receiving_indices, MPI_FLOAT_INT, MPI_COMM_WORLD);

	delete [] sending_indices_count;
	delete [] disps_sending_indices;
	return out_index_dis;
}


void drpt::MDRPT::finalize_final_dataowner(int *receiving_indices_count,int *disps_receiving_indices,
		index_distance_pair *out_index_dis,vector<index_distance_pair> &final_sent_indices_to_rank_map) {

	int my_end_index = this->starting_data_index + this->local_data_set_size;

#pragma omp parallel for
	for (int i = this->starting_data_index;i < my_end_index;i++)
	{
		int selected_rank = -1;
		int search_index = i;
		float minium_distance = std::numeric_limits<float>::max();

		for (int j = 0;j < this->world_size;j++)
		{
			int amount = receiving_indices_count[j];
			int offset = disps_receiving_indices[j];

			for (int k = offset;k < (offset + amount); k++)
			{
				if (search_index == out_index_dis[k].index)
				{
					if (minium_distance > out_index_dis[k].distance)
					{
						minium_distance = out_index_dis[k].distance;
						selected_rank = j;
					}
					break;
				}
			}
		}
		index_distance_pair rank_distance;
		rank_distance.index = selected_rank;  //TODO: replace with rank
		rank_distance.distance = minium_distance;
		final_sent_indices_to_rank_map[search_index - this->starting_data_index] = rank_distance;
	}
}

vector<vector<drpt::MDRPT::index_distance_pair>> drpt::MDRPT::announce_final_dataowner(int total_receving, int *receiving_indices_count, int *disps_receiving_indices,
		drpt::MDRPT::index_distance_pair *out_index_dis, vector<index_distance_pair> &final_sent_indices_to_rank_map) {

	int* minimal_selected_rank_sending = new int[total_receving]();
	index_distance_pair minimal_index_distance[total_receving];

#pragma omp parallel for
	for (int i = 0;i < total_receving;i++)
	{
		minimal_index_distance[i].index = out_index_dis[i].index;
		minimal_index_distance[i].
				distance = final_sent_indices_to_rank_map[out_index_dis[i].index - this->starting_data_index].distance;
		minimal_selected_rank_sending[i] = final_sent_indices_to_rank_map[out_index_dis[i].index - this->starting_data_index].
				index; //TODO: replace
	}

	int* receiving_indices_count_back = new int[this->world_size]();
	int* disps_receiving_indices_count_back = new int[this->world_size]();

	// we recalculate how much we are receiving for minimal dst distribution
	MPI_Alltoall(receiving_indices_count,
			1, MPI_INT, receiving_indices_count_back, 1, MPI_INT, MPI_COMM_WORLD);

	int total_receivce_back = 0;
	for (int i = 0;i < this->world_size;i++)
	{
		total_receivce_back += receiving_indices_count_back[i];
		disps_receiving_indices_count_back[i] = (i > 0) ?
				(disps_receiving_indices_count_back[i - 1] + receiving_indices_count_back[i - 1]) : 0;
	}

	int* minimal_selected_rank_reciving = new int[total_receivce_back]();
	index_distance_pair minimal_index_distance_receiv[total_receivce_back];

	MPI_Alltoallv(minimal_index_distance, receiving_indices_count, disps_receiving_indices, MPI_FLOAT_INT,
			minimal_index_distance_receiv,
			receiving_indices_count_back, disps_receiving_indices_count_back, MPI_FLOAT_INT, MPI_COMM_WORLD
	);

	MPI_Alltoallv(minimal_selected_rank_sending, receiving_indices_count, disps_receiving_indices, MPI_INT,
			minimal_selected_rank_reciving,
			receiving_indices_count_back, disps_receiving_indices_count_back, MPI_INT, MPI_COMM_WORLD
	);

	vector<vector<index_distance_pair>> final_indices_allocation(this->world_size);

#pragma omp parallel
	{
		vector<vector<index_distance_pair>> final_indices_allocation_local(this->world_size);

#pragma omp  for nowait
		for (int i = 0;i < total_receivce_back;i++)
		{
			index_distance_pair distance_pair;
			distance_pair.
					index = minimal_index_distance_receiv[i].index;
			distance_pair.
					distance = minimal_index_distance_receiv[i].distance;
			final_indices_allocation_local[minimal_selected_rank_reciving[i]].
					push_back(distance_pair);

		}

#pragma omp critical
		{
			for (int i = 0;i < this->world_size;i++)
			{
				final_indices_allocation[i].insert(final_indices_allocation[i].end(),
						final_indices_allocation_local[i].begin(),
						final_indices_allocation_local[i].end());
			}
		}
	}

	return final_indices_allocation;

}

void drpt::MDRPT::select_final_forwarding_nns(vector<vector<index_distance_pair>> &final_indices_allocation,
		map<int,vector<drpt::DataPoint>>& local_nns,
		map<int, vector<DataPoint>> &final_nn_sending_map,
		map<int, vector<DataPoint>>  &final_nn_map,
		int* sending_selected_indices_count,
		int* sending_selected_indices_nn_count){

	for (int i = 0;i < this->world_size;i++)
	{

#pragma omp parallel for
		for (int j = 0;j < final_indices_allocation[i].size();j++)
		{
			index_distance_pair in_dis = final_indices_allocation[i][j];
			int selected_index = in_dis.index;
			float dst_th = in_dis.distance;
			if (i != this->rank)
			{
				if (local_nns.find(selected_index)!= local_nns.end())
				{
					vector<drpt::DataPoint> target;
					std::copy_if(local_nns[selected_index].begin(),
							local_nns[selected_index].end(),
							std::back_inserter(target),
							[dst_th](
									drpt::DataPoint dataPoint
							)
							{
							  return dataPoint.distance < dst_th;
							});
					if (target.size()> 0)
					{
#pragma omp critical
						{
							if (final_nn_sending_map.find(selected_index)== final_nn_sending_map.end())
							{
								final_nn_sending_map.insert(pair < int, vector < DataPoint >>
																						   (selected_index, target));
								sending_selected_indices_nn_count[i] += target.size();
								sending_selected_indices_count[i] += 1;
							}
						}
					}
				}
			}
			else
			{
#pragma omp critical
				final_nn_map.insert(pair < int, vector < DataPoint >>(selected_index,
						local_nns[selected_index]));
			}
		}
	}

}

void drpt::MDRPT::send_nns(int *sending_selected_indices_count,int *sending_selected_indices_nn_count, int *receiving_selected_indices_count,
		std::map<int, vector<DataPoint>> &final_nn_map,std::map<int, vector<DataPoint>> &final_nn_sending_map,
		vector<vector<index_distance_pair>> &final_indices_allocation) {

	int total_receiving_count = 0;

	int total_receiving_nn_count = 0;

	MPI_Alltoall(sending_selected_indices_count,
			1, MPI_INT, receiving_selected_indices_count, 1, MPI_INT, MPI_COMM_WORLD);

	int* disps_receiving_selected_indices = new int[this->world_size]();
	int* disps_sending_selected_indices = new int[this->world_size]();
	int* disps_sending_selected_nn_indices = new int[this->world_size]();
	int* disps_receiving_selected_nn_indices = new int[this->world_size]();



	MPI_Alltoall(sending_selected_indices_count,
			1, MPI_INT, receiving_selected_indices_count, 1, MPI_INT, MPI_COMM_WORLD);

    int total_selected_indices_count=0;

	int total_selected_indices_nn_count=0;

	for (int i = 0;i < this->world_size;i++)
	{
		disps_receiving_selected_indices[i] = (i > 0) ? (disps_receiving_selected_indices[i - 1] +
				receiving_selected_indices_count[i - 1]) : 0;
		disps_sending_selected_indices[i] = (i > 0) ? (disps_sending_selected_indices[i - 1] +
				sending_selected_indices_count[i - 1]) : 0;
		disps_sending_selected_nn_indices[i] = (i > 0) ? (disps_sending_selected_nn_indices[i - 1] +
				sending_selected_indices_nn_count[i - 1]) : 0;

		total_selected_indices_count += sending_selected_indices_count[i];
		total_selected_indices_nn_count += sending_selected_indices_nn_count[i];
	}

	int* sending_selected_indices = new int[total_selected_indices_count]();

	int* sending_selected_nn_count_for_each_index = new int[total_selected_indices_count]();

	index_distance_pair sending_selected_nn[total_selected_indices_nn_count];

	int inc = 0;
	int selected_nn = 0;
	for (int i = 0;i < this->world_size;i++)
	{
		total_receiving_count += receiving_selected_indices_count[i];
		if (i != this->rank)
		{
			vector<index_distance_pair> final_indices = final_indices_allocation[i];
			for (int j = 0;j < final_indices.size();j++)
			{
				if (final_nn_sending_map.find(final_indices[j].index) != final_nn_sending_map.end())
				{
					vector<drpt::DataPoint> nn_sending = final_nn_sending_map[final_indices[j].index];
					if (nn_sending.size()> 0)
					{
						sending_selected_indices[inc] = final_indices[j].index;
						for (int k = 0;k < nn_sending.size();k++)
						{
							sending_selected_nn[selected_nn].
									index = nn_sending[k].index;
							sending_selected_nn[selected_nn].
									distance = nn_sending[k].distance;
							selected_nn++;
						}
						sending_selected_nn_count_for_each_index[inc] = nn_sending.
								size();
						inc++;
					}
				}
			}
		}
	}

	int* receiving_selected_nn_indices_count = new int[total_receiving_count]();

	int* receiving_selected_indices = new int[total_receiving_count]();

	MPI_Alltoallv(sending_selected_nn_count_for_each_index, sending_selected_indices_count,
			disps_sending_selected_indices, MPI_INT, receiving_selected_nn_indices_count,
			receiving_selected_indices_count, disps_receiving_selected_indices, MPI_INT, MPI_COMM_WORLD
	);

	MPI_Alltoallv(sending_selected_indices, sending_selected_indices_count, disps_sending_selected_indices, MPI_INT,
			receiving_selected_indices,
			receiving_selected_indices_count, disps_receiving_selected_indices, MPI_INT, MPI_COMM_WORLD
	);

	int* receiving_selected_nn_indices_count_process = new int[this->world_size]();

	for (int i = 0;i < this->world_size;i++)
	{
		int co = receiving_selected_indices_count[i];
		int offset = disps_receiving_selected_indices[i];
//        int per_pro_co = 0;
		for (int k = offset;k < (co + offset); k++)
		{
			receiving_selected_nn_indices_count_process[i] += receiving_selected_nn_indices_count[k];
		}
		total_receiving_nn_count += receiving_selected_nn_indices_count_process[i];
//        receiving_selected_nn_indices_count_process[i] =per_pro_co;
		disps_receiving_selected_nn_indices[i] = (i > 0) ? (disps_receiving_selected_nn_indices[i - 1] +
				receiving_selected_nn_indices_count_process[i - 1]) : 0;
	}

	index_distance_pair receving_selected_nn[total_receiving_nn_count];

//    cout << " rank " << rank << " total receiving nn indicies " << total_receiving_nn_count <<endl;

	MPI_Alltoallv(sending_selected_nn, sending_selected_indices_nn_count, disps_sending_selected_nn_indices,
			MPI_FLOAT_INT,
			receving_selected_nn,
			receiving_selected_nn_indices_count_process, disps_receiving_selected_nn_indices, MPI_FLOAT_INT,
			MPI_COMM_WORLD
	);

	int nn_index = 0;
	for (int i = 0;i < total_receiving_count;i++)
	{
		int src_index = receiving_selected_indices[i];
		int nn_count = receiving_selected_nn_indices_count[i];
		vector<DataPoint> vec;
		for (int j = 0;j < nn_count;j++)
		{
			int nn_indi = receving_selected_nn[nn_index].index;
			VALUE_TYPE distance = receving_selected_nn[nn_index].distance;
			DataPoint dataPoint;
			dataPoint.src_index = src_index;
			dataPoint.index = nn_indi;
			dataPoint.distance = distance;
			vec.push_back(dataPoint);
			nn_index++;
		}

		auto its = final_nn_map.find(src_index);
		if (its == final_nn_map.end())
		{
			final_nn_map.insert(pair < int, vector < DataPoint >>(src_index, vec));
		}
		else
		{
			vector<DataPoint> dst;
			vector<DataPoint> ex_vec = its->second;
			sort(vec.begin(), vec.end(),
					[](const DataPoint& lhs,const DataPoint& rhs)
					{
					  return lhs.distance < rhs.distance;
					});
			std::merge(ex_vec.begin(), ex_vec.end(), vec.begin(),
					vec.end(), std::back_inserter(dst),
					[](const DataPoint& lhs,const DataPoint& rhs
					)
					{
					  return lhs.distance < rhs.distance;
					});
			dst.
					erase(unique(dst.begin(), dst.end(), [](const DataPoint& lhs,
							const DataPoint& rhs)
					{
					  return lhs.index == rhs.index;
					}),
					dst.end()
			);
			(its->second) =dst;
		}
	}
}










