#include "drpt_global.hpp"
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
#include <fstream>
#include <map>
#include <unordered_map>
#include "mdrpt.hpp"
#include <chrono>
#include <algorithm>
#include <limits.h>
#include <unistd.h>
#include <cstring>

using namespace std;
using namespace std::chrono;

dmrpt::DRPTGlobal::DRPTGlobal () {

}

dmrpt::DRPTGlobal::DRPTGlobal (VALUE_TYPE *projected_matrix, VALUE_TYPE *projection_matrix,
		                       int no_of_data_points,
                               int dimension,
                               int tree_depth, int ntrees, int starting_index, int global_dataset_size,
                               int rank, int world_size)
{
  this->tree_depth = tree_depth;
  this->local_dataset_size = no_of_data_points;
  this->projected_matrix = projected_matrix;
  this->projection_matrix = projection_matrix;
  this->global_dataset_size = global_dataset_size;
  this->data_dimension = dimension;

  this->ntrees = ntrees;

  this->trees_data = vector < vector < vector < DataPoint>>>(ntrees);
  this->trees_splits = vector < vector < VALUE_TYPE >> (ntrees);
  this->trees_leaf_first_indices = vector < vector < vector < DataPoint > >>(ntrees);
  this->trees_leaf_first_indices_rearrange = vector < vector < vector < DataPoint > >>(ntrees);
  this->trees_leaf_first_indices_all = vector < vector < vector < DataPoint > >>(ntrees);

  this->starting_data_index = starting_index;
  this->rank = rank;
  this->world_size = world_size;

}

template<typename T> vector <T> slice (vector < T > const &v, int m, int n) {
     auto first = v.cbegin () + m;
     auto last = v.cbegin () + n + 1;
     std::vector <T> vec (first, last);
     return vec;
  }


template<class T, class X> void sortByFreq (std::vector <T> &v, std::vector <X> &vec, int world_size)
{
  std::unordered_map <T, size_t> count;

  for (int i=0; i< v.size();i++)
    {
      count[v[i]]++;
    }

  std::sort (v.begin (), v.end (), [&count] (T const &a, T const &b)
  {
    if (a == b)
      {
        return false;
      }
    if (count[a] > count[b])
      {
        return true;
      }
    else if (count[a] < count[b])
      {
        return false;
      }
    return a < b;
  });
  auto last = std::unique (v.begin (), v.end ());
  v.erase (last, v.end ());

  for (int i=0; i< v.size();i++)
    {
      float priority = (float) count[v[i]] / world_size;

      dmrpt::PriorityMap priorityMap;
      priorityMap.priority = priority;
      priorityMap.leaf_index = v[i];
      vec[v[i]] = priorityMap;

    }
}

int select_next_candidate (vector < vector < vector < vector < dmrpt::PriorityMap >> >> &candidate_mapping,
                           vector < vector < int >> &final_tree_leaf_mapping, int current_tree,
int selecting_tree, int selecting_leaf, int previouse_leaf,int total_leaf_size, int rank ) {
vector <dmrpt::PriorityMap> vec = candidate_mapping[current_tree][previouse_leaf][selecting_tree];
    sort(vec.begin(), vec.end(),
[](const dmrpt::PriorityMap &lhs, const dmrpt::PriorityMap &rhs) {
      return lhs.priority > rhs.priority;
   });

   for ( int i = 0; i<vec.size (); i++) {
          dmrpt::PriorityMap can_leaf = vec[i];
          int id = can_leaf.leaf_index;
          bool candidate = true;

// checking already taken
  for ( int j = selecting_leaf - 1; j >= 0; j--) {
     if (final_tree_leaf_mapping[j][selecting_tree] == id) {
       candidate = false;
      break;
     }
  }

   if (!candidate) {
     continue;
    }

 if (rank==0){
  cout<<"candidate leaf "<<id<<" i "<<i<<" vec_size "<< vec.size()<<endl;
 }

 for ( int j = 0; j<total_leaf_size; j++) {
     vector <dmrpt::PriorityMap> neighbour_vec = candidate_mapping[current_tree][j][selecting_tree];
 if (j != previouse_leaf and neighbour_vec[0].priority > can_leaf.priority and neighbour_vec[0] .leaf_index == can_leaf.leaf_index) {
      candidate = false;
     break;
    }
 }

 if (candidate){
     cout<<"rank "<<rank<<"candidate leaf finall selected "<<id<<endl;
  }

  if (candidate) {
	  cout<<"rank"<<rank<< "storing candidate leaf "<<id<<endl;
      final_tree_leaf_mapping[selecting_leaf][selecting_tree] = can_leaf.leaf_index;
	  cout<<"rank"<<rank<<" returniung candidate leaf "<<can_leaf.leaf_index<<endl;
	  return can_leaf.leaf_index;
   }
   cout<<"moiving to next iternation i "<<i<<endl;
  }
 return -1;
}

void dmrpt::DRPTGlobal::grow_global_tree (vector <vector<VALUE_TYPE>> &data_points) {

  if (this->tree_depth <= 0 || this->tree_depth > log2 (this->local_dataset_size))
    {
      throw std::out_of_range (" depth should be in range [1,....,log2(rows)]");
    }

  if (this->ntrees <= 0)
    {
      throw std::out_of_range (" no of trees should be greater than zero");
    }

  this->data_points = data_points;
  int total_split_size = 1 << (this->tree_depth + 1);
  int total_child_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

  cout << " rank " << rank << " start initial tree growing" << endl;

  this->index_to_tree_leaf_mapper = vector < vector < int >> (this->local_dataset_size);

  for (int k = 0; k < this->ntrees; k++)
    {
      this->trees_splits[k] = vector<VALUE_TYPE> (total_split_size);
      this->trees_data[k] = vector < vector < DataPoint >> (this->tree_depth);
      this->trees_leaf_first_indices[k] = vector < vector < DataPoint >> (total_child_size);
      this->trees_leaf_first_indices_all[k] = vector < vector < dmrpt::DataPoint >> (total_child_size);
      this->trees_leaf_first_indices_rearrange[k] = vector < vector < dmrpt::DataPoint >> (total_child_size);

#pragma  omp parallel for
      for (int i = 0; i < this->tree_depth; i++)
        {
          this->trees_data[k][i] = vector<DataPoint> (this->local_dataset_size);
        }
    }

// storing projected data
#pragma  omp parallel for
  for (int j = 0; j < this->local_dataset_size; j++)
    {
      this->index_to_tree_leaf_mapper[j] = vector<int> (this->ntrees);
      for (int k = 0; k < this->ntrees; k++)
        {
          for (int i = 0; i < this->tree_depth; i++)
            {
              int index = this->tree_depth * k + i + j * this->tree_depth * this->ntrees;
              DataPoint dataPoint;
              dataPoint.value = this->projected_matrix[index];
              dataPoint.index = j + this->starting_data_index;
//              dataPoint.image_data = this->data_points[j];
              this->trees_data[k][i][j] = dataPoint;
            }
        }
    }

  cout << " rank " << rank << " completed image data storing for all trees" << endl;


  for (int k = 0; k < this->ntrees; k++)
    {

      vector <vector<DataPoint>> child_data_tracker (total_split_size);
      vector<int> global_size_vector (total_split_size);
      child_data_tracker[0] = this->trees_data[k][0];
      global_size_vector[0] = this->global_dataset_size;


      for (int i = 0; i < this->tree_depth - 1; i++)
        {
          cout << " rank " << rank << " working on tree"<<k<<"  depth" << i << endl;
          this->grow_global_subtree (child_data_tracker, global_size_vector, i, k);

        }
    }
}

void dmrpt::DRPTGlobal::grow_global_subtree (vector <vector<DataPoint>> &child_data_tracker, vector<int> &global_size_vector,
		int depth, int tree) {

  int current_nodes = (1 << (depth));
  int split_starting_index = (1 << (depth)) - 1;
  int next_split = (1 << (depth + 1)) - 1;

  if (depth == 0) {
      split_starting_index = 0;
  }

  //object for math operations
  MathOp mathOp;

  vector<VALUE_TYPE> data(this->local_dataset_size);

  int total_data_count_prev = 0;
  vector<int> local_data_row_count (current_nodes);
  vector<int> global_data_row_count (current_nodes);

  int minimum_vector_size=INT32_MAX;
  for (int i = 0; i < current_nodes; i++) {
      vector <DataPoint> data_vector = child_data_tracker[split_starting_index + i];
      local_data_row_count[i] = data_vector.size ();
	  //keep track of minimum child size
      if(minimum_vector_size>data_vector.size ()){
          minimum_vector_size = data_vector.size ();
      }
      global_data_row_count[i] = global_size_vector[split_starting_index + i];

#pragma omp parallel for
      for (int j = 0; j < data_vector.size (); j++)
        {
          data[j + total_data_count_prev] = data_vector[j].value;
        }
//        cout << " rank " << rank << " level " << depth << " child" << i << " preprocess completed " << endl;
      total_data_count_prev += data_vector.size ();
    }

    // calculation of bins
    int no_of_bins = 1 + (3.322 * log2(minimum_vector_size));

  //calculation of distributed median
  VALUE_TYPE *result = mathOp.distributed_median (data, local_data_row_count, current_nodes,
		  global_data_row_count,
		  28,
                                                  dmrpt::StorageFormat::RAW, this->rank);

  for (int i = 0; i < current_nodes; i++)
    {

      int left_index = (next_split + 2 * i);
      int right_index = left_index + 1;

      int selected_leaf_left = left_index - (1 << (this->tree_depth - 1)) + 1;
      int selected_leaf_right = selected_leaf_left + 1;

      VALUE_TYPE median = result[i];

      if(rank == 0)
        {
          cout << " rank " << rank << " calculated median  " << median << " for i" << i << endl;
        }

	  //store median in tree_splits
      this->trees_splits[tree][split_starting_index + i] = median;

      auto start_loop_compute_index = high_resolution_clock::now ();
      vector <DataPoint> left_childs_global;
      vector <DataPoint> right_childs_global;
      vector <DataPoint> data_vector = child_data_tracker[split_starting_index + i];
#pragma omp parallel
      {
        vector <DataPoint> left_childs;
        vector <DataPoint> right_childs;
#pragma omp for  nowait
        for (int k = 0; k < data_vector.size (); k++)
          {
            int index = data_vector[k].index;

            int selected_index = index - this->starting_data_index;
            DataPoint selected_data = this->trees_data[tree][depth + 1][selected_index];

            if (data_vector[k].value <= median)
              {
                left_childs.push_back (selected_data);
                if (depth == this->tree_depth - 2)
                  {
					//keep track of respective leaf of selected_index
                    this->index_to_tree_leaf_mapper[selected_index][tree] = selected_leaf_left;
                  }
              }
            else
              {
                right_childs.push_back (selected_data);
                if (depth == this->tree_depth - 2)
                  {
                    int se_index = selected_data.index - this->starting_data_index;

					//keep track of respective leaf of selected_index
                    this->index_to_tree_leaf_mapper[selected_index][tree] = selected_leaf_right;
                  }
              }
          }

#pragma omp critical
        {
          left_childs_global.insert (left_childs_global.end (), left_childs.begin (), left_childs.end ());
          right_childs_global.insert (right_childs_global.end (), right_childs.begin (), right_childs.end ());
        }
      }

      child_data_tracker[left_index] = left_childs_global;
      child_data_tracker[right_index] = right_childs_global;
      if (depth == this->tree_depth - 2) {
          this->trees_leaf_first_indices[tree][selected_leaf_left] = left_childs_global;
          this->trees_leaf_first_indices[tree][selected_leaf_right] = right_childs_global;
        }
//        cout << " rank " << rank << " node completed  depth " << depth << " for i" << i << endl;
    }

  if (depth == this->tree_depth - 2) {
      return;
    }

  this->derive_global_datavector_sizes(child_data_tracker,global_size_vector,current_nodes,next_split);

  free (result);

}

vector <vector<dmrpt::DataPoint>> dmrpt::DRPTGlobal::collect_similar_data_points (int tree,
		bool use_data_locality_optimization, vector <set<int>> &index_distribution, std::map<int, vector<VALUE_TYPE>> &datamap)
{

  dmrpt::MathOp mathOp;

  int total_leaf_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

  int leafs_per_node = total_leaf_size / this->world_size;
  int my_start_count = 0;
  int end_count = 0;
  int sending_rank = -1;

  int *send_counts = new int[total_leaf_size]();
  int *recv_counts = new int[total_leaf_size]();


  int sum_per_node = 0;
  int process = 0;
  int *send_indices_count = new int[this->world_size]();
  int *disps_indices_count = new int[this->world_size]();
  int *send_values_count = new int[this->world_size]();
  int *disps_values_count = new int[this->world_size]();

  int my_total = 0;
  for (int i = 0; i < total_leaf_size; i++)
    {
      if (i > 0 && i % leafs_per_node == 0)
        {
          send_indices_count[process] = sum_per_node;
          send_values_count[process] = sum_per_node * this->data_dimension;
          sum_per_node = 0;
          process++;
        }
      vector <DataPoint> all_points = (use_data_locality_optimization)
                                      ? this->trees_leaf_first_indices_rearrange[tree][i]
                                      : this->trees_leaf_first_indices[tree][i];
      send_counts[i] = all_points.size ();
      sum_per_node += send_counts[i];
      my_total += send_counts[i];
    }

  send_indices_count[process] = sum_per_node;
  send_values_count[process] = sum_per_node * this->data_dimension;

  MPI_Alltoall (send_counts, leafs_per_node, MPI_INT, recv_counts, leafs_per_node,
                MPI_INT, MPI_COMM_WORLD);

  int *total_leaf_count = new int[leafs_per_node]();
  int *recev_indices_count = new int[this->world_size]();
  int *recev_values_count = new int[this->world_size]();
  int *recev_disps_count = new int[this->world_size]();
  int *recev_disps_values_count = new int[this->world_size]();

  int total_sum = 0;
  for (int j = 0; j < leafs_per_node; j++)
    {
      int count = 0;
      for (int i = 0; i < this->world_size; i++)
        {
          count += recv_counts[j + i * leafs_per_node];
        }
      total_leaf_count[j] = count;
      total_sum += count;
    }

  for (int i = 0; i < this->world_size; i++)
    {
      int count = 0;
      for (int j = 0; j < leafs_per_node; j++)
        {
          count += recv_counts[j + i * leafs_per_node];
        }
      recev_indices_count[i] = count;
      recev_values_count[i] = count * this->data_dimension;

    }

  for (int i = 0; i < this->world_size; i++)
    {
      disps_indices_count[i] = (i > 0) ? (disps_indices_count[i - 1] + send_indices_count[i - 1]) : 0;
      recev_disps_count[i] = (i > 0) ? (recev_disps_count[i - 1] + recev_indices_count[i - 1]) : 0;
      disps_values_count[i] = (i > 0) ? (disps_values_count[i - 1] + send_values_count[i - 1]) : 0;
      recev_disps_values_count[i] = (i > 0) ? (recev_disps_values_count[i - 1] + recev_values_count[i - 1]) : 0;
    }

  int *receive_indices = new int[total_sum]();
  VALUE_TYPE *receive_values = new VALUE_TYPE[total_sum * this->data_dimension]();
  int *send_indices = new int[my_total]();
  VALUE_TYPE *send_values = new VALUE_TYPE[my_total * this->data_dimension]();

  int co = 0;
  int current_process = 0;
  for (int i = 0; i < total_leaf_size; i++)
    {

      vector <DataPoint> all_points = (use_data_locality_optimization)
                                      ? this->trees_leaf_first_indices_rearrange[tree][i]
                                      : this->trees_leaf_first_indices[tree][i];


      if (i > 0 && i % leafs_per_node == 0)
        {
          current_process++;
        }

      for (int j = 0; j < all_points.size (); j++)
        {
          send_indices[co] = all_points[j].index;

#pragma omp parallel for
          for (int k = 0; k < this->data_dimension; k++)
            {
//              send_values[co * this->data_dimension + k] = all_points[j].image_data[k];
               int local_index  = all_points[j].index - this->starting_data_index;
			  send_values[co * this->data_dimension + k] = this->data_points[local_index][k];
            }
          co++;
        }
    }

  MPI_Alltoallv (send_indices, send_indices_count, disps_indices_count, MPI_INT, receive_indices,
                 recev_indices_count, recev_disps_count, MPI_INT, MPI_COMM_WORLD);

  MPI_Alltoallv (send_values, send_values_count, disps_values_count, MPI_VALUE_TYPE, receive_values,
                 recev_values_count, recev_disps_values_count, MPI_VALUE_TYPE, MPI_COMM_WORLD);

  my_start_count = leafs_per_node * this->rank;
  if (this->rank < this->world_size - 1)
    {
      end_count = leafs_per_node * (this->rank + 1);
    }
  else
    {
      end_count = total_leaf_size;
    }

  vector<int> process_read_offsets (this->world_size);
  vector<int> process_read_offsets_value (this->world_size);
  vector <vector<DataPoint>> all_leaf_nodes (leafs_per_node);

  for (int i = 0; i < leafs_per_node; i++)
    {
      vector <DataPoint> datavec (total_leaf_count[i]);
      int testcr = 0;
      for (int j = 0; j < this->world_size; j++)
        {
          int count_per_leaf_per_node = recv_counts[i + j * leafs_per_node];
          int read_offset = recev_disps_count[j];
          int read_offset_data = recev_disps_values_count[j];

          if (i == 0)
            {
              process_read_offsets[j] = read_offset + count_per_leaf_per_node;
              process_read_offsets_value[j] = read_offset_data + count_per_leaf_per_node * this->data_dimension;
            }
          else
            {
              read_offset = process_read_offsets[j];
              process_read_offsets[j] = read_offset + count_per_leaf_per_node;
              read_offset_data = process_read_offsets_value[j];
              process_read_offsets_value[j] = read_offset_data + count_per_leaf_per_node * this->data_dimension;
            }

          int value_read_count = read_offset_data;

          for (int k = read_offset; k < process_read_offsets[j]; k++)
            {
              DataPoint dataPoint;
              dataPoint.index = receive_indices[k];

//              if (j != this->rank) {
                index_distribution[j].insert(dataPoint.index);

//              }

              vector<VALUE_TYPE> image_values = vector<VALUE_TYPE> (this->data_dimension);

#pragma omp parallel for
              for (int m = value_read_count; m < (value_read_count + this->data_dimension); m++)
                {
                  int r = m - value_read_count;
                  image_values[r] = receive_values[m];
                }

		      datamap.insert(pair < int, vector <VALUE_TYPE>> (dataPoint.index, image_values));

              datavec[testcr] = dataPoint;
              value_read_count += this->data_dimension;
              testcr++;
            }
        }

      int id = i + my_start_count;
      this->trees_leaf_first_indices_all[tree][id] = datavec;
      all_leaf_nodes[i] = datavec;
    }

  delete[] send_counts;
  delete[] recv_counts;
  delete[] send_indices_count;
  delete[] disps_indices_count;
  delete[] send_values_count;
  delete[] disps_values_count;
  delete[] recev_indices_count;
  delete[] recev_values_count;
  delete[] recev_disps_count;
  delete[] recev_disps_values_count;

  return all_leaf_nodes;

}

void dmrpt::DRPTGlobal::calculate_tree_leaf_correlation ()
{

  vector < vector < vector < vector < dmrpt::PriorityMap >> >> candidate_mapping =
      vector < vector < vector < vector < dmrpt::PriorityMap >> >> (this->ntrees);

  int total_leaf_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

  vector <vector<int>> final_tree_leaf_mapping (total_leaf_size);

  int total_sending = this->ntrees * total_leaf_size * this->ntrees;
  int *my_sending_leafs = new int[total_sending] ();

  int total_receiving = this->ntrees * total_leaf_size * this->ntrees * this->world_size;

  int *total_receiving_leafs = new int[total_receiving] ();

  int *send_count = new int[this->world_size] ();
  int *disps_send = new int[this->world_size] ();
  int *recieve_count = new int[this->world_size] ();
  int *disps_recieve = new int[this->world_size] ();

  for (int i = 0; i < this->world_size; i++)
    {
      send_count[i] = total_sending;
      disps_send[i] = 0;
      recieve_count[i] = total_sending;
      disps_recieve[i] = (i > 0) ? (disps_recieve[i - 1] + recieve_count[i - 1]) : 0;
    }

  vector < vector < vector < vector < float >> >> correlation_matrix =
      vector < vector < vector < vector < float >> >> (ntrees);

  for (int tree = 0; tree < this->ntrees; tree++)
    {
      correlation_matrix[tree] = vector < vector < vector < float>>>(total_leaf_size);
      candidate_mapping[tree] = vector < vector < vector < dmrpt::PriorityMap>>>(total_leaf_size);
      for (int leaf = 0; leaf < total_leaf_size; leaf++)
        {
          correlation_matrix[tree][leaf] = vector < vector < float >> (this->ntrees);
          candidate_mapping[tree][leaf] = vector < vector < dmrpt::PriorityMap >> (this->ntrees);
          vector <DataPoint> data_points = this->trees_leaf_first_indices[tree][leaf];
          for (int j = 0; j < this->ntrees; j++)
            {
              correlation_matrix[tree][leaf][j] = vector<float> (total_leaf_size, 0);
            }
//#pragma omp parallel for
          for (int c = 0; c < data_points.size (); c++)
            {
              int selec_index = data_points[c].index - this->starting_data_index;
              vector<int> vec = this->index_to_tree_leaf_mapper[selec_index];
              for (int j = 0; j < vec.size (); j++)
                {
//#pragma omp atomic
                  correlation_matrix[tree][leaf][j][vec[j]] += 1;
                }
            }
        }
    }

  cout<<"rank "<<rank<<"corelation matrix population completed "<<endl;

  for (int tree = 0; tree < this->ntrees; tree++)
    {
//#pragma omp parallel for
      for (int leaf = 0; leaf < total_leaf_size; leaf++)
        {
          for (int c = 0; c < this->ntrees; c++)
            {
              vector <DataPoint> data_points = this->trees_leaf_first_indices[tree][leaf];
              int size = data_points.size ();
              std::transform (correlation_matrix[tree][leaf][c].begin (), correlation_matrix[tree][leaf][c].end (),
                              correlation_matrix[tree][leaf][c].begin (), [&] (float x)
                              { return (x / size) * 100; });
              int selected_leaf = std::max_element (correlation_matrix[tree][leaf][c].begin (),
                                                    correlation_matrix[tree][leaf][c].end ()) -
                                  correlation_matrix[tree][leaf][c].begin ();
              int count = c + leaf * this->ntrees + tree * total_leaf_size * this->ntrees;
              my_sending_leafs[count] = selected_leaf;
            }
        }
    }

  cout<<"rank "<<rank<<"local tree leaf correlation completed "<<endl;

  MPI_Allgatherv (my_sending_leafs, total_sending, MPI_INT, total_receiving_leafs,
                  recieve_count, disps_recieve, MPI_INT, MPI_COMM_WORLD);

  for (int j = 0; j < this->ntrees; j++)
    {
      for (int k = 0; k < total_leaf_size; k++)
        {
          final_tree_leaf_mapping[k] = vector<int> (this->ntrees, -1);
          for (int m = 0; m < this->ntrees; m++)
            {
              candidate_mapping[j][k][m] = vector<PriorityMap> (total_leaf_size);
            }
        }
    }

	cout<<"rank "<<rank<<"final tree leaf mapping filling completed "<<endl;

//#pragma omp parallel for
  for (int l = 0; l < this->ntrees * total_leaf_size * this->ntrees; l++)
    {

      int m = l % this->ntrees;
      int temp_val = (l - m) / this->ntrees;
      int j = temp_val / total_leaf_size;
      int k = temp_val % total_leaf_size;

       for (int n = 0; n < total_leaf_size; n++) {
                    PriorityMap priorityMap;
                    priorityMap.priority = 0;
                    priorityMap.leaf_index = n;
                    candidate_mapping[j][k][m][n]=priorityMap;
                }

      vector<int> vec;
      for (int p = 0; p < this->world_size; p++)
        {
          int id = p * total_sending + j * total_leaf_size * this->ntrees + k * this->ntrees + m;
          int value = total_receiving_leafs[id];
          vec.push_back (value);
        }
      sortByFreq (vec, candidate_mapping[j][k][m], this->world_size);
    }

	cout<<"rank "<<rank<<"global sorting completed "<<endl;

//#pragma  omp parallel for
  for (int k = 0; k < total_leaf_size; k++)
    {
      int prev_leaf = k;
      for (int m = 0; m < this->ntrees; m++)
        {
          int current_tree = m == 0 ? 0 : m - 1;
          prev_leaf = select_next_candidate (candidate_mapping, final_tree_leaf_mapping, current_tree, m, k, prev_leaf,
                                             total_leaf_size, this->rank);
		  if (prev_leaf = -1){
			  cout<<"rank "<<rank<< " prev_leaf negative"<<" k "<<k<<" m "<<m <<endl;
		  }
        }
    }

	cout<<"rank "<<rank<<"final selection completed "<<endl;

  for (int i = 0; i < this->ntrees; i++)
    {
//#pragma  omp parallel for
      for (int k = 0; k < total_leaf_size; k++)
        {
          int leaf_index = final_tree_leaf_mapping[k][i];

		  //clustered data is stored in the rearranged tree leaves.
          this->trees_leaf_first_indices_rearrange[i][k] = this->trees_leaf_first_indices[i][leaf_index];
        }
    }

	cout<<"rank "<<rank<<"final storing completed "<<endl;
  delete[]
      my_sending_leafs;
  delete[]
      total_receiving_leafs;
  delete[]
      send_count;
  delete[]
      disps_send;
  delete[]
      recieve_count;
  delete[]
      disps_recieve;
}

void dmrpt::DRPTGlobal::derive_global_datavector_sizes(vector<vector<DataPoint>>& child_data_tracker,
		vector<int> &global_size_vector,
		int current_nodes, int next_split) {

// Displacements in the receive buffer for MPI_GATHERV
	int *disps = new int[this->world_size];

// Displacement for the first chunk of data - 0
	for (int i = 0; i < this->world_size; i++)
	{
		disps[i] = (i > 0) ? (disps[i - 1] + 2 * current_nodes) : 0;
	}

	int *total_counts = new int[2 * this->world_size * current_nodes]();

	int *process_counts = new int[this->world_size]();

	for (int k = 0; k < this->world_size; k++)
	{
		process_counts[k] = 2 * current_nodes;
	}
	for (int j = 0; j < current_nodes; j++)
	{
		int id = (next_split + 2 * j);
		total_counts[2 * j + this->rank * current_nodes * 2] = child_data_tracker[id].size ();
		total_counts[2 * j + 1 + this->rank * current_nodes * 2] = child_data_tracker[id + 1].size ();
	}

	MPI_Allgatherv (MPI_IN_PLACE, 0, MPI_INT, total_counts, process_counts, disps, MPI_INT, MPI_COMM_WORLD);

	for (int j = 0; j < current_nodes; j++)
	{
		int left_totol = 0;
		int right_total = 0;
		int id = (next_split + 2 * j);
		for (int k = 0; k < this->world_size; k++)
		{

			left_totol = left_totol + total_counts[2 * j + k * current_nodes * 2];
			right_total = right_total + total_counts[2 * j + 1 + k * current_nodes * 2];
		}

		global_size_vector[id] = left_totol;
		global_size_vector[id + 1] = right_total;
	}

	free (process_counts);
	free (total_counts);
	free (disps);
}






