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
#include "../algo/drpt_global.hpp"
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <limits.h>
#include <cstring>
#include <set>

using namespace std;
using namespace std::chrono;

dmrpt::MDRPT::MDRPT (int ntrees,  int tree_depth,
                     double tree_depth_ratio,int local_tree_offset,
                     int total_data_set_size, int dimension,
                     int rank, int world_size, string input_path, string output_path) {
  this->data_dimension = dimension;
  this->tree_depth = tree_depth;
  this->total_data_set_size = total_data_set_size;
  this->rank = rank;
  this->world_size = world_size;
  this->ntrees = ntrees;
  this->input_path = input_path;
  this->output_path = output_path;
  this->tree_depth_ratio = tree_depth_ratio;
  this->trees_leaf_all = vector < vector < vector < DataPoint > >>(ntrees);
  this->index_distribution = vector<set<int>> (world_size);
  this->local_tree_offset =local_tree_offset;
}

template<typename T> vector <T> slice (vector < T > const &v, int m, int n) {
   auto first = v.cbegin () + m;
   auto last = v.cbegin () + n + 1;

    std::vector <T> vec (first, last);
    return vec;
   }

template<typename T> bool allEqual (std::vector < T > const &v) {
   return std::adjacent_find(v.begin (), v.end (), std::not_equal_to<T> ()) == v.end ();
  }

void dmrpt::MDRPT::grow_trees (vector <vector<VALUE_TYPE>> &original_data, float density,
                               bool use_locality_optimization, int nn, ofstream &fout)
{
  //original data comes as a matrix, N*D dimensions
  int rows = this->original_data[0].size (); // Calculating D
  int cols = this->original_data.size (); // Calculating N

  auto start_conversion_index = high_resolution_clock::now ();
  dmrpt::MathOp mathOp;
  VALUE_TYPE *imdataArr = mathOp.convert_to_row_major_format (this->original_data);

  int global_tree_depth = this->tree_depth * this->tree_depth_ratio;
  int local_tree_depth = this->tree_depth - global_tree_depth;
  auto stop_conversion_index = high_resolution_clock::now ();

  auto conversion_time = duration_cast<microseconds> (stop_conversion_index - start_conversion_index);

  // generate random seed at process 0 and broadcast it to multiple processes.
  auto start_matrix_index = high_resolution_clock::now ();
  int seed = 0;
  int *receive = new int[1] ();
  if (this->rank == 0)
    {
      std::random_device rd;
      seed = rd ();
      receive[0] = seed;
      MPI_Bcast (receive, 1, MPI_INT, this->rank, MPI_COMM_WORLD);
    }
  else
    {
      MPI_Bcast (receive, 1, MPI_INT, NULL, MPI_COMM_WORLD);
    }

  // build global sparse random project matrix for all trees
  VALUE_TYPE *B = mathOp.build_sparse_projection_matrix (this->rank, this->world_size, this->data_dimension,
                                                         global_tree_depth * this->ntrees, density, receive[0]);

  // get the matrix projection
  // P= X.R
  VALUE_TYPE *P = mathOp.multiply_mat (imdataArr, B, this->data_dimension, global_tree_depth * this->ntrees, cols,
                                       1.0);
  auto stop_matrix_index = high_resolution_clock::now ();
  auto matrix_time = duration_cast<microseconds> (stop_matrix_index - start_matrix_index);




  auto start_grow_index = high_resolution_clock::now ();

  int starting_index = (this->total_data_set_size / world_size) * this->rank;
  this->starting_data_index = starting_index;

  // creating DRPTGlobal class
  dmrpt::DRPTGlobal drpt_global = dmrpt::DRPTGlobal (P, B, cols, this->data_dimension, global_tree_depth, this->ntrees,
                                         starting_index,
                                         this->total_data_set_size, this->rank, this->world_size, this->output_path);

  cout << " rank " << rank << " starting growing trees" << endl;

  // start growing global tree
  drpt_global.grow_global_tree (this->original_data);
  auto stop_grow_index = high_resolution_clock::now ();
  auto index_time = duration_cast<microseconds> (stop_grow_index - start_grow_index);
  cout << " rank " << rank << " completing growing trees" << endl;

  cout << " rank " << rank << " start tree leaf correlation " << endl;
  auto start_calculate_tree_leaf_corr = high_resolution_clock::now ();

  //calculate locality optimization to improve data locality
  if (use_locality_optimization)
    {
      drpt_global.calculate_tree_leaf_correlation (this->output_path);
    }
  auto stop_calculate_tree_leaf_corr = high_resolution_clock::now ();
  auto tree_leaf_corr_time = duration_cast<microseconds> (
      stop_calculate_tree_leaf_corr - start_calculate_tree_leaf_corr);



  cout << " rank " << rank << " running  datapoint collection " << endl;
  auto start_collect = high_resolution_clock::now ();
  vector<vector<vector<DataPoint>>> leaf_nodes_of_trees(ntrees);

  // running the similar datapoint collection
  for (int i = 0; i < ntrees; i++)
    {
      leaf_nodes_of_trees[i] = drpt_global.collect_similar_data_points (i, use_locality_optimization,this->index_distribution);
    }

  cout << " rank " << rank << " similar datapoint collection completed" << endl;

  auto stop_collect = high_resolution_clock::now ();
  auto collect_time = duration_cast<microseconds> (stop_collect - start_collect);



  int my_minimum_count = INT32_MAX;
  for (int i = 0; i < ntrees; i++)
    {
      for (int j = 0; j < leaf_nodes_of_trees[i].size (); j++)
        {
          cout<<"rank after" << rank<<"  tree "<<i<<" leaf "<<j<<" size "<<leaf_nodes_of_trees[i][j].size ()<<endl;
          if (my_minimum_count > leaf_nodes_of_trees[i][j].size () and leaf_nodes_of_trees[i][j].size () > 0)
            my_minimum_count = leaf_nodes_of_trees[i][j].size ();
        }
    }

  int *minimum_arry = new int[1];
  minimum_arry[0] = my_minimum_count;
  int *minimum_arry_recev = new int[this->world_size] ();

  MPI_Allgather(minimum_arry, 1, MPI_INT,minimum_arry_recev, 1, MPI_INT, MPI_COMM_WORLD);

  int global_minimum = INT32_MAX;

  for (int i = 0; i < this->world_size; i++)
    {
      if (global_minimum > minimum_arry_recev[i])
        {
          global_minimum = minimum_arry_recev[i];
        }
    }

  local_tree_depth = log2 (global_minimum) - (log2 (nn) + local_tree_offset);
  this->tree_depth = local_tree_depth + global_tree_depth;

  cout << "rank " << rank << " adjusted local tree height " << local_tree_depth << " adjusted global tree depth "
       << global_tree_depth << endl;

  int total_leaf_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

  int leafs_per_node = total_leaf_size / this->world_size;

  cout << " leafs per node " << leafs_per_node << endl;

  int my_start_count = 0;
  int my_end_count = 0;

  //large trees
  if (total_leaf_size >= this->world_size)
    {
      my_start_count = leafs_per_node * this->rank;
      if (this->rank < this->world_size - 1)
        {
          my_end_count = leafs_per_node * (this->rank + 1);
        }
      else
        {
          my_end_count = total_leaf_size;
        }
    }

  cout << " start count " << my_start_count << " end count " << my_end_count << endl;

  auto start_collect_local = high_resolution_clock::now ();

  int *receive_ntrees = new int[this->ntrees] ();

  // random seed generation for all local trees
  if (this->rank == 0)
    {
      for (int i = 0; i < this->ntrees; i++)
        {
          std::random_device rd;
          int seed = rd ();
          receive_ntrees[i] = seed;
        }
      MPI_Bcast (receive_ntrees, this->ntrees, MPI_INT, this->rank, MPI_COMM_WORLD);
    }
  else
    {
      MPI_Bcast (receive_ntrees, this->ntrees, MPI_INT, NULL, MPI_COMM_WORLD);
    }

  for (int i = 0; i < ntrees; i++)
    {
      vector <vector<DataPoint>> leafs = leaf_nodes_of_trees[i];
      this->trees_leaf_all[i] = vector<vector<dmrpt::DataPoint>>(total_leaf_size);

      VALUE_TYPE *C = mathOp.build_sparse_projection_matrix (this->rank, this->world_size, this->data_dimension,
                                                             local_tree_depth, density, receive_ntrees[i]);

      cout << " tree " << i << " projection matrix completed and leafs size " << leafs.size () << endl;

      int data_nodes_count_per_process = 0;

      for (int j = 0; j < leafs.size (); j++)
        {
//          cout << " creating leaf " << j << endl;
          vector <vector<VALUE_TYPE>> local_data (leafs[j].size ());
          for (int k = 0; k < leafs[j].size (); k++)
            {
              local_data[k] = leafs[j][k].image_data;
            }
//          cout << " data filling complete for  leaf " << j << " size " << local_data.size () << endl;
          VALUE_TYPE *local_data_arr = mathOp.convert_to_row_major_format (local_data);
//            cout<< " row major version completed " <<j<<endl;

          VALUE_TYPE *LP = mathOp.multiply_mat (local_data_arr, C, this->data_dimension,
                                                local_tree_depth,
                                                leafs[j].size (), 1.0);
//            cout<<" creating drpt "<< j <<leafs.size()<<endl;
          DRPT drpt1 = dmrpt::DRPT (LP, C, leafs[j].size (),
                                    local_tree_depth, local_data, 1, starting_index, this->rank, this->world_size);

          drpt1.grow_local_tree ();
//          cout << "rank " << rank << " creating drpt " << j << " tree growing completed" << endl;

          vector <vector<int>> final_clustered_data = drpt1.get_all_leaf_node_indices (0);
//            cout << " final_clustered_data size for leaf " << j << final_clustered_data.size() << endl;

          for (int l = 0; l < final_clustered_data.size (); l++)
            {
              vector <DataPoint> data_vec;
              for (int m = 0; m < final_clustered_data[l].size (); m++)
                {
                  int index = final_clustered_data[l][m];
                  int real_index = leafs[j][index].index;
                  data_vec.push_back (leafs[j][index]);
                }

              int id = my_start_count + (data_nodes_count_per_process % leafs_per_node);
              this->trees_leaf_all[i][id] = data_vec;

              data_nodes_count_per_process++;

            }

          free (local_data_arr);
          free (LP);

        }
      free (C);

    }

  auto end_collect_local = high_resolution_clock::now ();
  auto collect_time_local = duration_cast<microseconds> (end_collect_local - start_collect_local);

  double *execution_times = new double[6] ();

  double *execution_times_global = new double[6] ();
  execution_times[0] = matrix_time.count () / 1000;
  execution_times[1] = index_time.count () / 1000;
  execution_times[2] = tree_leaf_corr_time.count () / 1000;
  execution_times[3] = collect_time.count () / 1000;
  execution_times[4] = collect_time_local.count () / 1000;
  execution_times[4] = collect_time_local.count () / 1000;
  execution_times[5] = conversion_time.count () / 1000;

  MPI_Allreduce (execution_times, execution_times_global, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  fout << rank << " conversion time " << execution_times_global[5] / this->world_size << " matrix  "
       << (execution_times_global[0] / this->world_size) << " global tree construction "
       << (execution_times_global[1] / this->world_size) << " data correlation "
       << (execution_times_global[2] / this->world_size) << " data gathering  "
       << (execution_times_global[3] / this->world_size) << " local tree growing "
       << (execution_times_global[4] / this->world_size) << endl;

  delete[] execution_times;
  delete[] execution_times_global;
//    delete[] receive;
}

void dmrpt::MDRPT::calculate_nns (map<int, vector<dmrpt::DataPoint>> &local_nns, set<int> &keys, int tree, int nn)
{

  dmrpt::MathOp mathOp;

  int total_leaf_size = (1 << (this->tree_depth)) - (1 << (this->tree_depth - 1));

  int leafs_per_node = total_leaf_size / this->world_size;

  int my_start_count = 0;
  int end_count = 0;

  //large trees
  if (total_leaf_size >= this->world_size)
    {
      my_start_count = leafs_per_node * this->rank;
      if (this->rank < this->world_size - 1)
        {
          end_count = leafs_per_node * (this->rank + 1);
        }
      else
        {
          end_count = total_leaf_size;
        }
    }

  for (int i = my_start_count; i < end_count; i++)
    {

      if (!this->trees_leaf_all[tree][i].empty ())
        {
          vector <DataPoint> data_points = this->trees_leaf_all[tree][i];

//      cout << " rank " << rank << " leaf index " << i << " data points size " << data_points.size () << endl;


          vector <vector<DataPoint>> vec (data_points.size ());

#pragma omp parallel for
          for (int k = 0; k < data_points.size (); k++)
            {
              vec[k] = vector<DataPoint> (data_points.size ());
            }

//      cout << " rank " << rank << " tree " << tree << " i " << my_start_count << "distance cal started " << endl;

#pragma omp parallel for
          for (int k = 0; k < data_points.size (); k++)
            {
              for (int j = 0; j < data_points.size (); j++)
                {
                  VALUE_TYPE distance = mathOp.calculate_distance (data_points[k].image_data,
                                                                   data_points[j].image_data);

//                VALUE_TYPE distance = mathOp.calculate_approx_distance(data_points[k].image_data,
//                                                                data_points[j].image_data,0,data_points[j].image_data.size());

                  DataPoint dataPoint;
                  dataPoint.src_index = data_points[k].index;
                  dataPoint.index = data_points[j].index;
                  dataPoint.distance = distance;
                  vec[k][j] = dataPoint;
                }
            }

//      cout << " rank " << rank << " tree " << tree << " i " << my_start_count << "distance cal completed " << endl;

#pragma omp parallel for
          for (int k = 0; k < data_points.size (); k++)
            {
              sort (vec[k].begin (), vec[k].end (),
                    [] (const DataPoint &lhs, const DataPoint &rhs)
                    {
                      return lhs.distance < rhs.distance;
                    });

              vector <DataPoint> sub_vec;
              if (vec.size () > nn)
                {
                  sub_vec = slice (vec[k], 0, nn - 1);
                }
              else
                {
                  sub_vec = vec[k];
                }

              int idx = sub_vec[0].src_index;

              if (local_nns.find (idx) != local_nns.end ())
                {
                  std::vector <DataPoint> dst;
                  auto it = local_nns.find (idx);
                  vector <DataPoint> ex_vec = it->second;
                  std::merge (ex_vec.begin (), ex_vec.end (), sub_vec.begin (),
                              sub_vec.end (), std::back_inserter (dst), [] (const DataPoint &lhs, const DataPoint &rhs)
                              {
                                return lhs.distance < rhs.distance;
                              });
                  dst.erase (unique (dst.begin (), dst.end (),
                                     [] (const DataPoint &lhs,
                                         const DataPoint &rhs)
                                     {
                                       return lhs.index == rhs.index;
                                     }), dst.end ());
                  (it->second) = dst;

                }
              else
                {
#pragma omp critical
                  {
                    if (local_nns.find (idx) == local_nns.end ())
                      {
                        local_nns.insert (pair < int, vector < dmrpt::DataPoint >> (idx, sub_vec));
                        keys.insert(idx);
                      }
                  }
                }
            }
        }
    }
}

std::map<int, vector < dmrpt::DataPoint>> dmrpt::MDRPT::communicate_nns (map<int, vector < dmrpt::DataPoint>> &local_nns, set<int> &keys, int nn) {

  char results[500];
//    char hostname[HOST_NAME_MAX];
//    gethostname(hostname, HOST_NAME_MAX);
  string file_path_stat = output_path + "stats_divided.txt.";
  std::strcpy (results, file_path_stat.c_str ());
//    std::strcpy(results + strlen(file_path_stat.c_str()), hostname);
  ofstream fout (results, std::ios_base::app);

  int *sending_indices_count = new int[this->world_size] ();
  int *receiving_indices_count = new int[this->world_size] ();

  int send_count = 0;
  int *disps_sending_indices = new int[this->world_size] ();

  vector<set<int>> index_distribution_filtered(this->world_size);


  for (int i = 0; i < this->world_size; i++)
    {

      sending_indices_count[i] = this->index_distribution[i].size();
      if(rank==0){
          cout<<" rank "<<rank<<" incoming rank  "<< i <<" "<<this->index_distribution[i].size()<<endl;
         }
      send_count += sending_indices_count[i];
      disps_sending_indices[i] = (i > 0) ? (disps_sending_indices[i - 1] + sending_indices_count[i - 1]) : 0;

    }

  MPI_Alltoall (sending_indices_count, 1, MPI_INT, receiving_indices_count, 1, MPI_INT, MPI_COMM_WORLD);

  cout << " rank " << rank << " first  MPIAll to all completed" << endl;

  int total_receving = 0;

  int *disps_receiving_indices = new int[this->world_size] ();

  for (int i = 0; i < this->world_size; i++)
    {
      total_receving += receiving_indices_count[i];
     cout << " rank " << rank << " receiving from rank "<< i <<" "<< receiving_indices_count[i] << endl;
      disps_receiving_indices[i] = (i > 0) ? (disps_receiving_indices[i - 1] + receiving_indices_count[i - 1]) : 0;
    }

  int *sending_indices = new int[send_count] ();
  VALUE_TYPE *sending_max_dist_thresholds = new VALUE_TYPE[send_count] ();

cout << " rank " << rank << " structure creation completed "<<send_count<<" receive count "<<total_receving << endl;

  struct index_distance_pair {
    float distance;
    int index;
  } in_index_dis[send_count], out_index_dis[total_receving];

cout << " rank " << rank << " structure creation completed" << endl;
  int co_process = 0;
  for (int i = 0; i < this->world_size; i++)
    {
      set<int> process_se_indexes = this->index_distribution[i];
      for (set<int> :: iterator it = process_se_indexes.begin() ; it!=process_se_indexes.end() ; it++)
        {
          in_index_dis[co_process].index = (*it);
          in_index_dis[co_process].distance = local_nns[(*it)][nn - 1].distance;
          co_process++;
        }
    }


  cout << " rank " << rank << " first key traversal completed" << endl;


  MPI_Alltoallv (in_index_dis, sending_indices_count, disps_sending_indices, MPI_FLOAT_INT,
                 out_index_dis,
                 receiving_indices_count, disps_receiving_indices, MPI_FLOAT_INT, MPI_COMM_WORLD);

  cout << " rank " << rank << " second MPI all to all completed" << endl;




// vector<vector<index_distance_pair>> final_indices_allocation (this->world_size);

 vector<vector<index_distance_pair>> final_sent_indices_allocation (this->world_size);

 vector<index_distance_pair> final_sent_indices_to_rank_map (this->original_data.size());

  int my_end_index = this->starting_data_index + this->original_data.size();

#pragma omp parallel for
  for(int i=this->starting_data_index;i<my_end_index;i++) {
     int selected_rank = -1;
     int search_index = i;
     float minium_distance = std::numeric_limits<float>::max();

//     if(local_nns.find (i) != local_nns.end()){
//       selected_rank = this->rank;
//        minium_distance = local_nns[i][nn-1].distance;
//     }

       for (int j = 0; j < this->world_size; j++) {
              int amount = receiving_indices_count[j];
              int offset = disps_receiving_indices[j];

              for (int k = offset; k <( offset+ amount); k++) {
                if (search_index == out_index_dis[k].index) {
                  if (minium_distance > out_index_dis[k].distance) {
                    minium_distance = out_index_dis[k].distance;
                    selected_rank = j;
                  }
                  break;
                }
              }
       }
       index_distance_pair rank_distance;
       rank_distance.index=selected_rank;  //TODO: replace with rank
       rank_distance.distance = minium_distance;
//       final_sent_indices_allocation[selected_rank].push_back (rank_distance);
       final_sent_indices_to_rank_map[search_index-this->starting_data_index]=rank_distance;
 }

   cout << " rank " << rank << " global distance calculation completed" << endl;

   index_distance_pair selected_indices_owner_dst[this->original_data.size()];

   int *sending_selected_indices_ow_co = new int[this->world_size]();
   int *disps_sending_selected_indices_ow_co = new int[this->world_size]();

   int *minimal_selected_rank_sending  = new int[total_receving]();
    index_distance_pair minimal_index_distance[total_receving];


#pragma omp parallel for
   for(int i=0;i<total_receving;i++) {
         minimal_index_distance[i].index =  out_index_dis[i].index;
         minimal_index_distance[i].distance = final_sent_indices_to_rank_map[out_index_dis[i].index-this->starting_data_index].distance;
         minimal_selected_rank_sending[i]=final_sent_indices_to_rank_map[out_index_dis[i].index-this->starting_data_index].index; //TODO: replace
   }


   int *receiving_indices_count_back = new int[this->world_size]();
   int *disps_receiving_indices_count_back = new int[this->world_size]();

   // we recalculate how much we are receiving for minimal dst distribution
   MPI_Alltoall (receiving_indices_count, 1, MPI_INT, receiving_indices_count_back, 1, MPI_INT, MPI_COMM_WORLD);

cout << " rank " << rank << " third MPI completed receiving count from my rank"<<receiving_indices_count_back[this->rank] << endl;

   int total_receivce_back=0;
   for(int i=0;i<this->world_size;i++){
       total_receivce_back +=receiving_indices_count_back[i];
       disps_receiving_indices_count_back[i] = (i > 0) ? (disps_receiving_indices_count_back[i - 1] + receiving_indices_count_back[i - 1]) : 0;
   }

   int *minimal_selected_rank_reciving  = new int[total_receivce_back]();
   index_distance_pair minimal_index_distance_receiv[total_receivce_back];

  MPI_Alltoallv (minimal_index_distance, receiving_indices_count, disps_receiving_indices, MPI_FLOAT_INT,
      minimal_index_distance_receiv,
      receiving_indices_count_back, disps_receiving_indices_count_back, MPI_FLOAT_INT, MPI_COMM_WORLD);

  MPI_Alltoallv (minimal_selected_rank_sending, receiving_indices_count, disps_receiving_indices, MPI_INT,
     minimal_selected_rank_reciving,
     receiving_indices_count_back, disps_receiving_indices_count_back, MPI_INT, MPI_COMM_WORLD);

  cout << " rank " << rank << "  MPI completed  total recive back"<<total_receivce_back << endl;

  vector <vector<index_distance_pair>> final_indices_allocation (this->world_size);



#pragma omp parallel
{
  vector <vector<index_distance_pair>> final_indices_allocation_local (this->world_size);

  #pragma omp  for nowait
  for(int i=0;i<total_receivce_back;i++) {
    index_distance_pair distance_pair;
    distance_pair.index=minimal_index_distance_receiv[i].index;
    distance_pair.distance = minimal_index_distance_receiv[i].distance;
    final_indices_allocation_local[minimal_selected_rank_reciving[i]].push_back (distance_pair);

  }

#pragma omp critical
  {
    for(int i=0;i<this->world_size;i++){

     final_indices_allocation[i].insert (final_indices_allocation[i].end(),
                                         final_indices_allocation_local[i].begin(),final_indices_allocation_local[i].end());
    }

  }

}



 cout<<" final indices size for my rank "<<rank<<" size "<<final_indices_allocation[this->rank].size()<<endl;


  cout << " rank " << rank << "  MPI minloc completed all to all completed" << endl;


  std::map<int, vector<DataPoint>> final_nn_sending_map;

  std::map<int, vector<DataPoint>> final_nn_map;

  int *sending_selected_indices_count = new int[this->world_size] ();
  int *sending_selected_indices_nn_count = new int[this->world_size] ();

  int *receiving_selected_indices_count = new int[this->world_size] ();
  int *receiving_selected_indices_nn_count = new int[this->world_size] ();

  int total_selected_indices_count = 0;
  int total_selected_indices_nn_count = 0;


  for (int i = 0; i < this->world_size; i++)
    {
      int count = 0;
      int nn_count = 0;

#pragma omp parallel for
      for (int j = 0; j < final_indices_allocation[i].size (); j++)
        {
          index_distance_pair in_dis = final_indices_allocation[i][j];
          int selected_index = in_dis.index;
          float dst_th = in_dis.distance;
          if (i != this->rank)
            {
              if (local_nns.find (selected_index) != local_nns.end ())
                {
                  vector <dmrpt::DataPoint> target;
                  std::copy_if (local_nns[selected_index].begin (),
                                local_nns[selected_index].end (),
                                std::back_inserter (target),

                                [dst_th] (
                                    dmrpt::DataPoint dataPoint
                                )
                                {
                                  return dataPoint.distance < dst_th;
                                });
                  if (target.size () > 0)
                    {
#pragma omp critical
{
                   if (final_nn_sending_map.find (selected_index) == final_nn_sending_map.end ())
                      {
                        final_nn_sending_map.insert (pair < int, vector < DataPoint >>
                                                                                    (selected_index, target));
                        sending_selected_indices_nn_count[i] += target.size ();
                        sending_selected_indices_count[i] += 1;
                      }
                    }
//                    local_nns.erase(local_nns.find(index));
                }
                }
            }
          else
            {
#pragma omp critical
              final_nn_map.insert (pair < int, vector < DataPoint >> (selected_index, local_nns[selected_index]));
            }
        }
    }

  cout << " rank " << rank << " allocation completed" << endl;

  MPI_Alltoall (sending_selected_indices_count,
                1, MPI_INT, receiving_selected_indices_count, 1, MPI_INT, MPI_COMM_WORLD);

  int total_receiving_count = 0;

  int *disps_receiving_selected_indices = new int[this->world_size] ();
  int *disps_sending_selected_indices = new int[this->world_size] ();
  int *disps_sending_selected_nn_indices = new int[this->world_size] ();
  int *disps_receiving_selected_nn_indices = new int[this->world_size] ();

  for (int i = 0; i < this->world_size; i++)
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

  int *sending_selected_indices = new int[total_selected_indices_count] ();

  int *sending_selected_nn_count_for_each_index = new int[total_selected_indices_count] ();
//  int *sending_selected_nn_indices = new int[total_selected_indices_nn_count] ();
//  VALUE_TYPE *sending_selected_nn_dst = new VALUE_TYPE[total_selected_indices_nn_count] ();
   index_distance_pair sending_selected_nn[total_selected_indices_nn_count];



  int inc = 0;
  int selected_nn = 0;
  for (int i = 0; i < this->world_size; i++)
    {
      total_receiving_count += receiving_selected_indices_count[i];
      if (i != this->rank)
        {
          vector<index_distance_pair> final_indices = final_indices_allocation[i];
          for (int j = 0; j < final_indices.size (); j++)
            {
              if (final_nn_sending_map.find (final_indices[j].index) != final_nn_sending_map.end ())
                {
                  vector <dmrpt::DataPoint> nn_sending = final_nn_sending_map[final_indices[j].index];
                  if (nn_sending.size () > 0)
                    {
                      sending_selected_indices[inc] = final_indices[j].index;
                      for (int k = 0; k < nn_sending.size (); k++)
                        {
                            sending_selected_nn[selected_nn].index = nn_sending[k].index;
                            sending_selected_nn[selected_nn].distance = nn_sending[k].distance;
                          selected_nn++;
                        }
                      sending_selected_nn_count_for_each_index[inc] = nn_sending.size ();
                      inc++;
                    }
                }
            }
        }
    }

  int *receiving_selected_nn_indices_count = new int[total_receiving_count] ();

  int *receiving_selected_indices = new int[total_receiving_count] ();

  MPI_Alltoallv (sending_selected_nn_count_for_each_index, sending_selected_indices_count,
                 disps_sending_selected_indices, MPI_INT, receiving_selected_nn_indices_count,
                 receiving_selected_indices_count, disps_receiving_selected_indices, MPI_INT, MPI_COMM_WORLD);

  MPI_Alltoallv (sending_selected_indices, sending_selected_indices_count, disps_sending_selected_indices, MPI_INT,
                 receiving_selected_indices,
                 receiving_selected_indices_count, disps_receiving_selected_indices, MPI_INT, MPI_COMM_WORLD);

  int total_receiving_nn_count = 0;

  int *receiving_selected_nn_indices_count_process = new int[this->world_size] ();

  for (int i = 0; i < this->world_size; i++)
    {
      int co = receiving_selected_indices_count[i];
      int offset = disps_receiving_selected_indices[i];
//        int per_pro_co = 0;
      for (int k = offset; k < (co + offset); k++)
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

  MPI_Alltoallv (sending_selected_nn, sending_selected_indices_nn_count, disps_sending_selected_nn_indices,
                 MPI_FLOAT_INT,
                 receving_selected_nn,
                 receiving_selected_nn_indices_count_process, disps_receiving_selected_nn_indices, MPI_FLOAT_INT,
                 MPI_COMM_WORLD
  );

  int nn_index = 0;
  for (int i = 0; i < total_receiving_count; i++)
    {
      int src_index = receiving_selected_indices[i];
      int nn_count = receiving_selected_nn_indices_count[i];
      vector <DataPoint> vec;
      for (int j = 0; j < nn_count; j++)
        {
          int nn_indi = receving_selected_nn[nn_index].index;
          VALUE_TYPE distance = receving_selected_nn[nn_index].distance;
          DataPoint dataPoint;
          dataPoint.src_index = src_index;
          dataPoint.index = nn_indi;
          dataPoint.distance = distance;
          vec.push_back (dataPoint);
          nn_index++;
        }

      auto its = final_nn_map.find (src_index);
      if (its == final_nn_map.end ())
        {
          final_nn_map.insert (pair < int, vector < DataPoint >>
                                                              (src_index, vec));
        }
      else
        {
          vector <DataPoint> dst;
          vector <DataPoint> ex_vec = its->second;
          sort (vec.begin (),vec.end (),[] (
                    const DataPoint &lhs,
                    const DataPoint &rhs
                )
                {
                  return lhs.distance < rhs.
                      distance;
                });
          std::merge (ex_vec.begin (), ex_vec.end (), vec.begin (),
                      vec.end (), std::back_inserter (dst),
                      [] (
                          const DataPoint &lhs,
                          const DataPoint &rhs
                      ){
                        return lhs.distance < rhs.distance;
                      });
          dst.erase (unique (dst.begin (), dst.end (), [] (const DataPoint &lhs,
                                                           const DataPoint &rhs){
                       return lhs.index == rhs.index;
                     }),
                     dst.end ()
          );
          (its->second) = dst;
        }
    }

  int *execution_times = new int[1];

  int *execution_times_global = new int[1];
  execution_times[0] =
      total_receiving_nn_count;

  MPI_Allreduce (execution_times, execution_times_global,
                 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  fout << " rank " << rank << " total receiving nn indicies " << execution_times_global[0] <<
       endl;

  delete[]
      sending_indices_count;
  delete[]
      receiving_indices_count;
  delete[]
      disps_receiving_indices;
  delete[]
      disps_sending_indices;
  delete[]
      sending_indices;
  delete[]
      sending_max_dist_thresholds;
//  delete[]
//      receiving_indices;
//  delete[]
//      receiving_max_dist_thresholds;
  delete[]
      sending_selected_indices_count;
  delete[]
      sending_selected_indices_nn_count;
  delete[]
      receiving_selected_indices_count;
  delete[]
      receiving_selected_indices_nn_count;
  delete[]
      sending_selected_indices;
  delete[]
      sending_selected_nn_count_for_each_index;
//  delete[]
//      sending_selected_nn_indices;
//  delete[]
//      sending_selected_nn_dst;
  delete[]
      disps_receiving_selected_indices;
  delete[]
      disps_sending_selected_indices;
  delete[]
      disps_sending_selected_nn_indices;
  delete[]
      disps_receiving_selected_nn_indices;
  delete[]
      receiving_selected_indices;
//  delete[]
//      receiving_selected_nn_indices;
//  delete[]
//      receiving_selected_nn_dst;
  delete[]
      receiving_selected_nn_indices_count_process;

  return
      final_nn_map;

}

std::map<int, vector < dmrpt::DataPoint>>

dmrpt::MDRPT::gather_nns (int nn, ofstream  &fout)
{

  cout << " rank " << rank << "gathering started " << endl;

  auto start_distance = high_resolution_clock::now ();

  int chunk_size = this->total_data_set_size / this->world_size;

  int last_chunk_size = this->total_data_set_size - chunk_size * (this->world_size - 1);

  int my_chunk_size = chunk_size;
  int my_starting_index = this->rank * chunk_size;

  int my_end_index = 0;
  if (this->rank < this->world_size - 1)
    {
      my_end_index = (this->rank + 1) * chunk_size;
    }
  else
    {
      my_end_index = this->total_data_set_size;
      my_chunk_size = last_chunk_size;
    }

  std::map<int, vector<DataPoint>> local_nn_map;

  set<int> keys;

  for (int i = 0; i < ntrees; i++)
    {
      this->calculate_nns (local_nn_map, keys, i, 2 * nn);
    }

  cout << " rank " << rank << " distance calculation completed " << endl;

  auto stop_distance = high_resolution_clock::now ();
  auto distance_time = duration_cast<microseconds> (stop_distance - start_distance);

  auto start_query = high_resolution_clock::now ();

  std::map<int, vector<dmrpt::DataPoint>> final_map = communicate_nns (local_nn_map, keys, nn);

  auto stop_query = high_resolution_clock::now ();
  auto query_time = duration_cast<microseconds> (stop_query - start_query);

  double *execution_times = new double[2];

  double *execution_times_global = new double[2];
  execution_times[0] = distance_time.count () / 1000;
  execution_times[1] = query_time.count () / 1000;

  MPI_Allreduce (execution_times, execution_times_global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  fout << " distance calculation " << (execution_times_global[0] / this->world_size) << " communication time "
       << (execution_times_global[1] / this->world_size) << endl;

  return final_map;
}











