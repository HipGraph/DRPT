#include "dmrpt/io/image_reader.hpp"
#include "dmrpt/math/matrix_multiply.hpp"
#include "dmrpt/algo/drpt.hpp"
#include "dmrpt/algo/mdrpt.hpp"
#include <vector>
#include <mpi.h>
#include <string>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <chrono>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <cstring>

using namespace std;
using namespace dmrpt;
using namespace std::chrono;

int main (int argc, char *argv[])
{

  string input_path = "";
  string output_path = "";
  int algo = 0;
  int data_set_size = 0;
  int dimension = 0;
  int ntrees = 10;
  int tree_depth = 0;
  int transfer_threshold = 10;
  int donate_per = 10;
  double density = 0;
  int nn = 0;
  int batch_size = 1000;
  double tree_depth_ratio = 0.5;
  bool use_locality_optimization = true;
  int local_tree_offset = 3;

  for (int p = 0; p < argc; p++)
    {

      if (strcmp (argv[p], "-input") == 0)
        {
          input_path = argv[p + 1];
        }
      else if (strcmp (argv[p], "-output") == 0)
        {
          output_path = argv[p + 1];
        }
      else if (strcmp (argv[p], "-algo") == 0)
        {
          algo = atoi (argv[p + 1]);
        }
      else if (strcmp (argv[p], "-data-set-size") == 0)
        {
          data_set_size = atoi (argv[p + 1]);
        }
      else if (strcmp (argv[p], "-dimension") == 0)
        {
          dimension = atoi (argv[p + 1]);
        }
      else if (strcmp (argv[p], "-ntrees") == 0)
        {
          ntrees = atoi (argv[p + 1]);
        }
      else if (strcmp (argv[p], "-tree-depth") == 0)
        {
          tree_depth = atoi (argv[p + 1]);
        }
      else if (strcmp (argv[p], "-tree-depth-ratio") == 0)
        {
          tree_depth_ratio = stof (argv[p + 1]);
        }
      else if (strcmp (argv[p], "-density") == 0)
        {
          density = atoi (argv[p + 1]);
        }
      else if (strcmp (argv[p], "-nn") == 0)
        {
          nn = atoi (argv[p + 1]);
        }
      else if (strcmp (argv[p], "-locality") == 0)
        {
          use_locality_optimization = atoi (argv[p + 1]) == 1 ? true : false;
        }
      else if (strcmp (argv[p], "-local-tree-offset") == 0)
        {
          local_tree_offset = atoi (argv[p + 1]);
        }

    }

  if (input_path.size () == 0)
    {
      printf ("Valid input path needed!...\n");
      exit (1);
    }
  if (output_path.size () == 0)
    {
      printf ("Valid out path needed!...\n");
      exit (1);
    }
  if (data_set_size == 0)
    {
      printf ("Dataset size should be greater than 0\n");
      exit (1);
    }

  if (dimension == 0)
    {
      printf ("Dimension size should be greater than 0\n");
      exit (1);
    }

  if (nn == 0)
    {
      printf ("Nearest neighbours size should be greater than 0\n");
      exit (1);
    }
  if (density == 0)
    {
      density = 1.0 / sqrt (dimension);
    }

  int rank, size;

  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  if (tree_depth == 0)
    {
      tree_depth = static_cast<int>(log2 ((data_set_size / size)) - 2);
      cout << " tree depth " << tree_depth << endl;
    }

  ImageReader imageReader;

  char hostname[HOST_NAME_MAX];
  char stats[500];
  char results[500];

  int host = gethostname (hostname, HOST_NAME_MAX);

  string file_path_stat = output_path + "stats.txt.";
  std::strcpy (stats, file_path_stat.c_str ());
//    std::strcpy(stats + strlen(file_path_stat.c_str()), hostname);

  string file_path = output_path + "results.txt.";
  std::strcpy (results, file_path.c_str ());
  std::strcpy (results + strlen (file_path.c_str ()), hostname);

  ofstream fout (stats, std::ios_base::app);
  ofstream fout1 (results, std::ios_base::app);

  auto start_io_index = high_resolution_clock::now ();

//    vector <vector<VALUE_TYPE>> imagedatas = imageReader.read_MNIST(
//            input_path, data_set_size, dimension,
//            rank, size);

//    vector <vector<VALUE_TYPE>> imagedatas = imageReader.read_File(
//            input_path, data_set_size, dimension,
//            rank, size);

//    vector <vector<VALUE_TYPE>> imagedatas = imageReader.mpi_file_read(
//            input_path, rank, size,
//            400000, data_set_size,' ',dimension);


  vector <vector<VALUE_TYPE>> imagedatas = imageReader.mpi_file_read (
      input_path, rank, size,
      400000, data_set_size, 384, 8, dimension);

  cout << " size " << imagedatas.size () << " *" << imagedatas[0].size () << endl;

  auto stop_io_index = high_resolution_clock::now ();
  auto io_time = duration_cast<microseconds> (stop_io_index - start_io_index);

  cout << "Rank " << rank << " Size of  images data " << imagedatas.size () << "*" << imagedatas[0].size () << endl;

  MathOp mathOp;

  int rows = imagedatas[0].size ();
  int cols = imagedatas.size ();

  MPI_Barrier (MPI_COMM_WORLD);

  int chunk_size = data_set_size / size;

  cout<<" total data set size "<<data_set_size<<endl;

  MDRPT mdrpt = MDRPT (ntrees, algo, tree_depth, tree_depth_ratio, local_tree_offset,data_set_size, rows, rank, size, input_path,
                       output_path);

  auto start_index_buildling = high_resolution_clock::now ();

  cout << " start growing trees " << rank << endl;

  mdrpt.grow_trees (imagedatas, density, use_locality_optimization,nn);
  cout << " completed growing trees " << rank << endl;
  auto stop_index_building = high_resolution_clock::now ();

  auto duration_index_building = duration_cast<microseconds> (stop_index_building - start_index_buildling);

  int co = 0;

  auto start_query = high_resolution_clock::now ();

  map<int, vector<DataPoint>> data_points = mdrpt.gather_nns (nn);

  auto stop_query = high_resolution_clock::now ();
  auto duration_query = duration_cast<microseconds> (stop_query - start_query);

  cout << "Time taken for total query "
       << duration_query.count () << " microseconds" << endl;

  if (fout.is_open ())
    {
      if (data_points.size () > 0)
        {

          cout << " rank " << rank << data_points.size () << endl;

          for (int k = 0; k < data_points.size (); k++)
            {
              if (data_points[k].size () > 0)
                {
                  vector <DataPoint> vec = data_points[k];
                  if (vec.size () > 0)
                    {
                      for (int l = 0; l < (vec.size () >= 10 ? 10 : vec.size ()); l++)
                        {
                          if (vec[l].src_index != vec[l].index)
                            {
                              fout1 << vec[l].src_index + 1 << ' ' << vec[l].index + 1 << endl;
                            }
                        }
                    }
                }
            }
        }
    }

  double *execution_times = new double[3];

  double *execution_times_global = new double[3];
  execution_times[0] = io_time.count () / 1000;
  execution_times[1] = duration_index_building.count () / 1000;
  execution_times[2] = duration_query.count () / 1000;

  MPI_Allreduce (execution_times, execution_times_global, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  fout << rank << ' ' << (execution_times_global[0] / size) << ' ' << (execution_times_global[1] / size) << ' '
       << (execution_times_global[2] / size)
       << endl;

  delete[] execution_times;
  delete[] execution_times_global;

  MPI_Finalize ();
}