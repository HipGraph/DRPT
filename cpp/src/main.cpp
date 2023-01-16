#include "dmrpt/io/file_reader.hpp"
#include "dmrpt/math/math_operations.hpp"
#include "dmrpt/algo/drpt_local.hpp"
#include "dmrpt/algo/mdrpt.hpp"
#include "dmrpt/io/file_writer.hpp"
#include "dmrpt/io/file_writer.cpp"
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

int main(int argc, char* argv[])
{
	string input_path = "";
	string output_path = "";
	long data_set_size = 0;
	int dimension = 0;
	int ntrees = 10;
	int tree_depth = 0;
	double density = 0;
	int nn = 0;
	double tree_depth_ratio = 0.6;
	bool use_locality_optimization = true;
	int local_tree_offset = 2;
	int file_format = 0;
	int data_starting_index = 8;

	int bytes_for_data_type = 4;

	for (int p = 0; p < argc; p++)
	{
		if (strcmp(argv[p], "-input") == 0)
		{
			input_path = argv[p + 1];
		}
		else if (strcmp(argv[p], "-output") == 0)
		{
			output_path = argv[p + 1];
		}
		else if (strcmp(argv[p], "-data-set-size") == 0)
		{
			data_set_size = atol(argv[p + 1]);
		}
		else if (strcmp(argv[p], "-dimension") == 0)
		{
			dimension = atoi(argv[p + 1]);
		}
		else if (strcmp(argv[p], "-ntrees") == 0)
		{
			ntrees = atoi(argv[p + 1]);
		}
		else if (strcmp(argv[p], "-tree-depth-ratio") == 0)
		{
			tree_depth_ratio = stof(argv[p + 1]);
		}
		else if (strcmp(argv[p], "-density") == 0)
		{
			density = atoi(argv[p + 1]);
		}
		else if (strcmp(argv[p], "-nn") == 0)
		{
			nn = atoi(argv[p + 1]);
		}
		else if (strcmp(argv[p], "-locality") == 0)
		{
			use_locality_optimization = atoi(argv[p + 1]) == 1 ? true : false;
		}
		else if (strcmp(argv[p], "-local-tree-offset") == 0)
		{
			local_tree_offset = atoi(argv[p + 1]);
		}
		else if (strcmp(argv[p], "-data-file-format") == 0)
		{
			file_format = atoi(argv[p + 1]);
		}
		else if (strcmp(argv[p], "-bytes-for-data-type") == 0)
		{
			bytes_for_data_type = atoi(argv[p + 1]);
		}
		else if (strcmp(argv[p], "-data-starting-index") == 0)
		{
			data_starting_index = atoi(argv[p + 1]);
		}
	}

	if (input_path.size() == 0)
	{
		printf("Valid input path needed!...\n");
		exit(1);
	}

	if (output_path.size() == 0)
	{
		printf("Valid out path needed!...\n");
		exit(1);
	}

	if (data_set_size == 0)
	{
		printf("Dataset size should be greater than 0\n");
		exit(1);
	}

	if (dimension == 0)
	{
		printf("Dimension size should be greater than 0\n");
		exit(1);
	}

	if (nn == 0)
	{
		printf("Nearest neighbours size should be greater than 0\n");
		exit(1);
	}

	if (density == 0)
	{
		density = 1.0 / sqrt(dimension);
	}


	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (tree_depth == 0)
	{
		tree_depth = static_cast<int>(log2((data_set_size / size)) - 2);
		cout << " tree depth " << tree_depth << endl;
	}

	ImageReader imageReader;

	char hostname[HOST_NAME_MAX];
	char stats[500];
	char results[500];

	char data[500];

	int host = gethostname(hostname, HOST_NAME_MAX);

	string file_path_stat = output_path + "stats.txt.";
	std::strcpy(stats, file_path_stat.c_str());
//    std::strcpy(stats + strlen(file_path_stat.c_str()), hostname);

	string file_path = output_path + "results.txt.";
	std::strcpy(results, file_path.c_str());
	std::strcpy(results + strlen(file_path.c_str()), hostname);

	ofstream fout(stats, std::ios_base::app);
	ofstream fout1(results, std::ios_base::app);

	auto start_io_index = high_resolution_clock::now();

	vector<vector<VALUE_TYPE>> imagedatas;

	if (file_format == 0)
	{
		imagedatas = imageReader.read_ubyte(
				input_path, data_set_size, dimension,
				rank, size);
	}
	else if (file_format == 1)
	{
		imagedatas = imageReader.mpi_file_read(
				input_path, rank,
				size,
				400000,
				data_set_size,
				bytes_for_data_type,
				data_starting_index, dimension);
	}
	else
	{
		printf("Unsupported data file format, ubyte and bin file formats are supported!...\n");
		exit(1);
	}

	cout << " size " << imagedatas.size() << " *" << imagedatas[0].size() << endl;


	auto stop_io_index = high_resolution_clock::now();
	auto io_time = duration_cast<microseconds>(stop_io_index - start_io_index);

	cout << "Rank " << rank << " Size of  images data " << imagedatas.size() << "*" << imagedatas[0].size() << endl;

	MathOp mathOp;

	int rows = imagedatas[0].size();
	int cols = imagedatas.size();

	MPI_Barrier(MPI_COMM_WORLD);

	int chunk_size = data_set_size / size;

	cout << " total data set size " << data_set_size << endl;

	MDRPT mdrpt = MDRPT(ntrees, tree_depth, tree_depth_ratio, local_tree_offset, data_set_size, cols,
			rows, rank, size, input_path,
			output_path);

	auto start_index_buildling = high_resolution_clock::now();

	cout << " start growing trees " << rank << endl;

	mdrpt.grow_trees(imagedatas, density, use_locality_optimization, nn, fout);
	cout << " completed growing trees " << rank << endl;

	auto stop_index_building = high_resolution_clock::now();

	auto duration_index_building = duration_cast<microseconds>(stop_index_building - start_index_buildling);

	int co = 0;

	auto start_query = high_resolution_clock::now();

	map<int, vector<DataPoint>> data_points = mdrpt.gather_nns(nn, fout);

	auto stop_query = high_resolution_clock::now();
	auto duration_query = duration_cast<microseconds>(stop_query - start_query);

	cout << "Time taken for total query "
		 << duration_query.count() << " microseconds" << endl;
//
//	FileWriter<int> fileWriter;
//
//	cout << "rank "<<rank<<"file writer initialization completed" << endl;
//	fileWriter.mpi_write_edge_list(data_points,file_path,nn,rank,size);

	if (fout.is_open())
	{
		if (data_points.size() > 0)
		{

			cout << " rank " << rank << data_points.size() << endl;

			for (int k = 0; k < data_points.size(); k++)
			{
				if (data_points[k].size() > 0)
				{
					vector<DataPoint> vec = data_points[k];
					if (vec.size() > 0)
					{
						for (int l = 0; l < (vec.size() >= nn ? nn : vec.size()); l++)
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

	double* execution_times = new double[3];

	double* execution_times_global = new double[3];
	execution_times[0] = io_time.count() / 1000;
	execution_times[1] = duration_index_building.count() / 1000;
	execution_times[2] = duration_query.count() / 1000;

	MPI_Allreduce(execution_times, execution_times_global, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	fout << rank << ' ' << (execution_times_global[0] / size) << ' ' << (execution_times_global[1] / size) << ' '
		 << (execution_times_global[2] / size)
		 << endl;

	delete[] execution_times;
	delete[] execution_times_global;

	MPI_Finalize();
}