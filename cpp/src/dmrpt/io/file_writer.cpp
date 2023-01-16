#include <iostream>#include <fstream>#include <string>#include <vector>#include <sstream>#include "mpi.h"#include <cstring>#include <cmath>#include <map>#include "../algo/drpt_local.hpp"#include "file_writer.hpp"using namespace std;template<typename IT> IT** dmrpt::FileWriter<IT>::alloc2d(int rows, int cols){	int size = rows*cols;	IT *data = (IT*)malloc(size*sizeof(IT));	IT **array = (IT**)malloc(rows*sizeof(IT*));	for (int i=0; i<rows; i++)		array[i] = &(data[i*cols]);	return array;}template<typename IT>void dmrpt::FileWriter<IT>::mpi_write_edge_list(std::map<int, vector<dmrpt::DataPoint>> &data_points, string output_path, int  nn,		int rank, int world_size) {	MPI_Offset offset;	MPI_File   file;	MPI_Status status;	MPI_Datatype num_as_string;	MPI_Datatype localarray;	IT **data;	char *const fmt="%1d ";	char *const endfmt="%1d\n";	const int charspernum=9;	int local_rows = 0;	for (int i=0; i<data_points.size(); i++)	{		vector<DataPoint> vec = data_points[i];		int l_rows =  (vec.size() >= nn ? nn : vec.size());		local_rows += l_rows;	}	int cols = 2;	int *local_total = new int[1] ();	int *global_total = new int[1] ();	local_total[0]=local_rows;//	MPI_Allreduce (local_total, global_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);//	int global_rows_n = global_total[0];	cout<<"mpi reduce all"<<global_rows_n<<endl;	int *send_counts = new int[world_size]();	int *recv_counts = new int[world_size]();	for(int i=0;i<world_size;i++){		send_counts[i]=local_rows;	}	MPI_Alltoall (send_counts, 1, MPI_INT, recv_counts, 1,			MPI_INT, MPI_COMM_WORLD);	int startrow =0;	int global_rows =0;	for(int j=0;j<world_size;j++){		if (j<rank){			startrow += recv_counts[j];		}		global_rows += recv_counts[j];	}	int endrow = startrow+local_rows-1;	cout << "rank "<<rank<<"global_rows" <<global_rows<<" local_rows "<<local_rows<<endl;	data = alloc2d(local_rows,cols);	cout << "rank "<<rank<<"data allocation completed" << endl;	int ind = 0;	for (int i=0; i<data_points.size(); i++)	{		vector<DataPoint> vec = data_points[i];		for (int j = 0; j < (vec.size() >= nn ? nn : vec.size()); j++)		{			if (vec[j].src_index != vec[j].index && ind < local_rows)			{				data[ind][0] =  vec[j].src_index;				data[ind][1] = vec[j].index;				ind++;			}		}	}	cout << "rank "<<rank<<"data filling completed" << endl;	MPI_Type_contiguous(charspernum, MPI_CHAR, &num_as_string);	MPI_Type_commit(&num_as_string);	/* convert our data into txt */	char *data_as_txt = (char*)malloc(local_rows*cols*charspernum*sizeof(char));	int count = 0;	for (int i=0; i<local_rows; i++) {		for (int j=0; j<cols-1; j++) {			sprintf(&data_as_txt[count*charspernum],fmt,data[i][j]);			count++;		}		sprintf(&data_as_txt[count*charspernum],endfmt,data[i][cols-1]);		count++;	}	/* create a type describing our piece of the array */	int globalsizes[2] = {global_rows, cols};	int localsizes [2] = {local_rows, cols};	int starts[2]      = {startrow, 0};	int order          = MPI_ORDER_C;	MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, num_as_string, &localarray);	MPI_Type_commit(&localarray);	/* open the file, and set the view */	MPI_File_open(MPI_COMM_WORLD, output_path.c_str(),			MPI_MODE_CREATE|MPI_MODE_WRONLY,			MPI_INFO_NULL, &file);	MPI_File_set_view(file, 0,  MPI_CHAR, localarray,			"native", MPI_INFO_NULL);	MPI_File_write_all(file, data_as_txt, local_rows*cols, num_as_string, &status);	MPI_File_close(&file);	cout << "rank "<<rank<<"file writing completed for path"<<output_path << endl;	MPI_Type_free(&localarray);	MPI_Type_free(&num_as_string);	free(data[0]);	free(data);}template class dmrpt::FileWriter<int>;