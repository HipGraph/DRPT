#include <iostream>#include <fstream>#include <string>#include <vector>#include <sstream>#include "mpi.h"#include <cstring>#include <cmath>#include <map>#include "../algo/drpt_local.hpp"#include "file_writer.hpp"#include <algorithm>using namespace std;int get_number_of_digits(int number) {	int count = 0;	while(number != 0) {		number = number / 10;		count++;	}	return count;}//template<typename IT> vector<vector<IT>> dmrpt::FileWriter<IT>::alloc2d(int rows, int cols)//{//	int size = rows*cols;//	IT *data = (IT*)malloc(size*sizeof(IT));//	IT **array = (IT**)malloc(rows*sizeof(IT*));//	for (int i=0; i<rows; i++)//		array[i] = &(data[i*cols]);//	return array;////}template<typename IT>int dmrpt::FileWriter<IT>::mpi_write_edge_list(std::map<int, vector<dmrpt::DataPoint>> &data_points, string output_path, int  nn,		int rank, int world_size, bool filter_self_edges) {	MPI_Offset offset;	MPI_File   file;	MPI_Status status;	MPI_Datatype num_as_string;	MPI_Datatype localarray;	char *const fmt="%d ";	char *const endfmt="%d\n";	int totalchars=0;	int local_rows = 0;	for (auto i = data_points.begin(); i != data_points.end(); i++)	{		vector<dmrpt::DataPoint> vec = i->second;		int l_rows = 0;		if(filter_self_edges){			vec.erase(remove_if(vec.begin(),vec.end(),[&](dmrpt::DataPoint dataPoint) {			  return dataPoint.src_index == dataPoint.index;}), vec.end());		}		l_rows =  vec.size() >= nn ? nn : vec.size();		local_rows += l_rows;	}	int cols = 2;	int *send_counts = new int[world_size]();	int *recv_counts = new int[world_size]();	for(int i=0;i<world_size;i++){		send_counts[i]=local_rows;	}	MPI_Alltoall (send_counts, 1, MPI_INT, recv_counts, 1,			MPI_INT, MPI_COMM_WORLD);	int startrow =0;	int global_rows =0;	for(int j=0;j<world_size;j++){		if (j<rank){			startrow += recv_counts[j];		}		global_rows += recv_counts[j];	}//	data = alloc2d(local_rows,cols);	vector<vector<IT>> data = vector<vector<IT>>(local_rows);	for(int i=0;i<local_rows;i++){		data[i]=vector<IT>(cols,-1);	}	int ind = 0;	for (auto i = data_points.begin(); i != data_points.end(); i++)	{		vector<DataPoint> vec = i->second;		if(filter_self_edges){			vec.erase(remove_if(vec.begin(),vec.end(),[&](dmrpt::DataPoint dataPoint) {			  return dataPoint.src_index == dataPoint.index;}), vec.end());		}		for (int j = 0; j < (vec.size() >= nn ? nn : vec.size()); j++)		{			if (ind < local_rows)			{				data[ind][0] =  vec[j].src_index +1;				data[ind][1] = vec[j].index +1;				totalchars = totalchars+ get_number_of_digits(vec[j].src_index +1)+get_number_of_digits(vec[j].index +1)+2;				ind++;			}		}	}//	MPI_Type_contiguous(charspernum, MPI_CHAR, &num_as_string);//	MPI_Type_commit(&num_as_string);	int *send_counts_bytes = new int[world_size]();	int *recv_counts_bytes = new int[world_size]();	for(int i=0;i<world_size;i++){		send_counts_bytes[i]=totalchars;	}	MPI_Alltoall (send_counts_bytes, 1, MPI_INT, recv_counts_bytes, 1,			MPI_INT, MPI_COMM_WORLD);	int bytes_disps=0;	for(int j=0;j<world_size;j++){		if (j<rank)		{			bytes_disps += recv_counts_bytes[j];		}	}	cout<<"rank "<<rank <<" before sprintf total chars "<<totalchars<<" bytes recvied "<<bytes_disps<<endl;	/* convert our data into txt */	char * data_as_txt = new char[totalchars]();	int count = 0;	int current_total_chars=0;	for (int i=0; i<local_rows; i++) {		for (int j=0; j<cols-1; j++) {			int local_chars = get_number_of_digits(data[i][j]);			sprintf(&data_as_txt[current_total_chars],fmt,data[i][j]);			current_total_chars =current_total_chars+local_chars+1;		}		int local_chars = get_number_of_digits(data[i][cols-1]);		sprintf(&data_as_txt[current_total_chars],endfmt,data[i][cols-1]);		current_total_chars =current_total_chars +local_chars+1;	}	cout<<"rank "<<rank <<" sprint f completed "<<endl;	/* create a type describing our piece of the array *///	int globalsizes[2] = {global_rows, cols};//	int localsizes [2] = {local_rows, cols};//	int starts[2]      = {startrow, 0};//	int order          = MPI_ORDER_C;////	MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, MPI_CHAR, &localarray);//	MPI_Type_commit(&localarray);	/* open the file, and set the view */	MPI_File_open(MPI_COMM_WORLD, output_path.c_str(),			MPI_MODE_CREATE|MPI_MODE_WRONLY,			MPI_INFO_NULL, &file);	MPI_File_set_view(file, bytes_disps,  MPI_CHAR, MPI_CHAR,			(char*)"native", MPI_INFO_NULL);	cout << "rank "<<rank<<"file view completed " << endl;	MPI_File_write_all(file, data_as_txt, totalchars, MPI_CHAR, &status);	MPI_File_close(&file);	cout << "rank "<<rank<<"file writing completed for path"<<output_path << endl;//	free(data);//	delete[] data_as_txt;	delete[] send_counts;	delete[] recv_counts;	delete[] send_counts_bytes;	delete[] recv_counts_bytes;	return 0;}template class dmrpt::FileWriter<int>;