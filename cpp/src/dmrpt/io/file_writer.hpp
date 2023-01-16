#ifndef _FILE_WRITER_H_#define _FILE_WRITER_H_#include <iostream>#include <fstream>#include <string>#include <vector>#include <sstream>#include "mpi.h"#include <cstring>#include <cmath>#include "../algo/drpt_local.hpp"#include <map>using namespace std;namespace dmrpt{	template<typename IT>	class FileWriter	{	 private:		IT** alloc2d(int rows, int cols);	 public:		void mpi_write_edge_list(std::map<int, vector<DataPoint>>& data_points, string output_path, int nn,				int rank, int world_size);	};}#endif //_FILE_WRITER_H_