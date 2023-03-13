#ifndef DISTRIBUTED_MRPT_IMAGE_READER_H
#define DISTRIBUTED_MRPT_IMAGE_READER_H
#include "../math/math_operations.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
namespace drpt {
class ImageReader {
 public:
  vector <vector<VALUE_TYPE>> read_ubyte(string path, int no_of_images,
										 int dimension, int rank,
										 int world_size);
  vector <vector<VALUE_TYPE>> read_mnist_labels(string path, int no_of_images,
												int dimension, int rank,
												int world_size);
  vector <vector<VALUE_TYPE>> read_File(string path, int no_of_data_points,
										int dimension, int rank, int world_size);
  vector <vector<VALUE_TYPE>> mpi_file_read(string path, int rank,
											int world_size, int overlap,
											long total_data_set_size, char delim,
											int dimension);
  vector <vector<VALUE_TYPE>> mpi_file_read(string path, int rank, int world_size, int overlap,
		  long total_data_set_size, int data_type_bytes,  int offset, int dimension);
  int reverse_int(int i);
};
} // namespace drpt

#endif // DISTRIBUTED_MRPT_IMAGE_READER_H
