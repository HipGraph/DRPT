
#ifndef DISTRIBUTED_MRPT_IMAGE_READER_H
#define DISTRIBUTED_MRPT_IMAGE_READER_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
# include "../math/matrix_multiply.hpp"

using namespace std;
namespace dmrpt{
    class ImageReader {
      public:
//        vector<VALUE_TYPE> readImage(string path);
//        vector<vector<VALUE_TYPE>> readImages(vector<string> imagePaths);
        vector<vector<VALUE_TYPE>> read_MNIST(string path,int no_of_images, int dimension, int rank, int world_size);
        vector <vector<VALUE_TYPE>>read_mnist_labels(string path, int no_of_images, int dimension, int rank, int world_size);
        vector<vector<VALUE_TYPE>> read_File(string path,int no_of_data_points, int dimension, int rank, int world_size);
        vector<vector<VALUE_TYPE>> mpi_file_read(string path, int rank, int world_size, int overlap,int total_data_set_size, char delim, int dimension);
        vector<vector<VALUE_TYPE>> mpi_file_read(string path, int rank, int world_size, int overlap,int total_data_set_size, int data_node_byte, int dimension);
        int reverse_int(int i);
      };
}

#endif //DISTRIBUTED_MRPT_IMAGE_READER_H
