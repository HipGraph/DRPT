
#ifndef DISTRIBUTED_MRPT_IMAGE_READER_H
#define DISTRIBUTED_MRPT_IMAGE_READER_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
namespace dmrpt{
    class ImageReader {
      public:
        vector<double> readImage(string path);
        vector<vector<double>> readImages(vector<string> imagePaths);
        vector<vector<double>> read_MNIST(string path,int no_of_images, int dimension, int rank, int world_size);
        vector <vector<double>>read_mnist_labels(string path, int no_of_images, int dimension, int rank, int world_size);
        int reverse_int(int i);
      };
}

#endif //DISTRIBUTED_MRPT_IMAGE_READER_H
