
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
      };
}

#endif //DISTRIBUTED_MRPT_IMAGE_READER_H
