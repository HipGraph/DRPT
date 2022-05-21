#include "image_reader.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

using namespace cv;
using namespace std;

vector<long> dmrpt::ImageReader::readImage(string path) {
    Mat image = imread(path, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Could not open or find the image " << path << endl;
    }
    vector<long> array;
    array.assign((long *) image.data, (long *) image.data + image.total() * image.channels());
    return array;
}

vector <vector<long>> dmrpt::ImageReader::readImages(vector <string> imagePaths) {
    vector <vector<long>> imagesdata;
    for (int i = 0; i < imagePaths.size(); i++) {
        imagesdata.push_back(readImage(imagePaths[i]));
    }
    return imagesdata;
}


