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

vector<double> dmrpt::ImageReader::readImage(string path) {
    Mat image = imread(path, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Could not open or find the image " << path << endl;
    }
    assert(image.channels()==1);
    vector<double> array;
    for(int r=0; r<image.rows;r++){
        for(int c=0;c<image.cols;c++){
            double val = (double )image.at<u_char>(r,c);
            array.push_back(val);
        }
    }
    return array;
}

vector <vector<double>> dmrpt::ImageReader::readImages(vector <string> imagePaths) {
    vector <vector<double>> imagesdata;
    for (int i = 0; i < imagePaths.size(); i++) {
        imagesdata.push_back(readImage(imagePaths[i]));
    }
    return imagesdata;
}


