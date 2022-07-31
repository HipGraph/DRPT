#include "image_reader.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
//#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs/imgcodecs.hpp>

//using namespace cv;
using namespace std;

//vector<VALUE_TYPE> dmrpt::ImageReader::readImage(string path) {
//    Mat image = imread(path, IMREAD_GRAYSCALE);
//    if (image.empty()) {
//        cout << "Could not open or find the image " << path << endl;
//    }
//    assert(image.channels() == 1);
//    vector<VALUE_TYPE> array;
//    for (int r = 0; r < image.rows; r++) {
//        for (int c = 0; c < image.cols; c++) {
//            VALUE_TYPE val = (VALUE_TYPE) image.at<u_char>(r, c);
//            array.push_back(val);
//        }
//    }
//    return array;
//}

//vector <vector<VALUE_TYPE>> dmrpt::ImageReader::readImages(vector <string> imagePaths) {
//    vector <vector<VALUE_TYPE>> imagesdata;
//    for (int i = 0; i < imagePaths.size(); i++) {
//        imagesdata.push_back(readImage(imagePaths[i]));
//    }
//    return imagesdata;
//}

int dmrpt::ImageReader::reverse_int(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

vector <vector<VALUE_TYPE>>
dmrpt::ImageReader::read_MNIST(string path, int no_of_images, int dimension, int rank, int world_size) {
    vector <vector<VALUE_TYPE>> arr;

    ifstream file(path, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *) &magic_number, sizeof(magic_number));
        magic_number = this->reverse_int(magic_number);
        file.read((char *) &number_of_images, sizeof(number_of_images));
        number_of_images = this->reverse_int(number_of_images);
        file.read((char *) &n_rows, sizeof(n_rows));
        n_rows = this->reverse_int(n_rows);
        file.read((char *) &n_cols, sizeof(n_cols));
        n_cols = this->reverse_int(n_cols);

        int chunk_size = number_of_images / world_size;
        if (rank < world_size - 1) {
            arr.resize(chunk_size, vector<VALUE_TYPE>(dimension));
        } else if (rank == world_size - 1) {
            chunk_size = no_of_images - chunk_size * (world_size - 1);
            arr.resize(chunk_size, vector<VALUE_TYPE>(dimension));
        }

        for (int i = 0; i < number_of_images; ++i) {
            for (int r = 0; r < n_rows; ++r) {
                for (int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char *) &temp, sizeof(temp));
                    if (i >= rank * chunk_size and i < (rank + 1) * chunk_size and rank < world_size - 1) {
                        arr[i - rank * chunk_size][(n_rows * r) + c] = (VALUE_TYPE) temp;
                    } else if (rank == world_size - 1 && i >= (rank) * chunk_size) {
                        arr[i - rank * chunk_size][(n_rows * r) + c] = (VALUE_TYPE) temp;
                    }
                }
            }
        }
    }
    return arr;
}

vector <vector<VALUE_TYPE>> dmrpt::ImageReader::read_mnist_labels(string path, int no_of_images, int dimension, int rank, int world_size) {

    vector <vector<VALUE_TYPE>> arr;

    ifstream file(path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        int number_of_labels = 0;
        file.read((char *) &magic_number, sizeof(magic_number));
        magic_number = this->reverse_int(magic_number);

        if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *) &number_of_labels, sizeof(number_of_labels)),
                number_of_labels = this->reverse_int(number_of_labels);

        int chunk_size = number_of_labels / world_size;
        if (rank < world_size - 1) {
            arr.resize(chunk_size, vector<VALUE_TYPE>(dimension));
        } else if (rank == world_size - 1) {
            chunk_size = no_of_images - chunk_size * (world_size - 1);
            arr.resize(chunk_size, vector<VALUE_TYPE>(dimension));
        }


        for (int i = 0; i < number_of_labels; ++i) {
            unsigned char temp = 0;
            file.read((char *) &temp, sizeof(temp));
            if (i >= rank * chunk_size and i < (rank + 1) * chunk_size and rank < world_size - 1) {
                arr[i - rank * chunk_size][0] = (VALUE_TYPE) temp;
            } else if (rank == world_size - 1 && i >= (rank) * chunk_size) {
                arr[i - rank * chunk_size][0] = (VALUE_TYPE) temp;
            }
        }
        return arr;
    } else {
        throw runtime_error("Unable to open file `" + path + "`!");
    }
}


