#include "dmrpt/io/file_reader.hpp"
#include "dmrpt/io/image_reader.hpp"
#include "dmrpt/math/matrix_multiply.hpp"
#include "dmrpt/algo/drpt.hpp"
#include "dmrpt/algo/mdrpt.hpp"
#include <vector>
#include <mpi.h>
#include <string>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <math.h>



using namespace std;
using namespace dmrpt;
using namespace std::chrono;

int main(int argc, char *argv[]) {

    string folderPath = argv[1];

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    ImageReader imageReader;

    vector<vector<VALUE_TYPE>> imagedatas =  imageReader.read_MNIST("/Users/isururanawaka/Documents/Master_IU_ISE_Courses/Summer_2022/train-images-idx3-ubyte",60000,784,rank,size);

    vector<vector<VALUE_TYPE>> labeldatas =  imageReader.read_mnist_labels("/Users/isururanawaka/Documents/Master_IU_ISE_Courses/Summer_2022/train-labels-idx1-ubyte",60000,1,rank,size);


    cout << "Rank " << rank << " Size of  images data " << imagedatas.size() << "*" << imagedatas[0].size() << endl;

    MathOp mathOp;

    int rows = imagedatas[0].size();
    int cols = imagedatas.size();
    int tree_levels = static_cast<int>(log2(cols)-2);


    int chunk_size = 60000/size;

    MDRPT mdrpt = MDRPT(10,imagedatas,tree_levels,60000,dmrpt::StorageFormat::RAW,rank,size);
    auto start = high_resolution_clock::now();
    mdrpt.grow_trees(1.0/ sqrt(rows));
//    mdrpt.grow_trees(0.9);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by grow trees: "
         << duration.count() << " microseconds" << endl;

    char filename[500];
    char labels[500];
    sprintf(filename, "/Users/isururanawaka/Documents/Master_IU_ISE_Courses/Summer_2022/distributed-mrpt/cpp/results.txt");
    sprintf(labels, "/Users/isururanawaka/Documents/Master_IU_ISE_Courses/Summer_2022/distributed-mrpt/cpp/labels.nodes.txt");

    //FILE* f = fopen("test.txt","wb+");
    ofstream fout(filename,std::ios_base::app);
    ofstream fout2(labels,std::ios_base::app);
    int co=0;

    start = high_resolution_clock::now();
     vector<vector<dmrpt::DRPT::DataPoint>> results =  mdrpt.batch_query(1000,5000.0,3,10);
     stop = high_resolution_clock::now();
     duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken for total query "
         << duration.count() << " microseconds" << endl;

    if (fout.is_open()) {
                if (results.size() > 0) {
                    for (int k = 0; k < results.size(); k++) {
                        for (int l = 0; l < results[k].size(); l++) {
                         if (k + rank * cols != results[k][l].index) {

                                fout << (k + rank * chunk_size) + 1 << ' ' << results[k][l].index + 1<< endl;
                         }
                        }
                    }
                }

            }

     results.clear();

     MPI_Finalize();

}

