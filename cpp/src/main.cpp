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

int main(int argc, char *argv[]) {

    string folderPath = argv[1];

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

//    FileReader fileReader = FileReader(rank);
//    vector <string> vec = fileReader.
//            parseFileNames(folderPath);
//    vector <string> splittedVales = fileReader.getMyFileNames(vec, size);


    ImageReader imageReader;
//    vector <vector<double>> imagedatas = imageReader.readImages(splittedVales);

    vector<vector<double>> imagedatas =  imageReader.read_MNIST("/Users/isururanawaka/Documents/Master_IU_ISE_Courses/Summer_2022/train-images-idx3-ubyte",60000,784,rank,size);

    vector<vector<double>> labeldatas =  imageReader.read_mnist_labels("/Users/isururanawaka/Documents/Master_IU_ISE_Courses/Summer_2022/train-labels-idx1-ubyte",60000,1,rank,size);



    cout << "Rank " << rank << " Size of  images data " << imagedatas.size() << "*" << imagedatas[0].size() << endl;

    MathOp mathOp;

    int rows = imagedatas[0].size();
    int cols = imagedatas.size();
    int tree_levels = static_cast<int>(log2(cols) - 1);
//    tree_levels =  tree_levels -1;

//
//    double *imdataArr = mathOp.convert_to_row_major_format(imagedatas);
//
//    double *B = mathOp.build_sparse_projection_matrix(rank, size, rows, tree_levels, 0.9);
//    // P= X.R
//    double *P = mathOp.multiply_mat(imdataArr, B, rows, tree_levels, cols, 1.0);

    int chunk_size = 60000/size;

//    DRPT drpt = DRPT(P, cols, tree_levels, imagedatas,rank*chunk_size,dmrpt::StorageFormat::RAW);
//
//    drpt.grow_local_tree(rank);
//
      MDRPT mdrpt = MDRPT(50,imagedatas,tree_levels,dmrpt::StorageFormat::RAW,rank,size);
      mdrpt.grow_trees(1.0/ sqrt(rows));


    MPI_File fh;
    char filename[500];
    char labels[500];
    sprintf(filename, "/Users/isururanawaka/Documents/Master_IU_ISE_Courses/Summer_2022/distributed-mrpt/cpp/results.txt");
    sprintf(labels, "/Users/isururanawaka/Documents/Master_IU_ISE_Courses/Summer_2022/distributed-mrpt/cpp/labels.nodes.txt");
    MPI_File_open(MPI_COMM_SELF, filename,MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
    //FILE* f = fopen("test.txt","wb+");
    ofstream fout(filename,std::ios_base::app);
    ofstream fout2(labels,std::ios_base::app);
    int co=0;

//    vector<vector<double>> selected;
//    selected.push_back(imagedatas[0]);

//    for(int i=0;i<size;++i){
//
//            vector <vector<dmrpt::DRPT::DataPoint>> results = drpt.batchQuery(imagedatas, B, 10, dmrpt::StorageFormat::RAW, rank, i, size,500.0);

     vector <vector<dmrpt::DRPT::DataPoint>> results =  mdrpt.batchQuery(10,5000.0,10);
            if (fout.is_open()) {
                if (results.size() > 0) {
                    for (int k = 0; k < results.size(); k++) {
                        for (int l = 0; l < results[k].size(); l++) {
                            if (k + rank * cols != results[k][l].index) {
                                //                            char buf[42];
                                //fprintf(f,"%d \n",i);
                                //                            snprintf(buf,42,"%d %d\n",k + rank * cols,results[k][l]);
                                //                            MPI_File_write(fh,buf,strlen(buf), MPI_CHAR,MPI_STATUS_IGNORE);
                                fout << (k + rank * chunk_size) + 1 << ' ' << results[k][l].index + 1<< ' ' << results[k][l].distance<< endl;
                                //                        }
                            }
                        }

                    }
                    MPI_File_close(&fh);
                }
            }
           results.clear();


//    for(int j=0;j<labeldatas.size();++j){
//        fout2 << (j + rank * chunk_size)+1 << ' ' << (int)labeldatas[j][0] << endl;
//    }

//
//    free(imdataArr);
//    free(B);
//    free(P);
//    free(medians);

    MPI_Finalize();

}

