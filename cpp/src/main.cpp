#include "dmrpt/io/file_reader.hpp"
#include "dmrpt/io/image_reader.hpp"
#include "dmrpt/math/matrix_multiply.hpp"
#include <vector>
#include <mpi.h>
#include <string>
#include <omp.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace dmrpt;

int main(int argc, char *argv[]) {

    string folderPath = argv[1];

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    FileReader fileReader = FileReader(rank);
    vector <string> vec = fileReader.
            parseFileNames(folderPath);
    vector <string> splittedVales = fileReader.getMyFileNames(vec, size);
    ImageReader imageReader;
    vector <vector<double>> imagedatas = imageReader.readImages(splittedVales);
    cout << "Rank " << rank << " Size of  images data " << imagedatas.size() << "*" << imagedatas[0].size() << endl;

    MathOp mathOp;

    int rows = imagedatas[0].size();
    int cols = imagedatas.size();
    int tree_levels = 8;
    double *imdataArr = mathOp.convert_to_row_major_format(imagedatas);

    double *B = mathOp.build_sparse_projection_matrix(rank, size, rows, tree_levels, 0.8);
    // P= X.R
    double *P = mathOp.multiply_mat(imdataArr, B, rows, tree_levels, cols, 1.0);

    string ran = to_string(rank);

    string filename=  "/Users/isururanawaka/Documents/Master_IU_ISE_Courses/Summer_2022/distributed-mrpt/cpp/my_text"+ran+".txt";
    ofstream fout(filename);
    if(fout.is_open())
    {
        for(int k=0;k<cols;k++){
            for (int i = 1; i < 2; i++)
                fout << P[i+k*tree_levels] << ' ';
            fout<<endl;
        }
    }



//    double *array1 = (double *) malloc(sizeof(double) * 4);
//    double *array2 = (double *) malloc(sizeof(double) * 4);
//    double *array3 = (double *) malloc(sizeof(double) * 4);
//    double *array4 = (double *) malloc(sizeof(double) * 4);

//    double array1[6] = {1,3,10,20,14,25};
//    double array2[6] = {30,50,25,50,56,24};
//    double array3[6] = {10,25,50,34,45,25};
//    double array4[6] = {7,1,53,16,56,35};


    double *medians = mathOp.distributed_median(P, cols, tree_levels, cols * size, 28, dmrpt::StorageFormat::RAW, rank);
    for (int i = 0; i < tree_levels; ++i) {
        std::cout << "rank " << rank << " gmedian " << medians[i] << std::endl;
    }

    free(imdataArr);
    free(B);
    free(P);
    free(medians);

    MPI_Finalize();

}

