#include "dmrpt/io/file_reader.hpp"
#include "dmrpt/io/image_reader.hpp"
#include "dmrpt/math/matrix_multiply.hpp"
#include <vector>
#include <mpi.h>
#include <string>
#include <omp.h>

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
    int tree_levels=8;
    double * imdataArr = mathOp.convert_to_row_major_format(imagedatas);

    double * B =  mathOp.build_sparse_projection_matrix(rank,size,rows,tree_levels,0.8);
    // P= X.R
    double *P = mathOp.multiply_mat(imdataArr,B,rows,tree_levels,cols,1.0);

    printf("Rank %d",rank);
    for(int k=0;k<rows;k++){
        for (int i = 0; i < cols; i++)
            printf("%lf ", imdataArr[i+k*cols]);
        printf("\n");
    }

    free(imdataArr);
    free(B);
//    free(P);
    MPI_Finalize();

}

