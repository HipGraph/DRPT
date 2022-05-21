#include "dmrpt/io/file_reader.hpp"
#include "dmrpt/io/image_reader.hpp"
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
    vector <vector<long>> imagedatas = imageReader.readImages(splittedVales);
    cout << "Rank " << rank << " Size of  imagesdata " << imagedatas.size() << "*" << imagedatas[0].size() << endl;

}

