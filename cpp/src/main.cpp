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
#include <chrono>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <cstring>

using namespace std;
using namespace dmrpt;
using namespace std::chrono;

int main(int argc, char *argv[]) {

    string input_path = "";
    string output_path = "";
    int algo = 0;
    int data_set_size = 0;
    int dimension = 0;
    int ntrees = 10;
    int tree_depth = 0;
    int transfer_threshold = 10;
    int donate_per = 10;
    double density = 0;
    int nn = 0;
    int batch_size = 1000;

    for (int p = 0; p < argc; p++) {

        if (strcmp(argv[p], "-input") == 0) {
            input_path = argv[p + 1];
        } else if (strcmp(argv[p], "-output") == 0) {
            output_path = argv[p + 1];
        } else if (strcmp(argv[p], "-algo") == 0) {
            algo = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-data-set-size") == 0) {
            data_set_size = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-dimension") == 0) {
            dimension = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-ntrees") == 0) {
            ntrees = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-tree-depth") == 0) {
            tree_depth = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-transfer_threshold") == 0) {
            transfer_threshold = atof(argv[p + 1]);
        } else if (strcmp(argv[p], "-donate_per") == 0) {
            donate_per = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-density") == 0) {
            density = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-nn") == 0) {
            nn = atoi(argv[p + 1]);
        } else if (strcmp(argv[p], "-batch_size") == 0) {
            batch_size = atoi(argv[p + 1]);
        }

    }

    if (input_path.size() == 0) {
        printf("Valid input path needed!...\n");
        exit(1);
    }
    if (output_path.size() == 0) {
        printf("Valid out path needed!...\n");
        exit(1);
    }
    if (data_set_size == 0) {
        printf("Dataset size should be greater than 0\n");
        exit(1);
    }

    if (dimension == 0) {
        printf("Dimension size should be greater than 0\n");
        exit(1);
    }

    if (nn == 0) {
        printf("Nearest neighbours size should be greater than 0\n");
        exit(1);
    }
    if (density == 0) {
        density = 1.0 / sqrt(dimension);
    }


    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (tree_depth == 0) {
        tree_depth = static_cast<int>(log2((data_set_size/size)) - 3);
    }


    ImageReader imageReader;

    char hostname[HOST_NAME_MAX];
    char stats[500];
    char results[500];

    int host = gethostname(hostname, HOST_NAME_MAX);

    string file_path_stat = output_path + "stats.txt.";
    std::strcpy(stats, file_path_stat.c_str());
    std::strcpy(stats+ strlen(file_path_stat.c_str()), hostname);

    string file_path = output_path + "results.txt.";
    std::strcpy(results, file_path.c_str());
    std::strcpy(results+ strlen(file_path.c_str()),hostname);

    ofstream fout(stats, std::ios_base::app);
    ofstream fout1(results, std::ios_base::app);


    auto start_io_index = high_resolution_clock::now();

    vector <vector<VALUE_TYPE>> imagedatas = imageReader.read_MNIST(
            input_path + "/train-images-idx3-ubyte", data_set_size, dimension,
            rank, size);



//    vector <vector<VALUE_TYPE>> labeldatas = imageReader.read_mnist_labels(
//            input_path + "/train-labels-idx1-ubyte", data_set_size, 1, rank,
//            size);

    auto stop_io_index = high_resolution_clock::now();
    auto io_time = duration_cast<microseconds>(stop_io_index - start_io_index);

    cout << "Rank " << rank << " Size of  images data " << imagedatas.size() << "*" << imagedatas[0].size() << endl;

    MathOp mathOp;

    int rows = imagedatas[0].size();
    int cols = imagedatas.size();



    int chunk_size = data_set_size / size;
    MDRPT mdrpt = MDRPT(ntrees, algo, imagedatas, tree_depth, data_set_size,
                        donate_per, transfer_threshold, dmrpt::StorageFormat::RAW, rank, size,input_path,output_path);
    auto start_index_buildling = high_resolution_clock::now();
    mdrpt.grow_trees(density);
    auto stop_index_building = high_resolution_clock::now();

    auto duration_index_building = duration_cast<microseconds>(stop_index_building - start_index_buildling);

    int co = 0;

    auto start_query = high_resolution_clock::now();
    vector <vector<DataPoint>> data_points;
    if (algo == 0) {
        cout << " starting batch query " << endl;
        data_points = mdrpt.batch_query(batch_size, 5000.0, nn);
        cout << "  batch querying completed " << endl;
    } else {
        data_points = mdrpt.get_knn(nn);
    }
    auto stop_query = high_resolution_clock::now();
    auto duration_query = duration_cast<microseconds>(stop_query - start_query);

    cout << "Time taken for total query "
         << duration_query.count() << " microseconds" << endl;

    if (fout.is_open()) {
        if (data_points.size() > 0) {
            for (int k = 0; k < data_points.size(); k++) {
                if (data_points[k].size() > 0) {
                    vector <DataPoint> vec = data_points[k];
                    for (int l = 0; l < 10; l++) {
                        if (vec[l].src_index != vec[l].index) {
                            if (algo == 0) {
                                fout1 << k + rank * chunk_size + 1 << ' ' << vec[l].index + 1 << endl;
                            } else {
                                fout1 << vec[l].src_index + 1 << ' ' << vec[l].index + 1 << endl;
                            }
                        }
                    }
                }
            }
        }
    }

    fout << rank << ' ' << io_time.count() << ' ' << duration_index_building.count() << ' ' << duration_query.count()
         << endl;
    MPI_Finalize();
}