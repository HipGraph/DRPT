cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(
        dmrpt SHARED
        io/file_reader.cpp
        io/image_reader.cpp
        math/matrix_multiply.cpp
        algo/drpt.cpp
        algo/mdrpt.cpp
)
target_link_libraries(dmrpt ${OpenCV_LIBS})
target_link_libraries(dmrpt ${OpenBLAS_CXX_LIBRARIES})
target_link_libraries(dmrpt ${MPI_CXX_LIBRARIES})

