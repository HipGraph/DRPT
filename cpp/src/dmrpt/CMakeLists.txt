cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(
        dmrpt SHARED
        #        io/file_reader.cpp
        io/file_reader.cpp
        math/math_operations.cpp
        algo/drpt_local.cpp
        algo/mdrpt.cpp
        algo/drpt_global.cpp
        io/file_writer.cpp
)
target_link_libraries(dmrpt ${OpenCV_LIBS})
target_link_libraries(dmrpt ${OpenBLAS_CXX_LIBRARIES})
target_link_libraries(dmrpt ${MPI_CXX_LIBRARIES})

