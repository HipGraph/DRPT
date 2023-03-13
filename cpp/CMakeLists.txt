cmake_minimum_required(VERSION 3.17)
project(drpt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LocationByValue MPI
message("Finding MPI")
find_package(MPI REQUIRED)
message(STATUS "MPI include dir: ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "MPI libs: ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI executable: ${MPIEXEC_EXECUTABLE}")

message("Finding openBLAS")
find_package(OpenBLAS REQUIRED)
message(STATUS "OpenBLAS include dir: ${OpenBLAS_CXX_INCLUDE_PATH}")
message(STATUS "OpenBLAS libs: ${OpenBLAS_CXX_LIBRARIES}")

include_directories(${OpenBLAS_CXX_INCLUDE_PATH})

message("Finding OpenMP")
find_package(OpenMP REQUIRED)

message("CMAKE_BINARY_PATH ${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

include_directories(src/drpt/io)

set(SOURCES src/main.cpp)

add_executable(drpt ${SOURCES})

add_subdirectory(src/drpt)

target_link_libraries(drpt mdrpt)

target_link_libraries(drpt MPI::MPI_CXX)
target_link_libraries(drpt OpenMP::OpenMP_CXX)