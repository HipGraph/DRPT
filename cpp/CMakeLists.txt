cmake_minimum_required(VERSION 3.21)
project(distributed_mrpt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_COMPILER  "/usr/local/opt/llvm/bin/clang++")
set(CMAKE_C_COMPILER "/usr/local/opt/llvm/bin/clang")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/opt/llvm/include -fopenmp  -I/usr/local/opt/openblas/include")
set(CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS} -L/usr/local/opt/llvm/lib -L/usr/local/opt/openblas/lib")
set(CMAKE_MODULE_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS} -L/usr/local/opt/llvm/lib -L/usr/local/opt/openblas/lib")
set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS} -L/usr/local/opt/llvm/lib -L/usr/local/opt/openblas/lib")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS} -L/usr/local/opt/llvm/lib -L/usr/local/opt/openblas/lib")
SET (CMAKE_AR      "/usr/local/opt/llvm/bin/llvm-ar")
set(OpenCV_DIR /Users/isururanawaka/Documents/Master_IU_ISE_Courses/Summer_2022/opencv/install/lib/cmake/opencv4)



# LocationByValue MPI
message("Finding MPI")
find_package(MPI REQUIRED COMPONENTS CXX)
message(STATUS "MPI include dir: ${MPI_CXX_INCLUDE_PATH}")
message(STATUS "MPI libs: ${MPI_CXX_LIBRARIES}")
message(STATUS "MPI executable: ${MPIEXEC_EXECUTABLE}")

include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

message("Finding OpenCV")
find_package( OpenCV REQUIRED )

message(STATUS "OPENCV include  dir: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OPENCV libs  dir: ${OpenCV_LIBS}")
include_directories( ${OpenCV_INCLUDE_DIRS} )


set(OpenBLAS_DIR "/usr/local/opt/openblas/lib/cmake/openblas")
message("Finding openBLAS")
find_package(OpenBLAS REQUIRED)
set(OpenBLAS_CXX_INCLUDE_PATH "/usr/local/opt/openblas/include")
set(OpenBLAS_CXX_LIBRARIES "/usr/local/opt/openblas/lib/libopenblas.dylib")
message(STATUS "OpenBLAS include dir: ${OpenBLAS_CXX_INCLUDE_PATH}")
message(STATUS "OpenBLAS libs: ${OpenBLAS_CXX_LIBRARIES}")

include_directories( ${OpenBLAS_CXX_INCLUDE_PATH})



#try_compile(HAS_FS "${CMAKE_BINARY_DIR}/temp"
#        "${CMAKE_SOURCE_DIR}/src/main.cpp"
#        CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
#        LINK_LIBRARIES stdc++fs)
#if(HAS_FS)
#    message(STATUS "Compiler has filesystem support")
#else()
#    #   .... You could also try searching for boost::filesystem here.
#    message(FATAL_ERROR "Compiler is missing filesystem capabilities")
#endif(HAS_FS)
message("CMAKE_BINARY_PATH ${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

include_directories(src/dmrpt/io)

set(SOURCES src/main.cpp)

add_executable(distributed_mrpt ${SOURCES})

add_subdirectory(src/dmrpt)

#target_link_libraries(distributed_mrpt stdc++fs)

target_link_libraries(distributed_mrpt dmrpt)
target_link_libraries(distributed_mrpt ${MPI_CXX_LIBRARIES})

