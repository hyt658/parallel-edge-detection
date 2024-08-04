cmake_minimum_required(VERSION 3.10)
project(EdgeDetection VERSION 1.0)

find_package(PNG REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_executable(main src/main.cpp)

add_executable(sobel_seq
    src/gray_image.cpp
    src/sobel/sobel_seq.cpp
)

add_executable(sobel_mpi
    src/gray_image.cpp
    src/sobel/sobel_mpi.cpp
)

target_link_libraries(sobel_seq
    PRIVATE PNG::PNG
)

target_link_libraries(sobel_mpi 
    PRIVATE PNG::PNG
    PRIVATE MPI::MPI_CXX
)
