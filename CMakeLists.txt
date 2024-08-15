cmake_minimum_required(VERSION 3.10)
project(EdgeDetection VERSION 1.0)
enable_language(CUDA)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

add_executable(main src/main.cpp)

add_executable(sobel_seq
    src/gray_image.cpp
    src/sobel/sobel_seq.cpp
)
target_link_libraries(sobel_seq
    PRIVATE opencv_core
    PRIVATE opencv_highgui
    PRIVATE opencv_imgproc
)

add_executable(sobel_mpi
    src/gray_image.cpp
    src/sobel/sobel_mpi.cpp
)
target_link_libraries(sobel_mpi 
    PRIVATE opencv_core
    PRIVATE opencv_highgui
    PRIVATE opencv_imgproc
    PRIVATE MPI::MPI_CXX
)

add_executable(canny_seq
    src/gray_image.cpp
    src/canny/canny_seq.cpp
)
target_link_libraries(canny_seq
    PRIVATE opencv_core
    PRIVATE opencv_highgui
    PRIVATE opencv_imgproc
)

add_executable(canny_omp
    src/gray_image.cpp
    src/canny/canny_omp.cpp
)
target_link_libraries(canny_omp
    PRIVATE opencv_core
    PRIVATE opencv_highgui
    PRIVATE opencv_imgproc
    PRIVATE OpenMP::OpenMP_CXX
)

add_executable(canny_mpi
    src/gray_image.cpp
    src/canny/canny_mpi.cpp
)
target_link_libraries(canny_mpi
    PRIVATE opencv_core
    PRIVATE opencv_highgui
    PRIVATE opencv_imgproc
    PRIVATE MPI::MPI_CXX
)

add_executable(canny_cuda
    src/gray_image.cpp
    src/canny/canny_cuda.cu
)
target_link_libraries(canny_cuda
    PRIVATE opencv_core
    PRIVATE opencv_highgui
    PRIVATE opencv_imgproc
)
