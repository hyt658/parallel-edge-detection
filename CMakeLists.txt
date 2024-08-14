cmake_minimum_required(VERSION 3.10)
project(EdgeDetection VERSION 1.0)

find_package(MPI REQUIRED)
find_package(OpenCV REQUIRED)

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

add_executable(canny_seq
    src/gray_image.cpp
    src/canny/canny_seq.cpp
)

add_executable(canny_mpi
    src/gray_image.cpp
    src/canny/canny_mpi.cpp
)

target_link_libraries(main
    PRIVATE opencv_core
    PRIVATE opencv_highgui
    PRIVATE opencv_imgproc
)

target_link_libraries(sobel_seq
    PRIVATE opencv_core
    PRIVATE opencv_highgui
    PRIVATE opencv_imgproc
)

target_link_libraries(sobel_mpi 
    PRIVATE opencv_core
    PRIVATE opencv_highgui
    PRIVATE opencv_imgproc
    PRIVATE MPI::MPI_CXX
)

target_link_libraries(canny_seq
    PRIVATE opencv_core
    PRIVATE opencv_highgui
    PRIVATE opencv_imgproc
)

target_link_libraries(canny_mpi
    PRIVATE opencv_core
    PRIVATE opencv_highgui
    PRIVATE opencv_imgproc
    PRIVATE MPI::MPI_CXX
)
