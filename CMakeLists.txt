cmake_minimum_required(VERSION 3.10)
project(EdgeDetection VERSION 1.0)

find_package(PNG REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_executable(main 
    main.cpp 
    gray_image.cpp
)

target_link_libraries(main 
    PRIVATE PNG::PNG
)
