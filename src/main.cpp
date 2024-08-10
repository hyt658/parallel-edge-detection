#include <cstdlib>
#include <iostream>
#include <chrono>

void executeCMD(const char* cmd) {
    int result = system(cmd);
    if (result != 0) {
        std::cerr << "Execute [" << cmd 
            << "] failed with error code " << result << std::endl;
    }
}

int main() {
    executeCMD("./sobel_seq");
    executeCMD("mpirun -np 8 ./sobel_mpi");
}
