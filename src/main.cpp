#include <iostream>
#include <cstdlib>
#include <string>

void executeCMD(std::string cmd, bool verbose) {
    if (verbose) {
        cmd += " -v";
    }

    int result = system(cmd.c_str());
    if (result != 0) {
        std::cerr << "Execute [" << cmd 
            << "] failed with error code " << result << std::endl;
    }
}

int main(int argc, char* argv[]) {
    bool verbose = false;
    if (argc > 1) {
        auto arg1 = std::string(argv[1]);
        if (arg1 == "-v" || arg1 == "--verbose") {
            verbose = true;
        }
    }

    for(int i = 0; i < 10; i++) {
    executeCMD("./sobel_seq", verbose);
    executeCMD("./sobel_omp", verbose);
    executeCMD("mpirun -np 6 ./sobel_mpi", verbose);
    executeCMD("./sobel_cuda", verbose);

    executeCMD("./canny_seq", verbose);
    executeCMD("./canny_omp", verbose);
    executeCMD("mpirun -np 6 ./canny_mpi", verbose);
    executeCMD("./canny_cuda", verbose);
    }
}
