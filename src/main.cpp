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

    executeCMD("./sobel_seq", verbose);
    executeCMD("mpirun -np 6 ./sobel_mpi", verbose);
}
