#include <cstdlib>
#include <chrono>

int main() {
    // int result = system("./sobel_seq");
    int result = system("mpirun -np 4 ./sobel_mpi");
}
