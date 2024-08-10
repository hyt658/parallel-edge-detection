# Comparative Analysis of Edge Detection Using Different Parallel Computing Techniques

## What is this

Using OpenMP, MPI, and CUDA to implement edge detection in C++. Analyze their results and performance.

## Building

### Library Requirements

1. libomp
2. open-mpi
3. CUDA Toolkit
4. OpenCV

### Build Step

```
mkdir build
cd build
cmake ..
make
```

Each parallel technique will have a separate executable file. `main` will execute all of them and record the running time. All of them will be in the `build/` directory.
