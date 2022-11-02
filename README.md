# varRCWA


## Examples

some example codes are in src/tests

- test_cpu.cpp: simple examples on CPU (here I write a few comments, you can start here)
- test_gpu.cpp: simple examples on GPU
- test_gds_cpu.cpp: GDSII examples on CPU
- test_gds_gpu.cpp: GDSII examples on GPU
- test_metasurface_cpu.cpp: metasurface (periodic boundary) examples on CPU
- test_metasurface_gpu.cpp: metasurface (periodic boundary) examples on CPU

Some other tests
- test_cpu_differential_method.cpp: test the stability of differential method
- test_oe_model_gpu.cpp: test the optics express mode converter on GPU
- the rest are for development only, please overlook them

- The main algorithm of varRCWA is located in src/core/RedhefferIntegrator.cpp
- The main algorithm of RCWA is located in src/core/RCWAIntegrator.cpp
- The main algorithm of the differential method is located in src/core/DifferentialIntegrator.cpp
- The GPU version of the above algorithms are in src/core/XXXXIntegratorGPU.cpp

## Build and Dependencies

We use [Cmake](https://cmake.org/download/) 3.25.0 and Make (the standard build system on Linux(Ubuntu)) for building the library. 

A tutorial on CMake is available [here](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)

There are some options in CMakeLists.txt
```
option(WITH_OPENMP "enable OpenMP acceleration or not" ON)
option(BUILD_DEBUG "Turn on the debug mode" OFF)
option(BUILD_TESTS "Build unit test cases" ON)
option(BUILD_GPU "Build GPU examples" OFF)
```

Here I set the BUILD_GPU to be off, you can turn it on if CUDA and Magma is correctly installed. It takes some time but will be worthwhile!

### Eigen (header only)

Download from [here](https://eigen.tuxfamily.org/index.php?title=Main_Page)

### OpenMP

```sudo apt-get install libomp-dev```


### Boost (header only)

```sudo apt-get install libboost-all-dev```

### Intel OneAPI (MKL, TBB)

Download from [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)


### Magma (Optional for GPU code)

Download from [here](https://bitbucket.org/icl/magma/src/master/)

A tutorial is [here](https://rgb.sh/blog/magma)

### CUDA 11.4 (Optional for GPU code)

Download from [here](https://developer.nvidia.com/cuda-downloads)

### G++-10
tutorial [here](https://askubuntu.com/questions/1192955/how-to-install-g-10-on-ubuntu-18-04)


To build

For CPU
```
cmake -D CMAKE_CUDA_HOST_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ .
make -j8
```


For GPU (use CUDA 11.4 please!)

```
cmake -D CMAKE_CUDA_HOST_COMPILER=gcc -D CMAKE_CUDA_COMPILER=nvcc -D CMAKE_CXX_COMPILER=g++ .
make -j8
```

## Citation

```
@article{zhu2022VarRCWA,
  author = {Zhu, Ziwei and Zheng, Changxi},
  title = {VarRCWA: An Adaptive High-Order Rigorous Coupled Wave Analysis Method},
  journal = {ACS Photonics},
  volume = {9},
  number = {10},
  pages = {3310-3317},
  year = {2022},
  doi = {10.1021/acsphotonics.2c00662},
  URL = {https://doi.org/10.1021/acsphotonics.2c00662},
  eprint = {https://doi.org/10.1021/acsphotonics.2c00662}
}
```