# CabanaMD

is a proxy application for molecular dynamics (MD) based 
on ExaMiniMD, modified to replace features with the CoPA 
Cabana Particle Toolkit:
https://github.com/ECP-copa/Cabana


ExaMiniMD is a proxy app and research vehicle for 
MD using Kokkos:
https://github.com/ECP-copa/ExaMiniMD



# Build instructions
The following shows how to configure and build CabanaMD.

## Dependencies
CabanaMD has the following dependencies:

|Dependency | Version | Required | Details|
|---------- | ------- |--------  |------- |
|CMake      | 3.9+    | Yes      | Build system
|MPI        | GPU Aware if CUDA Enabled | Yes | Message Passing Interface
|Kokkos     | 2.7.0   | Yes      | Provides portable on-node parallelism
|Cabana     | 0.3-dev | Yes      | Performance portable particle algorithms
|libnnp     | 1.0.0   | No       | Neural network potential utilities

Build Kokkos, followed by Cabana:
https://github.com/ECP-copa/Cabana/wiki/Build-Instructions

Build instructions are available for both CPU and GPU. Note that Cabana with
MPI is required (`-D Cabana_ENABLE_MPI=ON`)

## CPU Build
After building Kokkos and Cabana for CPU:
```
# Change directories as needed
export KOKKOS_INSTALL_DIR=$HOME/install/kokkos
export CABANA_INSTALL_DIR=$HOME/install/cabana

cd ./CabanaMD
mkdir build
cd build
pwd
cmake \
    -D CMAKE_PREFIX_PATH="$HOME/Cabana;$HOME/kokkos" \
    -D CabanaMD_ENABLE_Serial=OFF \
    -D CabanaMD_ENABLE_OpenMP=ON \
    -D CabanaMD_ENABLE_Cuda=OFF \
    \
    .. ;
make install
cd ../../
```

## GPU Build
After building Kokkos and Cabana for GPU:
https://github.com/ECP-copa/Cabana/wiki/Build-Instructions#GPU-Build

the GPU build is identical to that above except the options passed to CMake:
```
cmake \
    -D CMAKE_CXX_COMPILER=$KOKKOS_SRC_DIR/bin/nvcc_wrapper \
    -D CMAKE_PREFIX_PATH="$HOME/Cabana;$HOME/kokkos" \
    -D CabanaMD_ENABLE_Serial=OFF \
    -D CabanaMD_ENABLE_OpenMP=OFF \
    -D CabanaMD_ENABLE_Cuda=ON \
    \
    .. ;
```

## Neural network potential build
If using the optional neural network potential, additional flags for the
location of the libnnp library (https://github.com/CompPhysVienna/n2p2)
and enabling the potential are needed for CMake:
```
    -D N2P2_DIR=$N2P2_INSTALL_DIR \
    -D CabanaMD_ENABLE_NNP=ON \
```
