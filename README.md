# CabanaMD

is a proxy application for molecular dynamics (MD) based 
on ExaMiniMD, modified to replace features with the CoPA 
Cabana Particle Toolkit:
https://github.com/ECP-copa/Cabana


ExaMiniMD is a proxy app and research vehicle for 
MD using the Kokkos performance portability library
("KokkosMD"):
https://github.com/ECP-copa/ExaMiniMD
https://github.com/kokkos/kokkos


# Build instructions
The following shows how to configure and build CabanaMD.

## Dependencies
CabanaMD has the following dependencies:

|Dependency | Version | Required | Details|
|---------- | ------- |--------  |------- |
|CMake      | 3.9+    | Yes      | Build system
|MPI        | GPU Aware if CUDA Enabled | Yes | Message Passing Interface
|Kokkos     | 3.0     | Yes      | Provides portable on-node parallelism
|Cabana     | master  | Yes      | Performance portable particle algorithms
|ArborX     | master  | No       | Performance portable geometric search
|libnnp     | 1.0.0   | No       | Neural network potential utilities

Build Kokkos, followed by Cabana:
https://github.com/ECP-copa/Cabana/wiki/Build-Instructions

Build instructions are available for both CPU and GPU. Note that Cabana with
MPI is required (`-D Cabana_ENABLE_MPI=ON`)

## CPU Build
After building Kokkos and Cabana for CPU:
```
# Change directories as needed
export KOKKOS_DIR=$HOME/kokkos
export CABANA_DIR=$HOME/Cabana

cd ./CabanaMD
mkdir build
cd build
pwd
cmake \
    -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$CABANA_DIR" \
    -D CabanaMD_ENABLE_Serial=OFF \
    -D CabanaMD_ENABLE_OpenMP=ON \
    -D CabanaMD_ENABLE_Cuda=OFF \
    -D CabanaMD_VECTORLENGTH=1 \
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
    -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$CABANA_DIR" \
    -D CabanaMD_ENABLE_Serial=OFF \
    -D CabanaMD_ENABLE_OpenMP=OFF \
    -D CabanaMD_ENABLE_Cuda=ON \
    -D CabanaMD_VECTORLENGTH=32 \
    \
    .. ;
```

## Neural network potential build
If using the optional neural network potential, additional CMake flags for
the location of the libnnp library (https://github.com/CompPhysVienna/n2p2)
and enabling the potential are needed (with one additional optional vector
length setting):
```
    -D N2P2_DIR=$N2P2_DIR \
    -D CabanaMD_ENABLE_NNP=ON \
    -D CabanaMD_VECTORLENGTH_NNP=1 \
```

## ArborX neighbor list build
If using the optional ArborX library (https://github.com/arborx/ArborX),
changes to the Cabana CMake flags are needed:
```
    -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$ARBORX_DIR"
    -D Cabana_ENABLE_ARBORX=ON
```
