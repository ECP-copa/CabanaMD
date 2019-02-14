# CabanaMD

is based on ExaMiniMD, modified to replace features with
the CoPA Cabana Particle Toolkit:
https://github.com/ECP-copa/Cabana


### ExaMiniMD

ExaMiniMD is a proxy application and research vehicle for 
particle codes, in particular Molecular Dynamics (MD): 
https://github.com/ECP-copa/ExaMiniMD



# Build instructions
The following shows how to configure and build CabanaMD.

## Dependencies
CabanaMD has the following dependencies:

|Dependency | Version | Required | Details|
|---------- | ------- |--------  |------- |
|CMake      | 3.9+    | Yes      | Build system
|Kokkos     | 2.7.0   | Yes      | Provides portable on-node parallelism
|Cabana     | 0.1     | Yes      | Performance portable particle algorithms


Build Kokkos, followed by Cabana:
https://github.com/ECP-copa/Cabana/wiki/Build-Instructions

Build instructions are available for both CPU and GPU

## CPU Build
Build CabanaMD just like Cabana (using the same default build directories):
```
cd
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:`pwd`/build/install/cabana/lib64/pkgconfig
export KOKKOS_SRC_DIR=`pwd`/kokkos
export KOKKOS_INSTALL_DIR=`pwd`/build/install/kokkos

cd ./CabanaMD
mkdir build
cd build
pwd
cmake \
    -D KOKKOS_SETTINGS_DIR=$KOKKOS_INSTALL_DIR \
    -D KOKKOS_LIBRARY=$KOKKOS_INSTALL_DIR/lib/libkokkos.a \
    -D KOKKOS_INCLUDE_DIR=$KOKKOS_INSTALL_DIR/include \
    -D Cabana_ENABLE_Serial=ON \
    -D Cabana_ENABLE_OpenMP=ON \
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
    -D KOKKOS_SETTINGS_DIR=$KOKKOS_INSTALL_DIR \
    -D KOKKOS_LIBRARY=$KOKKOS_INSTALL_DIR/lib/libkokkos.a \
    -D KOKKOS_INCLUDE_DIR=$KOKKOS_INSTALL_DIR/include \
    -D CMAKE_CXX_COMPILER=$KOKKOS_SRC_DIR/bin/nvcc_wrapper \
    -D Cabana_ENABLE_Serial=ON \
    -D Cabana_ENABLE_Cuda:BOOL=ON \
    \
    .. ;
```
