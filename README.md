# cabanaMD

is based on ExaMiniMD, modified to incrementally replace features with
the CoPA Cabana Particle Toolkit



## ExaMiniMD

ExaMiniMD is a proxy application and research vehicle for 
particle codes, in particular Molecular Dynamics (MD): 
https://github.com/ECP-copa/ExaMiniMD



# Build instructions

CabanaMD has the following dependencies:

|Dependency | Version | Required | Details|
|---------- | ------- |--------  |------- |
|CMake      | 3.9+    | Yes      | Build system
|Kokkos     | 2.7.0   | Yes      | Provides portable on-node parallelism
|Cabana     | 0.1     | Yes      | Performance portable particle algorithms


Build Kokkos, followed by Cabana:
https://github.com/ECP-copa/Cabana/wiki/Build-Instructions

Build CabanaMD just as Cabana (using the same default build directories):
```
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:~/build/install/cabana/lib64/pkgconfig

cd
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
    -DCMAKE_INSTALL_PREFIX=$HOME \
    \
    .. ;
make install
```