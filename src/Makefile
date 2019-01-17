KOKKOS_PATH = ${HOME}/build/install/kokkos
CABANA_PATH = ${HOME}/build/install/cabana
KOKKOS_LIB = ${KOKKOS_PATH}/lib
KOKKOS_INCLUDE = ${KOKKOS_PATH}/include
CABANA_LIB = ${CABANA_PATH}/lib64
CABANA_INCLUDE = ${CABANA_PATH}/include

#MPI On or off (1/0)
MPI = 0

MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CABANAMD_PATH = $(subst Makefile,,$(MAKEFILE_PATH))/..

default: build
	echo "Start Build"


EXE = CabanaMD
CXXFLAGS = -O3 -g
LINKFLAGS = -O3 -g

CXXFLAGS += -I${KOKKOS_INCLUDE}
LDFLAGS = -L${KOKKOS_LIB} -lkokkos

#CXXFLAGS += `pkg-config --cflags Cabana`
CXXFLAGS += -I${CABANA_INCLUDE} --std=c++11 -fopenmp
#LDFLAGS = `pkg-config --libs Cabana`
LDFLAGS += -L${CABANA_LIB} -lcabanacore -ldl -fopenmp

ifeq ($(MPI), 1)
  CXX = mpicxx
  CXXFLAGS += -DCABANAMD_ENABLE_MPI
else
  ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
    CXX = $(KOKKOS_PATH)/bin/nvcc_wrapper
  else
    CXX = g++
  endif
endif

LINK = ${CXX}

KOKKOS_OPTIONS=aggressive_vectorization

DEPFLAGS = -M

vpath %.cpp $(CABANAMD_PATH)/src
SRC =  $(wildcard $(CABANAMD_PATH)/src/*.cpp)
HEADERS = $(wildcard $(CABANAMD_PATH)/src/*.h)
CXXFLAGS += -I$(CABANAMD_PATH)/src

SRC_NOTDIR = $(notdir $(SRC))
OBJ = $(SRC_NOTDIR:.cpp=.o)
LIB =


build: $(EXE)

$(EXE): $(OBJ)
	$(LINK) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(LDFLAGS) $(LIB) -o $(EXE)

clean:
	rm -f *.o $(EXE)

# Compilation rules

%.o:%.cpp ${HEADERS}
	$(CXX) $(CXXFLAGS) $(EXTRA_INC) -c $<

