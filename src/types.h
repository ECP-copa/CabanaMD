/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//************************************************************************

#ifndef TYPES_H
#define TYPES_H

#include <CabanaMD_config.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

// Module Types etc
// Units to be used
enum
{
    UNITS_REAL,
    UNITS_LJ,
    UNITS_METAL
};
// Lattice Type
enum
{
    LATTICE_SC,
    LATTICE_FCC
};
// Integrator Type
enum
{
    INTEGRATOR_NVE
};
// Binning Type
enum
{
    BINNING_LINKEDCELL
};
// Comm Type
enum
{
    COMM_MPI
};
// Force Type
enum
{
    FORCE_LJ,
    FORCE_SNAP,
    FORCE_NNP
};
// Force Iteration Type
enum
{
    FORCE_ITER_NEIGH_FULL,
    FORCE_ITER_NEIGH_HALF
};
// Force neighbor parallel Type
enum
{
    FORCE_PARALLEL_NEIGH_SERIAL,
    FORCE_PARALLEL_NEIGH_TEAM,
    FORCE_PARALLEL_NEIGH_VECTOR
};
// Neighbor Type
enum
{
    NEIGH_2D,
    NEIGH_CSR
};
// Input File Type
enum
{
    INPUT_LAMMPS
};

// Macros to work around the fact that std::max/min is not available on GPUs
#define MAX( a, b ) ( a > b ? a : b )
#define MIN( a, b ) ( a < b ? a : b )

#define MAX_TYPES_STACKPARAMS 12

// Define Scalar Types
#ifndef T_INT
#define T_INT int
#endif

#ifndef T_FLOAT
#define T_FLOAT double
#endif
#ifndef T_X_FLOAT
#define T_X_FLOAT T_FLOAT
#endif
#ifndef T_V_FLOAT
#define T_V_FLOAT T_FLOAT
#endif
#ifndef T_F_FLOAT
#define T_F_FLOAT T_FLOAT
#endif

typedef Kokkos::View<T_V_FLOAT *> t_mass;             // Mass
typedef Kokkos::View<const T_V_FLOAT *> t_mass_const; // Mass

// Cabana

#ifdef CabanaMD_ENABLE_Cuda
using MemorySpace = Kokkos::CudaUVMSpace;
using ExecutionSpace = Kokkos::Cuda;
#else
using MemorySpace = Kokkos::HostSpace;
#ifdef CabanaMD_ENABLE_Serial
using ExecutionSpace = Kokkos::Serial;
#elif defined( CabanaMD_ENABLE_Threads )
using ExecutionSpace = Kokkos::Threads;
#elif defined( CabanaMD_ENABLE_OpenMP )
using ExecutionSpace = Kokkos::OpenMP;
#endif
#endif
using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

using MemoryAccess = Cabana::DefaultAccessMemory;
using AtomicAccess = Cabana::AtomicAccessMemory;

using t_linkedcell = Cabana::LinkedCellList<DeviceType>;
using t_distributor = Cabana::Distributor<DeviceType>;
using t_halo = Cabana::Halo<DeviceType>;

using t_verletlist_full_2D =
    Cabana::VerletList<DeviceType, Cabana::FullNeighborTag,
                       Cabana::VerletLayout2D>;
using t_verletlist_half_2D =
    Cabana::VerletList<DeviceType, Cabana::HalfNeighborTag,
                       Cabana::VerletLayout2D>;
using t_verletlist_full_CSR =
    Cabana::VerletList<DeviceType, Cabana::FullNeighborTag,
                       Cabana::VerletLayoutCSR>;
using t_verletlist_half_CSR =
    Cabana::VerletList<DeviceType, Cabana::HalfNeighborTag,
                       Cabana::VerletLayoutCSR>;

using t_neighborop_serial = Cabana::SerialOpTag;
using t_neighborop_team = Cabana::TeamOpTag;
using t_neighborop_vector = Cabana::TeamVectorOpTag;

#ifdef CabanaMD_LAYOUT_1AoSoA
using t_tuple = Cabana::MemberTypes<T_FLOAT[3], T_FLOAT[3], T_FLOAT[3], T_INT,
                                    T_INT, T_FLOAT>;
using AoSoA_1 = Cabana::AoSoA<t_tuple, DeviceType, CabanaMD_VECTORLENGTH>;
using t_x = AoSoA_1::member_slice_type<0>;
using t_v = AoSoA_1::member_slice_type<1>;
using t_f = AoSoA_1::member_slice_type<2>;
using t_type = AoSoA_1::member_slice_type<3>;
using t_id = AoSoA_1::member_slice_type<4>;
using t_q = AoSoA_1::member_slice_type<5>;

#elif defined( CabanaMD_LAYOUT_2AoSoA )
using t_tuple_0 = Cabana::MemberTypes<T_FLOAT[3], T_FLOAT[3], T_INT>;
using t_tuple_1 = Cabana::MemberTypes<T_FLOAT[3], T_INT, T_FLOAT>;
using AoSoA_2_0 = Cabana::AoSoA<t_tuple_0, DeviceType, CabanaMD_VECTORLENGTH>;
using AoSoA_2_1 = Cabana::AoSoA<t_tuple_1, DeviceType, CabanaMD_VECTORLENGTH>;

using t_x = AoSoA_2_0::member_slice_type<0>;
using t_v = AoSoA_2_1::member_slice_type<0>;
using t_f = AoSoA_2_0::member_slice_type<1>;
using t_type = AoSoA_2_0::member_slice_type<2>;
using t_id = AoSoA_2_1::member_slice_type<1>;
using t_q = AoSoA_2_1::member_slice_type<2>;

#elif defined( CabanaMD_LAYOUT_6AoSoA )
using t_tuple_x = Cabana::MemberTypes<T_FLOAT[3]>;
using t_tuple_int = Cabana::MemberTypes<T_INT>;
using t_tuple_fl = Cabana::MemberTypes<T_FLOAT>;
using AoSoA_x = Cabana::AoSoA<t_tuple_x, DeviceType, CabanaMD_VECTORLENGTH>;
using AoSoA_int = Cabana::AoSoA<t_tuple_int, DeviceType, CabanaMD_VECTORLENGTH>;
using AoSoA_fl = Cabana::AoSoA<t_tuple_fl, DeviceType, CabanaMD_VECTORLENGTH>;

using t_x = AoSoA_x::member_slice_type<0>;
using t_v = AoSoA_x::member_slice_type<0>;
using t_f = AoSoA_x::member_slice_type<0>;
using t_type = AoSoA_int::member_slice_type<0>;
using t_id = AoSoA_int::member_slice_type<0>;
using t_q = AoSoA_fl::member_slice_type<0>;
#endif

#endif // TYPES_H
