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
//    2. Redistributions in binary form must reproduce the above copyright notice,
//       this list of conditions and the following disclaimer in the documentation
//       and/or other materials provided with the distribution.
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
//  Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//************************************************************************

#ifndef TYPES_H
#define TYPES_H
#include<Kokkos_Core.hpp>
#include<Cabana_AoSoA.hpp>
#include<Cabana_VerletList.hpp>
#include<CabanaCore_config.hpp>

#define VECLEN 16
#define MAX_SF 30 //TODO: hardcoded 

// Module Types etc
// Units to be used
enum {UNITS_REAL,UNITS_LJ,UNITS_METAL};
// Lattice Type
enum {LATTICE_SC,LATTICE_FCC};
// Integrator Type
enum {INTEGRATOR_NVE};
// Binning Type
enum {BINNING_LINKEDCELL};
// Comm Type
enum {COMM_SERIAL};
// Force Type
enum {FORCE_LJ,FORCE_SNAP, FORCE_NNP};
// Force Iteration Type
enum {FORCE_ITER_NEIGH_FULL, FORCE_ITER_NEIGH_HALF};
// Neighbor Type
enum {NEIGH_2D,NEIGH_CSR};
// Input File Type
enum {INPUT_LAMMPS};

// Macros to work around the fact that std::max/min is not available on GPUs
#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a<b?a:b)

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

typedef Kokkos::View<T_V_FLOAT*>          t_mass;       // Mass
typedef Kokkos::View<const T_V_FLOAT*>    t_mass_const; // Mass


//Cabana
using t_tuple = Cabana::MemberTypes<T_FLOAT[3], T_FLOAT[3], T_FLOAT[3],
                                    T_INT, T_INT, T_FLOAT>;
enum TypeNames { Positions = 0, Velocities = 1, Forces = 2,
                 Types = 3, IDs = 4, Charges = 5 };
enum NNPNames { G = 0, dEdG = 1, energy = 2};
enum ScalingType
{
    /** @f$G_\text{scaled} = G@f$
     */
    ST_NONE,
    /** @f$G_\text{scaled} = S_\text{min} + \left(S_\text{max} -
     * S_\text{min}\right) \cdot \frac{G - G_\text{min}}
     * {G_\text{max} - G_\text{min}} @f$
     */
    ST_SCALE,
    /** @f$G_\text{scaled} = G - \left<G\right>@f$
     */
    ST_CENTER,
    /** @f$G_\text{scaled} = S_\text{min} + \left(S_\text{max} -
     * S_\text{min}\right) \cdot \frac{G - \left<G\right>}
     * {G_\text{max} - G_\text{min}} @f$
     */
    ST_SCALECENTER,
    /** @f$G_\text{scaled} = S_\text{min} + \left(S_\text{max} -
     * S_\text{min}\right) \cdot \frac{G - \left<G\right>}{\sigma_G} @f$
     */
    ST_SCALESIGMA
};

#ifdef CabanaMD_ENABLE_Cuda
using MemorySpace = Kokkos::CudaUVMSpace;
using ExecutionSpace = Kokkos::Cuda;
#else
using MemorySpace = Kokkos::HostSpace;
#ifdef CabanaMD_ENABLE_Serial
using ExecutionSpace = Kokkos::Serial;
#else // CabanaMD_ENABLE_OpenMP
using ExecutionSpace = Kokkos::OpenMP;
#endif
#endif
using DeviceType = Kokkos::Device<ExecutionSpace,MemorySpace>;

using MemoryAccess = Cabana::DefaultAccessMemory;
using AtomicAccess = Cabana::AtomicAccessMemory;
using AoSoA = Cabana::AoSoA<t_tuple,DeviceType,VECLEN>;
using t_particle = Cabana::Tuple<t_tuple>;
using t_tuple_NNP = Cabana::MemberTypes<T_FLOAT[MAX_SF], T_FLOAT[MAX_SF], T_FLOAT>;
using AoSoA_NNP = Cabana::AoSoA<t_tuple_NNP,MemorySpace,VECLEN>;
using t_dGdr = Kokkos::View<T_V_FLOAT*[MAX_SF][3],Kokkos::MemoryTraits<Kokkos::Atomic>>;


typedef ExecutionSpace::array_layout array_layout; //TODO: check this
using h_t_mass =  Kokkos::View<T_V_FLOAT*,array_layout,Kokkos::HostSpace>;
using d_t_SF = Kokkos::View<T_FLOAT**[15]>;
using t_SF = Kokkos::View<T_FLOAT**[15],array_layout,Kokkos::HostSpace>;
using d_t_SFscaling = Kokkos::View<T_FLOAT**[8]>;
using t_SFscaling = Kokkos::View<T_FLOAT**[8],array_layout,Kokkos::HostSpace>;
using d_t_SFGmemberlist = Kokkos::View<T_INT*[MAX_SF][MAX_SF]>; 
using t_SFGmemberlist = Kokkos::View<T_INT*[MAX_SF][MAX_SF],array_layout,Kokkos::HostSpace>; 
    
using d_t_bias = Kokkos::View<T_FLOAT***>;
using t_bias = Kokkos::View<T_FLOAT***,array_layout,Kokkos::HostSpace>;
using d_t_weights = Kokkos::View<T_FLOAT****>;
using t_weights = Kokkos::View<T_FLOAT****,array_layout,Kokkos::HostSpace>;

using t_linkedcell = Cabana::LinkedCellList<DeviceType>;
using t_verletlist_full_2D = Cabana::VerletList<DeviceType,Cabana::FullNeighborTag,Cabana::VerletLayout2D>;
using t_verletlist_half_2D = Cabana::VerletList<DeviceType,Cabana::HalfNeighborTag,Cabana::VerletLayout2D>;
using t_verletlist_full_CSR = Cabana::VerletList<DeviceType,Cabana::FullNeighborTag,Cabana::VerletLayoutCSR>;
using t_verletlist_half_CSR = Cabana::VerletList<DeviceType,Cabana::HalfNeighborTag,Cabana::VerletLayoutCSR>;

#endif

