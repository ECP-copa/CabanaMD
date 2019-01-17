/****************************************************************************
 * Copyright (c) 2018 by the Cabana authors                                 *
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
#include<Cabana_Core.hpp>
#define VECLEN 16

// Module Types etc
// Units to be used
enum {UNITS_REAL,UNITS_LJ,UNITS_METAL};
// Lattice Type
enum {LATTICE_SC,LATTICE_FCC};
// Integrator Type
enum {INTEGRATOR_NVE};
// Binning Type
enum {BINNING_CABANA};
// Comm Type
enum {COMM_SERIAL};
// Force Type
enum {FORCE_LJ_CABANA_NEIGH};
// Force Iteration Type
enum {FORCE_ITER_NEIGH_FULL, FORCE_ITER_NEIGH_HALF};
// Neighbor Type
enum {NEIGH_CABANA_VERLET};

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

using MemorySpace = Cabana::HostSpace;
using MemoryAccess = Cabana::DefaultAccessMemory;
using AtomicAccess = Cabana::AtomicAccessMemory;
using AoSoA = Cabana::AoSoA<t_tuple,MemorySpace,VECLEN>;
using t_particle = Cabana::Tuple<t_tuple>;
using t_slice_x = Cabana::Slice<T_FLOAT[3],MemorySpace,MemoryAccess,VECLEN>;
using t_slice_x_atomic = Cabana::Slice<T_FLOAT[3],MemorySpace,AtomicAccess,VECLEN>;
using t_slice_int = Cabana::Slice<T_INT,MemorySpace,MemoryAccess,VECLEN>;
using t_slice_fl = Cabana::Slice<T_FLOAT,MemorySpace,MemoryAccess,VECLEN>;

using t_verletlist_full = Cabana::VerletList<MemorySpace,Cabana::FullNeighborTag>;
using t_verletlist_half = Cabana::VerletList<MemorySpace,Cabana::HalfNeighborTag>;

#endif

