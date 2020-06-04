/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
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

// Device type
enum
{
    CUDA,
    OPENMP,
    SERIAL
};

// AoSoA layout type
enum
{
    AOSOA_1,
    AOSOA_2,
    AOSOA_3,
    AOSOA_6
};
struct AoSoA1
{
};
struct AoSoA2
{
};
struct AoSoA3
{
};
struct AoSoA6
{
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
    NEIGH_VERLET_2D,
    NEIGH_VERLET_CSR,
    NEIGH_TREE
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

#endif // TYPES_H
