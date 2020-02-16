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

#include <cabanamd.h>
#include <property_pote.h>
#include "mpi.h"

static constexpr sqrt2 = std::sqrt(2.0);
static constexpr sqrt3 = std::sqrt(3.0);

// Count the number of nearest and next-nearest neighbors in a given lattice.
struct {
  char lattice[4];
  int n1, n2;
  T_FLOAT d1, d2;
  T_FLOAT atoms_per_unit; // dens = atoms_per_unit / lattice_const^3
} nbrs[] = { {"sc",   6, 12,       1.0, sqrt2, 1.0},
             {"bcc",  8,  6, sqrt3/2.0,   1.0, 2.0},
             {"fcc", 12,  6, 1.0/sqrt2,   1.0, 4.0},
             {"hcp", 12,  0,       1.0,   1.0, 1.0}
           };

T_FLOAT compute_pe(CabanaMD &md) {
    PotE pote( md.comm );
    T_FLOAT PE = pote.compute( md.system, md.force ) / md.system->N;
    return PE;
}

void init(CabanaMD &cmd, T_FLOAT lattice_constant) {
    auto units_style = UNITS_LJ;
    system->boltz = 1.0;
    system->mvv2e = 1.0;
    system->dt = 0.005;
    auto lattice_style = LATTICE_SC;
    system->ntypes = 1;
    system->mass = t_mass( "System::mass", system->ntypes );

}

int main( int argc, char *argv[] ) {
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    CabanaMD cabanamd;
    cabanamd.init( argc, argv );

    setup_sys(cabanamd);

    T_FLOAT PE = compute_pe(cabanamd);
    printf("Energy = %g\n", PE);

    //cabanamd.run( cabanamd.input->nsteps );
    //cabanamd.check_correctness();
    //cabanamd.print_performance();

    cabanamd.shutdown();

    MPI_Finalize();

    return 0;
}

