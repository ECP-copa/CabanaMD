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
#include <stdio.h>

static constexpr T_FLOAT sqrt2 = std::sqrt(2.0);
static constexpr T_FLOAT sqrt3 = std::sqrt(3.0);

// Count the number of nearest and next-nearest neighbors in a given lattice.
struct Nbrs {
  char lattice[4];
  int n1, n2;
  T_FLOAT d1, d2;
  T_FLOAT atoms_per_unit; // dens = atoms_per_unit / lattice_const^3
} nbrs[] = { {"sc",   6, 12,       1.0, sqrt2, 1.0},
             {"bcc",  8,  6, sqrt3/2.0,   1.0, 2.0},
             {"fcc", 12,  6, 1.0/sqrt2,   1.0, 4.0},
             {"hcp", 12,  0,       1.0,   1.0, 1.0}
           };

// E_LJ = 4*eps*( (sigma/r)^12 - (sigma/r)^6 )

T_FLOAT compute_pe(CabanaMD &cmd) {
    PotE pote( cmd.comm );
    T_FLOAT PE = pote.compute( cmd.system, cmd.force ) / cmd.system->N;
    return PE;
}

void init(CabanaMD &cmd, Lattice lat,
          T_FLOAT lattice_constant, T_FLOAT force_cutoff)
{
    // These are needed because modules_force is macro-expanded.
    Input *input = cmd.input;
    Force * &force = cmd.force;
    System *system = cmd.system;
    System &s = *system;

    // Parse command line arguments
    // Read input file
    auto units_style = UNITS_LJ;

    s.boltz = 1.0;
    s.mvv2e = 1.0;
    s.dt = 0.005;
    input->lattice_style = lat;
    input->lattice_constant = lattice_constant;

    input->box[1] = 10;
    input->box[3] = 10;
    input->box[5] = 10;
    input->lattice_nx = input->box[1];
    input->lattice_ny = input->box[3];
    input->lattice_nz = input->box[5];
    s.ntypes = 1;
    s.mass = t_mass( "System::mass", s.ntypes );
    //input->neighbor_skin = 0.3;
    input->neighbor_skin = 0.0;
    //input->force_cutoff = force_cutoff; // only used in initialization

    // CabanaMD.init() resumes
    T_X_FLOAT neigh_cutoff = force_cutoff + input->neighbor_skin;
    cmd.integrator = new Integrator( system ); // NVE integrator
    cmd.binning    = new Binning( system ); // linked cell bin sort
    cmd.comm       = new Comm( system, neigh_cutoff ); // MPI Communicator
 
    if( false ) {}
#define FORCE_MODULES_INSTANTIATION
#include <modules_force.h>
#undef FORCE_MODULES_INSTANTIATION
    else
        cmd.comm->error( "Invalid ForceType" );

    /*for ( std::size_t line = 0; line < input->force_coeff_lines.extent( 0 );
                      line++ )
        {
        cmd.force->init_coeff(
            neigh_cutoff,
            input->input_data.words[input->force_coeff_lines( line )] );
        }*/
    char _args[][8] = {"", "", "",
                       "1.0", //  eps
                       "1.0", // sigma
                       "3.0", // cut
                      };
    char *args[6] = {_args[0], _args[1], _args[2],
                     _args[3], _args[4], _args[5]};
    snprintf(args[5], 8, "%7.4f", force_cutoff);
    cmd.force->init_coeff(neigh_cutoff, args);
    cmd.force->comm_newton = input->comm_newton; // Newton's 2nd law setting

    // ** create lattice **
    input->create_lattice( cmd.comm );

    // exchange atoms across all MPI ranks
    cmd.comm->exchange();

    // Sort atoms
    cmd.binning->create_binning( neigh_cutoff, neigh_cutoff, neigh_cutoff,
                                 1, true, false, true );

    // Add ghost atoms from other MPI ranks (gather)
    cmd.comm->exchange_halo();

    // Compute atom neighbors
    cmd.force->create_neigh_list( cmd.system );

    // Compute initial forces
    auto f = Cabana::slice<Forces>( s.xvf );
    Cabana::deep_copy( f, 0.0 );
    cmd.force->compute( cmd.system );

    // Scatter ghost atom forces back to original MPI rank
    //   (update force for pair_style nnp even if full neighbor list)
    if ( input->comm_newton or input->force_type == FORCE_NNP ) {
        cmd.comm->update_force();
    }
}

Lattice parse_lattice(const char *style) {
    if ( strcmp( style, "fcc" ) == 0 )
    {
        return Lattice::FCC;
    }
    else if ( strcmp( style, "bcc" ) == 0 )
    {
        return Lattice::BCC;
    }
    return Lattice::SC;
}

T_FLOAT lattice_energy(const char *style, T_FLOAT a, T_FLOAT cut) {
    struct Nbrs *nbr = nbrs;
    T_FLOAT en;

    for(int i=0; i<sizeof(nbrs)/sizeof(nbrs[0]); i++, nbr++) {
        if(!strncmp(style, nbr->lattice, 4)) {
            break;
        }
    }
    if(nbr >= nbrs+sizeof(nbrs)/sizeof(nbrs[0])) {
        return 0.0;
    }
    if(a*nbr->d1 < cut) {
        T_FLOAT ir6 = std::pow(a*nbr->d1, -6.0);
        en += nbr->n1*4.0*(ir6*ir6 - ir6);
    }
    if(a*nbr->d2 < cut) {
        T_FLOAT ir6 = std::pow(a*nbr->d2, -6.0);
        en += nbr->n2*4.0*(ir6*ir6 - ir6);
    }

    return en;
}

int main( int argc, char *argv[] ) {
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );
    if(argc != 4) {
        printf("Usage: %s <sc/bcc/fcc> <a> <cutoff>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    {
    Lattice shape = parse_lattice(argv[1]);
    T_FLOAT a = atof(argv[2]);
    T_FLOAT cut = atof(argv[3]);
    CabanaMD cabanamd;
    //cabanamd.init( argc, argv );
    init(cabanamd, shape, a, cut);

    T_FLOAT PE = compute_pe(cabanamd);
    printf("Energy = %g\n", PE);
    printf("Lattice Energy = %g per atom\n",
            lattice_energy(argv[1], a, cut));

    //cabanamd.run( cabanamd.input->nsteps );
    //cabanamd.check_correctness();
    //cabanamd.print_performance();
    }

    MPI_Finalize();

    return 0;
}

