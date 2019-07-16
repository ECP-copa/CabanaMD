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

#include<system.h>
#include<mpi.h>

System::System() {
  N = 0;
  N_max = 0;
  N_local = 0;
  N_ghost = 0;
  ntypes = 1;
  atom_style = "atomic"; 

  mass = t_mass();
  domain_x = domain_y = domain_z = 0.0;
  sub_domain_x = sub_domain_y = sub_domain_z = 0.0;
  sub_domain_hi_x = sub_domain_hi_y = sub_domain_hi_z = 0.0;
  sub_domain_lo_x = sub_domain_lo_y = sub_domain_lo_z = 0.0;
  mvv2e = boltz = dt = 0.0;

  int proc_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  do_print = proc_rank == 0;
  print_lammps = false;
}

void System::init() {
  AoSoA xvf ( "All", N_max );
  mass = t_mass("System::mass",ntypes);
}

void System::destroy() {
  N_max = 0;
  N_local = 0;
  N_ghost = 0;
  ntypes = 1;
  AoSoA xvf( "All", 0 );
  mass = t_mass();
}

void System::resize(T_INT N_new) {
  if(N_new > N_max) {
    N_max = N_new; // Number of global Particles
  }
  // Grow/shrink, slice.size() needs to be accurate
  xvf.resize( N_new );
}

void System::print_particles() {

  auto x = Cabana::slice<Positions>(xvf);
  auto v = Cabana::slice<Velocities>(xvf);
  auto f = Cabana::slice<Forces>(xvf);
  auto type = Cabana::slice<Types>(xvf);
  auto q = Cabana::slice<Charges>(xvf);

  printf("Print all particles: \n");
  printf("  Owned: %d\n",N_local);
  for(T_INT i=0;i<N_local;i++) {
    printf("    %d %lf %lf %lf | %lf %lf %lf | %lf %lf %lf | %d %e\n",i,
        double(x(i,0)),double(x(i,1)),double(x(i,2)),
        double(v(i,0)),double(v(i,1)),double(v(i,2)),
        double(f(i,0)),double(f(i,1)),double(f(i,2)),
        type(i),q(i)
        );
  }

  printf("  Ghost: %d\n",N_ghost);
  for(T_INT i=N_local;i<N_local+N_ghost;i++) {
    printf("    %d %lf %lf %lf | %lf %lf %lf | %lf %lf %lf | %d %e\n",i,
        double(x(i,0)),double(x(i,1)),double(x(i,2)),
        double(v(i,0)),double(v(i,1)),double(v(i,2)),
        double(f(i,0)),double(f(i,1)),double(f(i,2)),
        type(i),q(i)
        );
  }

}
