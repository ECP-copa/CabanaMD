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

#ifndef SYSTEM_H
#define SYSTEM_H
#include <Cabana_Slice.hpp>
#include<types.h>
#include <string>

class System {
public:
  T_INT N;       // Number of Global Particles
  T_INT N_max;   // Number of Particles I could have in available storage
  T_INT N_local; // Number of owned Particles
  T_INT N_ghost; // Number of non-owned Particles

  int ntypes;
  std::string atom_style; 

  // Per Particle Property
  AoSoA xvf;
  // Per Type Property
  t_mass mass;

  //NNP constants passed in via pair_style(TODO: remove hardcoding)
  bool normalize = true;
  double cflength = 1.889726;
  double cfenergy = 0.036749;
  double cfforce = cflength/cfenergy;
  //TODO: remove this hardcoding as well
  double convLength = 5.8038448995319847E+00; 
  double convEnergy = 2.4265748255366972E+02;
  double mean_energy = -2.5521343547039809E+01;
  // Simulation domain
  T_X_FLOAT domain_x, domain_y, domain_z;
  T_X_FLOAT box[6];
  T_X_FLOAT domain_lo_x, domain_lo_y, domain_lo_z;
  T_X_FLOAT domain_hi_x, domain_hi_y, domain_hi_z;

  // Simulation sub domain (for example of a single MPI rank)
  T_X_FLOAT sub_domain_x, sub_domain_y, sub_domain_z;
  T_X_FLOAT sub_domain_lo_x, sub_domain_lo_y, sub_domain_lo_z;
  T_X_FLOAT sub_domain_hi_x, sub_domain_hi_y, sub_domain_hi_z;

  // Units
  T_FLOAT boltz,mvv2e,dt;

  // Should this process print messages
  bool do_print;
  // Should print LAMMPS style messages
  bool print_lammps;

  System();
  ~System() {};
  void init();
  void destroy();

  void resize(T_INT new_N);

  KOKKOS_INLINE_FUNCTION
  t_particle get_particle(const T_INT& i) const {
    return xvf.getTuple(i);
  }

  KOKKOS_INLINE_FUNCTION
  void set_particle(const T_INT& i, const t_particle& p) const {
    xvf.setTuple(i, p);
  }

  KOKKOS_INLINE_FUNCTION
  void copy(T_INT dest, T_INT src, int nx, int ny, int nz) const {
    auto x = Cabana::slice<Positions>(xvf);
    auto v = Cabana::slice<Velocities>(xvf);
    auto q = Cabana::slice<Charges>(xvf);
    auto id = Cabana::slice<IDs>(xvf);
    auto type = Cabana::slice<Types>(xvf);

    x(dest,0) = x(src,0) + domain_x * nx;
    x(dest,1) = x(src,1) + domain_y * ny;
    x(dest,2) = x(src,2) + domain_z * nz;
    v(dest,0) = v(src,0);
    v(dest,1) = v(src,1);
    v(dest,2) = v(src,2);
    type(dest) = type(src);
    id(dest) = id(src);
    q(dest) = q(src);
  }

  KOKKOS_INLINE_FUNCTION
  void copy_halo_update(T_INT dest, T_INT src, int nx, int ny, int nz) const {
    auto x = Cabana::slice<Positions>(xvf);

    x(dest,0) = x(src,0) + domain_x * nx;
    x(dest,1) = x(src,1) + domain_y * ny;
    x(dest,2) = x(src,2) + domain_z * nz;
  }

  void print_particles();
};
#endif
