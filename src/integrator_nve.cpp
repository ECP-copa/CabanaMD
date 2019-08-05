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

#include <integrator_nve.h>

Integrator::Integrator(System* p):system(p) {
  dtf = 0.5 * system->dt / system->mvv2e;
  dtv = system->dt;
}

namespace {
  struct InitialIntegrateFunctor {
    t_AoSoA_x aosoa_x;
    t_AoSoA_x aosoa_v;
    t_AoSoA_x aosoa_f;
    t_AoSoA_int aosoa_type;
    typename t_AoSoA_x::member_slice_type<0> x;
    typename t_AoSoA_x::member_slice_type<0> v;
    typename t_AoSoA_x::member_slice_type<0> f;
    typename t_AoSoA_int::member_slice_type<0> type;

    t_mass_const mass;

    T_V_FLOAT dtf,dtv;
    int step;

    InitialIntegrateFunctor(t_AoSoA_x &aosoa_x_, t_AoSoA_x &aosoa_v_,
                            t_AoSoA_x &aosoa_f_, t_AoSoA_int &aosoa_type_,
                            t_mass mass_, T_V_FLOAT dtf_,
                            T_V_FLOAT dtv_, int step_
                            ): aosoa_x(aosoa_x_),aosoa_v(aosoa_v_),
                               aosoa_f(aosoa_f_),aosoa_type(aosoa_type_),
                               mass(mass_),dtf(dtf_),dtv(dtv_),step(step_)
    {
      x = Cabana::slice<0>(aosoa_x);
      v = Cabana::slice<0>(aosoa_v);
      f = Cabana::slice<0>(aosoa_f);
      type = Cabana::slice<0>(aosoa_type);
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (const T_INT& i) const {
      const T_V_FLOAT dtfm = dtf / mass(type(i));
      v(i,0) += dtfm * f(i,0);
      v(i,1) += dtfm * f(i,1);
      v(i,2) += dtfm * f(i,2);
      x(i,0) += dtv * v(i,0);
      x(i,1) += dtv * v(i,1);
      x(i,2) += dtv * v(i,2);
    }
  };
}

void Integrator::initial_integrate() {
  static int step =1;
  Kokkos::parallel_for("IntegratorNVE::initial_integrate",system->N_local,
                       InitialIntegrateFunctor(system->aosoa_x, system->aosoa_v,
                                               system->aosoa_f, system->aosoa_type,
                                               system->mass, dtf, dtv, step)
                       );
  step++;
}


namespace {
  struct FinalIntegrateFunctor {
    t_AoSoA_x aosoa_x;
    t_AoSoA_x aosoa_v;
    t_AoSoA_x aosoa_f;
    t_AoSoA_int aosoa_type;
    typename t_AoSoA_x::member_slice_type<0> x;
    typename t_AoSoA_x::member_slice_type<0> v;
    typename t_AoSoA_x::member_slice_type<0> f;
    typename t_AoSoA_int::member_slice_type<0> type;

    t_mass_const mass;

    T_V_FLOAT dtf,dtv;
    int step;

    FinalIntegrateFunctor(t_AoSoA_x &aosoa_x_, t_AoSoA_x &aosoa_v_,
                          t_AoSoA_x &aosoa_f_, t_AoSoA_int &aosoa_type_,
                          t_mass mass_, T_V_FLOAT dtf_,
                          T_V_FLOAT dtv_, int step_
                          ): aosoa_x(aosoa_x_),aosoa_v(aosoa_v_),
                             aosoa_f(aosoa_f_),aosoa_type(aosoa_type_),
                             mass(mass_),dtf(dtf_),dtv(dtv_),step(step_)
    {
      x = Cabana::slice<0>(aosoa_x);
      v = Cabana::slice<0>(aosoa_v);
      f = Cabana::slice<0>(aosoa_f);
      type = Cabana::slice<0>(aosoa_type);
    }

    KOKKOS_INLINE_FUNCTION
    void operator() (const T_INT& i) const {
      const T_V_FLOAT dtfm = dtf / mass(type(i));
      v(i,0) += dtfm * f(i,0);
      v(i,1) += dtfm * f(i,1);
      v(i,2) += dtfm * f(i,2);
    }
  };
}

void Integrator::final_integrate() {
  static int step = 1;
  Kokkos::parallel_for("IntegratorNVE::final_integrate",system->N_local, 
                       FinalIntegrateFunctor(system->aosoa_x, system->aosoa_v,
                                             system->aosoa_f, system->aosoa_type,
                                             system->mass, dtf, dtv, step)
                       );
  step++;
}

const char* Integrator::name() { return "Integrator:NVE"; }
