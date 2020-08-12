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

#ifndef FORCE_LJ_CABANA_NEIGH_H
#define FORCE_LJ_CABANA_NEIGH_H

#include <force.h>
#include <neighbor.h>
#include <system.h>
#include <types.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <string>
#include <vector>

template <class t_System, class t_Neighbor, class t_parallel>
class ForceLJ : public Force<t_System, t_Neighbor>
{
  private:
    int N_local, ntypes;

    typedef typename t_System::t_f::atomic_access_slice t_f_a;

    int step;

    typedef Kokkos::View<T_F_FLOAT **> t_fparams;
    typedef Kokkos::View<const T_F_FLOAT **,
                         Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        t_fparams_rnd;
    t_fparams lj1, lj2, cutsq;
    t_fparams_rnd rnd_lj1, rnd_lj2, rnd_cutsq;

    T_F_FLOAT stack_lj1[MAX_TYPES_STACKPARAMS + 1]
                       [MAX_TYPES_STACKPARAMS +
                        1]; // hardwired space for 12 atom types
    T_F_FLOAT stack_lj2[MAX_TYPES_STACKPARAMS + 1][MAX_TYPES_STACKPARAMS + 1];
    T_F_FLOAT stack_cutsq[MAX_TYPES_STACKPARAMS + 1][MAX_TYPES_STACKPARAMS + 1];

    using exe_space = typename t_System::execution_space;

  public:
    ForceLJ( t_System *system );

    void init_coeff( std::vector<std::vector<std::string>> args ) override;
    void compute( t_System *system, t_Neighbor *neighbor ) override;
    T_F_FLOAT compute_energy( t_System *system, t_Neighbor *neighbor ) override;

    template <class t_f, class t_x, class t_type, class t_neigh>
    void compute_force_full( t_f f, const t_x x, const t_type type,
                             const t_neigh neigh_list );
    template <class t_f, class t_x, class t_type, class t_neigh>
    void compute_force_half( t_f f, const t_x x, const t_type type,
                             const t_neigh neigh_list );

    template <class t_x, class t_type, class t_neigh>
    T_F_FLOAT compute_energy_full( const t_x x, const t_type type,
                                   const t_neigh neigh_list );
    template <class t_x, class t_type, class t_neigh>
    T_F_FLOAT compute_energy_half( const t_x x, const t_type type,
                                   const t_neigh neigh_list );

    const char *name() override;
};

#include <force_lj_cabana_neigh_impl.h>

#endif
