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

#ifndef INTEGRATOR_NVE_H
#define INTEGRATOR_NVE_H

#include <system.h>
#include <types.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

template <class t_System>
class Integrator
{
    T_V_FLOAT dtv, dtf;

    typename t_System::t_x x;
    typename t_System::t_v v;
    typename t_System::t_f f;
    typename t_System::t_type type;

    t_mass_const mass;

  public:
    Integrator( t_System *s );
    ~Integrator() {}
    T_V_FLOAT timestep_size;

    void initial_integrate( t_System *s );
    void final_integrate( t_System *s );

    const char *name();

    struct TagInitial
    {
    };
    struct TagFinal
    {
    };
    typedef Kokkos::RangePolicy<TagInitial, Kokkos::IndexType<T_INT>>
        t_policy_initial;
    typedef Kokkos::RangePolicy<TagFinal, Kokkos::IndexType<T_INT>>
        t_policy_final;

    KOKKOS_INLINE_FUNCTION
    void operator()( TagInitial, const T_INT &i ) const
    {
        const T_V_FLOAT dtfm = dtf / mass( type( i ) );
        v( i, 0 ) += dtfm * f( i, 0 );
        v( i, 1 ) += dtfm * f( i, 1 );
        v( i, 2 ) += dtfm * f( i, 2 );
        x( i, 0 ) += dtv * v( i, 0 );
        x( i, 1 ) += dtv * v( i, 1 );
        x( i, 2 ) += dtv * v( i, 2 );
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( TagFinal, const T_INT &i ) const
    {
        const T_V_FLOAT dtfm = dtf / mass( type( i ) );
        v( i, 0 ) += dtfm * f( i, 0 );
        v( i, 1 ) += dtfm * f( i, 1 );
        v( i, 2 ) += dtfm * f( i, 2 );
    }
};

#include <integrator_nve_impl.h>
#endif
