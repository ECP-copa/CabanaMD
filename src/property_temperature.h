/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
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

#ifndef PROPERTY_TEMP_H
#define PROPERTY_TEMP_H

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <comm_mpi.h>
#include <types.h>

template <class t_System>
class Temperature
{
  private:
    typename t_System::t_v v;
    typename t_System::t_type type;
    typename t_System::t_mass mass;

    Comm<t_System> *comm;

  public:
    Temperature( Comm<t_System> *comm_ );

    T_V_FLOAT compute( t_System * );

    KOKKOS_INLINE_FUNCTION
    void operator()( const T_INT &i, T_V_FLOAT &T ) const
    {
        T += ( v( i, 0 ) * v( i, 0 ) + v( i, 1 ) * v( i, 1 ) +
               v( i, 2 ) * v( i, 2 ) ) *
             mass( type( i ) );
    }
};

#include <property_temperature_impl.h>
#endif
