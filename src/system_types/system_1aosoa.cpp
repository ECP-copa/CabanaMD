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

#include <system_1aosoa.h>

void System1AoSoA::init() { AoSoA_1 aosoa_0( "All", N_max ); }

void System1AoSoA::resize( T_INT N_new )
{
    if ( N_new > N_max )
    {
        N_max = N_new; // Number of global Particles
    }
    // Grow/shrink, slice.size() needs to be accurate
    aosoa_0.resize( N_new );
}

void System1AoSoA::slice_x() { x = Cabana::slice<0>( aosoa_0 ); }
void System1AoSoA::slice_v() { v = Cabana::slice<1>( aosoa_0 ); }
void System1AoSoA::slice_f() { f = Cabana::slice<2>( aosoa_0 ); }
void System1AoSoA::slice_type() { type = Cabana::slice<3>( aosoa_0 ); }
void System1AoSoA::slice_id() { id = Cabana::slice<4>( aosoa_0 ); }
void System1AoSoA::slice_q() { q = Cabana::slice<5>( aosoa_0 ); }

void System1AoSoA::permute( t_linkedcell cell_list )
{
    Cabana::permute( cell_list, aosoa_0 );
}

void System1AoSoA::migrate( t_distributor distributor )
{
    Cabana::migrate( distributor, aosoa_0 );
}

void System1AoSoA::gather( t_halo halo ) { Cabana::gather( halo, aosoa_0 ); }
