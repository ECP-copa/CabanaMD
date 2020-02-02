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

#include <system_2aosoa.h>

void System2AoSoA::init()
{
    AoSoA_2_0 aosoa_0( "X,F,Type", N_max );
    AoSoA_2_1 aosoa_1( "V,ID,Q", N_max );
}

void System2AoSoA::resize( T_INT N_new )
{
    if ( N_new > N_max )
    {
        N_max = N_new; // Number of global Particles
    }
    // Grow/shrink, slice.size() needs to be accurate
    aosoa_0.resize( N_new );
    aosoa_1.resize( N_new );
}

void System2AoSoA::slice_x() { x = Cabana::slice<0>( aosoa_0 ); }
void System2AoSoA::slice_v() { v = Cabana::slice<0>( aosoa_1 ); }
void System2AoSoA::slice_f() { f = Cabana::slice<1>( aosoa_0 ); }
void System2AoSoA::slice_type() { type = Cabana::slice<2>( aosoa_0 ); }
void System2AoSoA::slice_id() { id = Cabana::slice<1>( aosoa_1 ); }
void System2AoSoA::slice_q() { q = Cabana::slice<2>( aosoa_1 ); }

void System2AoSoA::permute( t_linkedcell linkedcell )
{
    Cabana::permute( linkedcell, aosoa_0 );
    Cabana::permute( linkedcell, aosoa_1 );
}

void System2AoSoA::migrate( t_distributor distributor )
{
    Cabana::migrate( distributor, aosoa_0 );
    Cabana::migrate( distributor, aosoa_1 );
}

void System2AoSoA::gather( t_halo halo )
{
    Cabana::gather( halo, aosoa_0 );
    Cabana::gather( halo, aosoa_1 );
}
