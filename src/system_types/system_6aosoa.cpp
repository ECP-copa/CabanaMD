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

#include <system_6aosoa.h>

void System6AoSoA::init()
{
    AoSoA_x aosoa_x( "X", N_max );
    AoSoA_x aosoa_v( "V", N_max );
    AoSoA_x aosoa_f( "F", N_max );
    AoSoA_int aosoa_id( "ID", N_max );
    AoSoA_int aosoa_type( "Type", N_max );
    AoSoA_fl aosoa_q( "Q", N_max );
}

void System6AoSoA::resize( T_INT N_new )
{
    if ( N_new > N_max )
    {
        N_max = N_new; // Number of global Particles
    }
    // Grow/shrink, slice.size() needs to be accurate
    aosoa_x.resize( N_new );
    aosoa_v.resize( N_new );
    aosoa_f.resize( N_new );
    aosoa_id.resize( N_new );
    aosoa_type.resize( N_new );
    aosoa_q.resize( N_new );
}

void System6AoSoA::slice_x() { x = Cabana::slice<0>( aosoa_x ); }
void System6AoSoA::slice_v() { v = Cabana::slice<0>( aosoa_v ); }
void System6AoSoA::slice_f() { f = Cabana::slice<0>( aosoa_f ); }
void System6AoSoA::slice_type() { type = Cabana::slice<0>( aosoa_type ); }
void System6AoSoA::slice_id() { id = Cabana::slice<0>( aosoa_id ); }
void System6AoSoA::slice_q() { q = Cabana::slice<0>( aosoa_q ); }

void System6AoSoA::permute( t_linkedcell cell_list )
{
    Cabana::permute( cell_list, aosoa_x );
    Cabana::permute( cell_list, aosoa_v );
    Cabana::permute( cell_list, aosoa_f );
    Cabana::permute( cell_list, aosoa_type );
    Cabana::permute( cell_list, aosoa_id );
    Cabana::permute( cell_list, aosoa_q );
}

void System6AoSoA::migrate( std::shared_ptr<t_distributor> distributor )
{
    Cabana::migrate( *distributor, aosoa_x );
    Cabana::migrate( *distributor, aosoa_v );
    Cabana::migrate( *distributor, aosoa_f );
    Cabana::migrate( *distributor, aosoa_type );
    Cabana::migrate( *distributor, aosoa_id );
    Cabana::migrate( *distributor, aosoa_q );
}

void System6AoSoA::gather( std::shared_ptr<t_halo> halo )
{
    Cabana::gather( *halo, aosoa_x );
    Cabana::gather( *halo, aosoa_v );
    Cabana::gather( *halo, aosoa_f );
    Cabana::gather( *halo, aosoa_type );
    Cabana::gather( *halo, aosoa_id );
    Cabana::gather( *halo, aosoa_q );
}
