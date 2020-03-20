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

#ifndef SYSTEM_6AOSOA_H
#define SYSTEM_6AOSOA_H

#include <system.h>

template <>
class System<AoSoA6> : public SystemCommon<AoSoA6>
{
    using t_tuple_x = Cabana::MemberTypes<T_FLOAT[3]>;
    using t_tuple_int = Cabana::MemberTypes<T_INT>;
    using t_tuple_fl = Cabana::MemberTypes<T_FLOAT>;
    using AoSoA_x = Cabana::AoSoA<t_tuple_x, DeviceType, CabanaMD_VECTORLENGTH>;
    using AoSoA_int =
        Cabana::AoSoA<t_tuple_int, DeviceType, CabanaMD_VECTORLENGTH>;
    using AoSoA_fl =
        Cabana::AoSoA<t_tuple_fl, DeviceType, CabanaMD_VECTORLENGTH>;
    AoSoA_x aosoa_x;
    AoSoA_x aosoa_v;
    AoSoA_x aosoa_f;
    AoSoA_int aosoa_id;
    AoSoA_int aosoa_type;
    AoSoA_fl aosoa_q;

  public:
    using SystemCommon<AoSoA6>::SystemCommon;

    // Per Particle Property
    using t_x = AoSoA_x::member_slice_type<0>;
    using t_v = AoSoA_x::member_slice_type<0>;
    using t_f = AoSoA_x::member_slice_type<0>;
    using t_type = AoSoA_int::member_slice_type<0>;
    using t_id = AoSoA_int::member_slice_type<0>;
    using t_q = AoSoA_fl::member_slice_type<0>;
    t_x x;
    t_v v;
    t_f f;
    t_type type;
    t_id id;
    t_q q;

    void init()
    {
        AoSoA_x aosoa_x( "X", N_max );
        AoSoA_x aosoa_v( "V", N_max );
        AoSoA_x aosoa_f( "F", N_max );
        AoSoA_int aosoa_id( "ID", N_max );
        AoSoA_int aosoa_type( "Type", N_max );
        AoSoA_fl aosoa_q( "Q", N_max );
    }

    void resize( T_INT N_new )
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

    void slice_x() { x = Cabana::slice<0>( aosoa_x ); }
    void slice_v() { v = Cabana::slice<0>( aosoa_v ); }
    void slice_f() { f = Cabana::slice<0>( aosoa_f ); }
    void slice_type() { type = Cabana::slice<0>( aosoa_type ); }
    void slice_id() { id = Cabana::slice<0>( aosoa_id ); }
    void slice_q() { q = Cabana::slice<0>( aosoa_q ); }

    void permute( t_linkedcell cell_list )
    {
        Cabana::permute( cell_list, aosoa_x );
        Cabana::permute( cell_list, aosoa_v );
        Cabana::permute( cell_list, aosoa_f );
        Cabana::permute( cell_list, aosoa_type );
        Cabana::permute( cell_list, aosoa_id );
        Cabana::permute( cell_list, aosoa_q );
    }

    void migrate( std::shared_ptr<t_distributor> distributor )
    {
        Cabana::migrate( *distributor, aosoa_x );
        Cabana::migrate( *distributor, aosoa_v );
        Cabana::migrate( *distributor, aosoa_f );
        Cabana::migrate( *distributor, aosoa_type );
        Cabana::migrate( *distributor, aosoa_id );
        Cabana::migrate( *distributor, aosoa_q );
    }

    void gather( std::shared_ptr<t_halo> halo )
    {
        Cabana::gather( *halo, aosoa_x );
        Cabana::gather( *halo, aosoa_v );
        Cabana::gather( *halo, aosoa_f );
        Cabana::gather( *halo, aosoa_type );
        Cabana::gather( *halo, aosoa_id );
        Cabana::gather( *halo, aosoa_q );
    }

    const char *name() { return "System:6AoSoA"; }
};
#endif
