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

#ifndef SYSTEM_2AOSOA_H
#define SYSTEM_2AOSOA_H

#include <system.h>
#include <types.h>

#include <CabanaMD_config.hpp>

#include <Cabana_Core.hpp>

template <>
class System<AoSoA2> : public SystemCommon
{
    using t_tuple_0 = Cabana::MemberTypes<T_FLOAT[3], T_FLOAT[3], T_INT>;
    using t_tuple_1 = Cabana::MemberTypes<T_FLOAT[3], T_INT, T_FLOAT>;
    using AoSoA_2_0 =
        Cabana::AoSoA<t_tuple_0, DeviceType, CabanaMD_VECTORLENGTH>;
    using AoSoA_2_1 =
        Cabana::AoSoA<t_tuple_1, DeviceType, CabanaMD_VECTORLENGTH>;
    AoSoA_2_0 aosoa_0;
    AoSoA_2_1 aosoa_1;

  public:
    using SystemCommon::SystemCommon;

    // Per Particle Property
    using t_x = AoSoA_2_0::member_slice_type<0>;
    using t_v = AoSoA_2_1::member_slice_type<0>;
    using t_f = AoSoA_2_0::member_slice_type<1>;
    using t_type = AoSoA_2_0::member_slice_type<2>;
    using t_id = AoSoA_2_1::member_slice_type<1>;
    using t_q = AoSoA_2_1::member_slice_type<2>;
    t_x x;
    t_v v;
    t_f f;
    t_type type;
    t_id id;
    t_q q;

    void init() override
    {
        AoSoA_2_0 aosoa_0( "X,F,Type", N_max );
        AoSoA_2_1 aosoa_1( "V,ID,Q", N_max );
    }

    void resize( T_INT N_new ) override
    {
        if ( N_new > N_max )
        {
            N_max = N_new; // Number of global Particles
        }
        // Grow/shrink, slice.size() needs to be accurate
        aosoa_0.resize( N_new );
        aosoa_1.resize( N_new );
    }

    void slice_x() override { x = Cabana::slice<0>( aosoa_0 ); }
    void slice_v() override { v = Cabana::slice<0>( aosoa_1 ); }
    void slice_f() override { f = Cabana::slice<1>( aosoa_0 ); }
    void slice_type() override { type = Cabana::slice<2>( aosoa_0 ); }
    void slice_id() override { id = Cabana::slice<1>( aosoa_1 ); }
    void slice_q() override { q = Cabana::slice<2>( aosoa_1 ); }

    void permute( t_linkedcell linkedcell ) override
    {
        Cabana::permute( linkedcell, aosoa_0 );
        Cabana::permute( linkedcell, aosoa_1 );
    }

    void migrate( std::shared_ptr<t_distributor> distributor ) override
    {
        Cabana::migrate( *distributor, aosoa_0 );
        Cabana::migrate( *distributor, aosoa_1 );
    }

    void gather( std::shared_ptr<t_halo> halo ) override
    {
        Cabana::gather( *halo, aosoa_0 );
        Cabana::gather( *halo, aosoa_1 );
    }

    const char *name() override { return "System:2AoSoA"; }
};
#endif
