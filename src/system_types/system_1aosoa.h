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

#ifndef SYSTEM_AOSOA1_H
#define SYSTEM_AOSOA1_H

#include <system.h>
#include <types.h>

#include <CabanaMD_config.hpp>

#include <Cabana_Core.hpp>

template <>
class System<AoSoA1> : public SystemCommon
{
    using t_tuple = Cabana::MemberTypes<T_FLOAT[3], T_FLOAT[3], T_FLOAT[3],
                                        T_INT, T_INT, T_FLOAT>;
    using AoSoA_1 = Cabana::AoSoA<t_tuple, DeviceType, CabanaMD_VECTORLENGTH>;
    AoSoA_1 aosoa_0;

  public:
    using SystemCommon::SystemCommon;

    // Per Particle Property
    using t_x = AoSoA_1::member_slice_type<0>;
    using t_v = AoSoA_1::member_slice_type<1>;
    using t_f = AoSoA_1::member_slice_type<2>;
    using t_type = AoSoA_1::member_slice_type<3>;
    using t_id = AoSoA_1::member_slice_type<4>;
    using t_q = AoSoA_1::member_slice_type<5>;
    t_x x;
    t_v v;
    t_f f;
    t_type type;
    t_id id;
    t_q q;

    void init() override { AoSoA_1 aosoa_0( "All", N_max ); }

    void resize( T_INT N_new ) override
    {
        if ( N_new > N_max )
        {
            N_max = N_new; // Number of global Particles
        }
        // Grow/shrink, slice.size() needs to be accurate
        aosoa_0.resize( N_new );
    }

    void slice_x() override { x = Cabana::slice<0>( aosoa_0 ); }
    void slice_v() override { v = Cabana::slice<1>( aosoa_0 ); }
    void slice_f() override { f = Cabana::slice<2>( aosoa_0 ); }
    void slice_type() override { type = Cabana::slice<3>( aosoa_0 ); }
    void slice_id() override { id = Cabana::slice<4>( aosoa_0 ); }
    void slice_q() override { q = Cabana::slice<5>( aosoa_0 ); }

    void permute( t_linkedcell cell_list ) override
    {
        Cabana::permute( cell_list, aosoa_0 );
    }

    void migrate( std::shared_ptr<t_distributor> distributor ) override
    {
        Cabana::migrate( *distributor, aosoa_0 );
    }

    void gather( std::shared_ptr<t_halo> halo ) override
    {
        Cabana::gather( *halo, aosoa_0 );
    }

    const char *name() override { return "System:1AoSoA"; }
};
#endif
