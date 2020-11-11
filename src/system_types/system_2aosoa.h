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

#ifndef SYSTEM_2AOSOA_H
#define SYSTEM_2AOSOA_H

#include <CabanaMD_config.hpp>

#include <system.h>

template <class t_device>
class System<t_device, 2> : public SystemCommon<t_device>
{
    using t_tuple_0 = Cabana::MemberTypes<T_FLOAT[3], T_FLOAT[3], T_INT>;
    using t_tuple_1 = Cabana::MemberTypes<T_FLOAT[3], T_INT, T_FLOAT>;
    using AoSoA_2_0 =
        typename Cabana::AoSoA<t_tuple_0, t_device, CabanaMD_VECTORLENGTH_0>;
    using AoSoA_2_1 =
        typename Cabana::AoSoA<t_tuple_1, t_device, CabanaMD_VECTORLENGTH_1>;
    AoSoA_2_0 aosoa_0;
    AoSoA_2_1 aosoa_1;

    using SystemCommon<t_device>::N_max;
    // using SystemCommon<t_device>::mass;

  public:
    using SystemCommon<t_device>::SystemCommon;

    using memory_space = typename t_device::memory_space;
    using execution_space = typename t_device::execution_space;

    // Per Particle Property
    using t_x =
        typename System<t_device, 2>::AoSoA_2_0::template member_slice_type<0>;
    using t_v =
        typename System<t_device, 2>::AoSoA_2_1::template member_slice_type<0>;
    using t_f =
        typename System<t_device, 2>::AoSoA_2_0::template member_slice_type<1>;
    using t_type =
        typename System<t_device, 2>::AoSoA_2_0::template member_slice_type<2>;
    using t_id =
        typename System<t_device, 2>::AoSoA_2_1::template member_slice_type<1>;
    using t_q =
        typename System<t_device, 2>::AoSoA_2_1::template member_slice_type<2>;
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

    template <class SrcSystem>
    void deep_copy( SrcSystem src_system )
    {
        Cabana::deep_copy( aosoa_0, src_system.get_aosoa_x() );
        Cabana::deep_copy( aosoa_1, src_system.get_aosoa_v() );
    }

    void slice_x() override { x = Cabana::slice<0>( aosoa_0 ); }
    void slice_v() override { v = Cabana::slice<0>( aosoa_1 ); }
    void slice_f() override { f = Cabana::slice<1>( aosoa_0 ); }
    void slice_type() override { type = Cabana::slice<2>( aosoa_0 ); }
    void slice_id() override { id = Cabana::slice<1>( aosoa_1 ); }
    void slice_q() override { q = Cabana::slice<2>( aosoa_1 ); }

    void permute( Cabana::LinkedCellList<t_device> linkedcell ) override
    {
        Cabana::permute( linkedcell, aosoa_0 );
        Cabana::permute( linkedcell, aosoa_1 );
    }

    void migrate(
        std::shared_ptr<Cabana::Distributor<t_device>> distributor ) override
    {
        Cabana::migrate( *distributor, aosoa_0 );
        Cabana::migrate( *distributor, aosoa_1 );
    }

    void gather( std::shared_ptr<Cabana::Halo<t_device>> halo ) override
    {
        Cabana::gather( *halo, aosoa_0 );
        Cabana::gather( *halo, aosoa_1 );
    }

    const char *name() override { return "System:2AoSoA"; }

    AoSoA_2_0 get_aosoa_x() { return aosoa_0; }
    AoSoA_2_0 get_aosoa_f() { return aosoa_0; }
    AoSoA_2_0 get_aosoa_type() { return aosoa_0; }
    AoSoA_2_1 get_aosoa_v() { return aosoa_1; }
    AoSoA_2_1 get_aosoa_id() { return aosoa_1; }
    AoSoA_2_1 get_aosoa_q() { return aosoa_1; }
};
#endif
