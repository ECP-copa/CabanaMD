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

#ifndef SYSTEM_6AOSOA_H
#define SYSTEM_6AOSOA_H

#include <CabanaMD_config.hpp>

#include <system.h>

template <class DeviceType>
class System<DeviceType, 6> : public SystemCommon<DeviceType>
{
  public:
    using SystemCommon<DeviceType>::SystemCommon;

    using memory_space = typename DeviceType::memory_space;
    using execution_space = typename DeviceType::execution_space;

  protected:
    using t_tuple_x = Cabana::MemberTypes<T_FLOAT[3]>;
    using t_tuple_int = Cabana::MemberTypes<T_INT>;
    using t_tuple_fl = Cabana::MemberTypes<T_FLOAT>;
    using AoSoA_x = typename Cabana::AoSoA<t_tuple_x, memory_space,
                                           CabanaMD_VECTORLENGTH_0>;
    using AoSoA_v = typename Cabana::AoSoA<t_tuple_x, memory_space,
                                           CabanaMD_VECTORLENGTH_1>;
    using AoSoA_f = typename Cabana::AoSoA<t_tuple_x, memory_space,
                                           CabanaMD_VECTORLENGTH_2>;
    using AoSoA_id = typename Cabana::AoSoA<t_tuple_int, memory_space,
                                            CabanaMD_VECTORLENGTH_3>;
    using AoSoA_type = typename Cabana::AoSoA<t_tuple_int, memory_space,
                                              CabanaMD_VECTORLENGTH_4>;
    using AoSoA_q = typename Cabana::AoSoA<t_tuple_fl, memory_space,
                                           CabanaMD_VECTORLENGTH_5>;
    AoSoA_x aosoa_x;
    AoSoA_v aosoa_v;
    AoSoA_f aosoa_f;
    AoSoA_id aosoa_id;
    AoSoA_type aosoa_type;
    AoSoA_q aosoa_q;

    using SystemCommon<DeviceType>::N_max;
    // using SystemCommon<DeviceType>::mass;

  public:
    // Per Particle Property
    using t_x =
        typename System<DeviceType, 6>::AoSoA_x::template member_slice_type<0>;
    using t_v =
        typename System<DeviceType, 6>::AoSoA_v::template member_slice_type<0>;
    using t_f =
        typename System<DeviceType, 6>::AoSoA_f::template member_slice_type<0>;
    using t_type =
        typename System<DeviceType,
                        6>::AoSoA_type::template member_slice_type<0>;
    using t_id =
        typename System<DeviceType, 6>::AoSoA_id::template member_slice_type<0>;
    using t_q =
        typename System<DeviceType, 6>::AoSoA_q::template member_slice_type<0>;

    t_x x;
    t_v v;
    t_f f;
    t_type type;
    t_id id;
    t_q q;

    void init() override
    {
        AoSoA_x aosoa_x( "X", N_max );
        AoSoA_v aosoa_v( "V", N_max );
        AoSoA_f aosoa_f( "F", N_max );
        AoSoA_id aosoa_id( "ID", N_max );
        AoSoA_type aosoa_type( "Type", N_max );
        AoSoA_q aosoa_q( "Q", N_max );
    }

    void resize( T_INT N_new ) override
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

    template <class SrcSystem>
    void deep_copy( SrcSystem src_system )
    {
        Cabana::deep_copy( aosoa_x, src_system.get_aosoa_x() );
        Cabana::deep_copy( aosoa_v, src_system.get_aosoa_v() );
        Cabana::deep_copy( aosoa_f, src_system.get_aosoa_f() );
        Cabana::deep_copy( aosoa_id, src_system.get_aosoa_id() );
        Cabana::deep_copy( aosoa_type, src_system.get_aosoa_type() );
        Cabana::deep_copy( aosoa_q, src_system.get_aosoa_q() );
    }

    void slice_x() override { x = Cabana::slice<0>( aosoa_x ); }
    void slice_v() override { v = Cabana::slice<0>( aosoa_v ); }
    void slice_f() override { f = Cabana::slice<0>( aosoa_f ); }
    void slice_type() override { type = Cabana::slice<0>( aosoa_type ); }
    void slice_id() override { id = Cabana::slice<0>( aosoa_id ); }
    void slice_q() override { q = Cabana::slice<0>( aosoa_q ); }

    void permute( Cabana::LinkedCellList<memory_space> cell_list ) override
    {
        Cabana::permute( cell_list, aosoa_x );
        Cabana::permute( cell_list, aosoa_v );
        Cabana::permute( cell_list, aosoa_f );
        Cabana::permute( cell_list, aosoa_type );
        Cabana::permute( cell_list, aosoa_id );
        Cabana::permute( cell_list, aosoa_q );
    }

    void migrate( std::shared_ptr<Cabana::Distributor<memory_space>>
                      distributor ) override
    {
        Cabana::migrate( *distributor, aosoa_x );
        Cabana::migrate( *distributor, aosoa_v );
        Cabana::migrate( *distributor, aosoa_f );
        Cabana::migrate( *distributor, aosoa_type );
        Cabana::migrate( *distributor, aosoa_id );
        Cabana::migrate( *distributor, aosoa_q );
    }

    void gather( std::shared_ptr<Cabana::Halo<memory_space>> halo ) override
    {
        Cabana::gather( *halo, aosoa_x );
        Cabana::gather( *halo, aosoa_v );
        Cabana::gather( *halo, aosoa_f );
        Cabana::gather( *halo, aosoa_type );
        Cabana::gather( *halo, aosoa_id );
        Cabana::gather( *halo, aosoa_q );
    }

    const char *name() override { return "System:6AoSoA"; }

    AoSoA_x get_aosoa_x() { return aosoa_x; }
    AoSoA_v get_aosoa_v() { return aosoa_v; }
    AoSoA_f get_aosoa_f() { return aosoa_f; }
    AoSoA_type get_aosoa_type() { return aosoa_type; }
    AoSoA_id get_aosoa_id() { return aosoa_id; }
    AoSoA_q get_aosoa_q() { return aosoa_q; }
};

#endif
