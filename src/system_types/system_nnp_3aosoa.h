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

#ifndef SYSTEM_NNP_3AOSOA_H
#define SYSTEM_NNP_3AOSOA_H

#include <CabanaMD_config.hpp>

#include <system_nnp.h>

template <class DeviceType>
class System_NNP<DeviceType, 3>
{
  public:
    using memory_space = typename DeviceType::memory_space;

  protected:
    using t_tuple_NNP_SF =
        Cabana::MemberTypes<T_FLOAT[CabanaMD_MAXSYMMFUNC_NNP]>;
    using t_tuple_NNP_fl = Cabana::MemberTypes<T_FLOAT>;
    using AoSoA_NNP_G = typename Cabana::AoSoA<t_tuple_NNP_SF, memory_space,
                                               CabanaMD_VECTORLENGTH_NNP_0>;
    using AoSoA_NNP_dEdG = typename Cabana::AoSoA<t_tuple_NNP_SF, memory_space,
                                                  CabanaMD_VECTORLENGTH_NNP_1>;
    using AoSoA_NNP_E = typename Cabana::AoSoA<t_tuple_NNP_fl, memory_space,
                                               CabanaMD_VECTORLENGTH_NNP_2>;
    AoSoA_NNP_G aosoa_G;
    AoSoA_NNP_dEdG aosoa_dEdG;
    AoSoA_NNP_E aosoa_E;

  public:
    using t_G =
        typename System_NNP<DeviceType,
                            3>::AoSoA_NNP_G::template member_slice_type<0>;
    using t_dEdG =
        typename System_NNP<DeviceType,
                            3>::AoSoA_NNP_dEdG::template member_slice_type<0>;
    using t_E =
        typename System_NNP<DeviceType,
                            3>::AoSoA_NNP_E::template member_slice_type<0>;
    t_G G;
    t_dEdG dEdG;
    t_E E;

    System_NNP<DeviceType, 3>()
    {
        AoSoA_NNP_G aosoa_G( "G", 0 );
        AoSoA_NNP_dEdG aosoa_dEdG( "dEdG", 0 );
        AoSoA_NNP_E aosoa_E( "E", 0 );
    }
    ~System_NNP<DeviceType, 3>() {}

    void resize( T_INT N_new )
    {
        aosoa_G.resize( N_new );
        aosoa_dEdG.resize( N_new );
        aosoa_E.resize( N_new );
    }

    void slice_G() { G = Cabana::slice<0>( aosoa_G ); }
    void slice_dEdG() { dEdG = Cabana::slice<0>( aosoa_dEdG ); }
    void slice_E() { E = Cabana::slice<0>( aosoa_E ); }

    const char *name() { return "NNPSystem:3AoSoA"; }
};
#endif
