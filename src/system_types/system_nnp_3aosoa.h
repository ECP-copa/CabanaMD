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

#ifndef SYSTEM_NNP_3AOSOA_H
#define SYSTEM_NNP_3AOSOA_H

#include <system_nnp.h>
#include <types_nnp.h>

#include <types.h>

#include <CabanaMD_config.hpp>

#include <Cabana_Core.hpp>

template <>
class System_NNP<AoSoA3>
{
    using t_tuple_NNP_SF = Cabana::MemberTypes<T_FLOAT[MAX_SF]>;
    using t_tuple_NNP_fl = Cabana::MemberTypes<T_FLOAT>;
    using AoSoA_NNP_SF =
        Cabana::AoSoA<t_tuple_NNP_SF, MemorySpace, CabanaMD_VECTORLENGTH_NNP>;
    using AoSoA_NNP_fl =
        Cabana::AoSoA<t_tuple_NNP_fl, MemorySpace, CabanaMD_VECTORLENGTH_NNP>;
    AoSoA_NNP_SF aosoa_G;
    AoSoA_NNP_SF aosoa_dEdG;
    AoSoA_NNP_fl aosoa_E;

  public:
    using t_G = AoSoA_NNP_SF::member_slice_type<0>;
    using t_dEdG = AoSoA_NNP_SF::member_slice_type<0>;
    using t_E = AoSoA_NNP_fl::member_slice_type<0>;
    t_G G;
    t_dEdG dEdG;
    t_E E;

    System_NNP()
    {
        AoSoA_NNP_SF aosoa_G( "G", 0 );
        AoSoA_NNP_SF aosoa_dEdG( "dEdG", 0 );
        AoSoA_NNP_fl aosoa_E( "E", 0 );
    }
    ~System_NNP() {}

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
