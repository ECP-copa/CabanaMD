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

#ifndef SYSTEM_NNP_1AOSOA_H
#define SYSTEM_NNP_1AOSOA_H

#include <system_nnp.h>
#include <types_nnp.h>

#include <types.h>

template <>
class System_NNP<AoSoA1>
{
    using t_tuple_NNP =
        Cabana::MemberTypes<T_FLOAT[MAX_SF], T_FLOAT[MAX_SF], T_FLOAT>;
    using AoSoA_NNP_1 =
        Cabana::AoSoA<t_tuple_NNP, MemorySpace, CabanaMD_VECTORLENGTH_NNP>;
    AoSoA_NNP_1 aosoa_0;

  public:
    using t_G = AoSoA_NNP_1::member_slice_type<0>;
    using t_dEdG = AoSoA_NNP_1::member_slice_type<1>;
    using t_E = AoSoA_NNP_1::member_slice_type<2>;
    t_G G;
    t_dEdG dEdG;
    t_E E;

    System_NNP<AoSoA1>() { AoSoA_NNP_1 aosoa_0( "All", 0 ); }
    ~System_NNP<AoSoA1>(){}

    void resize( T_INT N_new ) { aosoa_0.resize( N_new ); }

    void slice_G() { G = Cabana::slice<0>( aosoa_0 ); }
    void slice_dEdG() { dEdG = Cabana::slice<1>( aosoa_0 ); }
    void slice_E() { E = Cabana::slice<2>( aosoa_0 ); }

    const char *name() { return "NNPSystem:1AoSoA"; }
};
#endif
