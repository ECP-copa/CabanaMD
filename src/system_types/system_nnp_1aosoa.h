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

#ifndef SYSTEM_NNP_1AOSOA_H
#define SYSTEM_NNP_1AOSOA_H

#include <CabanaMD_config.hpp>

#include <system_nnp.h>

template <class t_device>
class System_NNP<t_device, AoSoA1>
{
    using t_tuple_NNP =
        Cabana::MemberTypes<T_FLOAT[CabanaMD_MAXSYMMFUNC_NNP],
                            T_FLOAT[CabanaMD_MAXSYMMFUNC_NNP], T_FLOAT>;
    using AoSoA_NNP_1 = typename Cabana::AoSoA<t_tuple_NNP, t_device,
                                               CabanaMD_VECTORLENGTH_NNP>;
    AoSoA_NNP_1 aosoa_0;

  public:
    using t_G =
        typename System_NNP<t_device,
                            AoSoA1>::AoSoA_NNP_1::template member_slice_type<0>;
    using t_dEdG =
        typename System_NNP<t_device,
                            AoSoA1>::AoSoA_NNP_1::template member_slice_type<1>;
    using t_E =
        typename System_NNP<t_device,
                            AoSoA1>::AoSoA_NNP_1::template member_slice_type<2>;
    t_G G;
    t_dEdG dEdG;
    t_E E;

    System_NNP<t_device, AoSoA1>() { AoSoA_NNP_1 aosoa_0( "All", 0 ); }
    ~System_NNP<t_device, AoSoA1>() {}

    void resize( T_INT N_new ) { aosoa_0.resize( N_new ); }

    void slice_G() { G = Cabana::slice<0>( aosoa_0 ); }
    void slice_dEdG() { dEdG = Cabana::slice<1>( aosoa_0 ); }
    void slice_E() { E = Cabana::slice<2>( aosoa_0 ); }

    const char *name() { return "NNPSystem:1AoSoA"; }
};
#endif
