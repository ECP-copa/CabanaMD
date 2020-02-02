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

#include <system_nnp_3aosoa.h>

System_NNP_3AoSoA::System_NNP_3AoSoA()
{
    AoSoA_NNP_SF aosoa_G( "G", 0 );
    AoSoA_NNP_SF aosoa_dEdG( "dEdG", 0 );
    AoSoA_NNP_fl aosoa_E( "E", 0 );
}

void System_NNP_3AoSoA::resize( T_INT N_new )
{
    aosoa_G.resize( N_new );
    aosoa_dEdG.resize( N_new );
    aosoa_E.resize( N_new );
}

void System_NNP_3AoSoA::slice_G() { G = Cabana::slice<0>( aosoa_G ); }
void System_NNP_3AoSoA::slice_dEdG() { dEdG = Cabana::slice<0>( aosoa_dEdG ); }
void System_NNP_3AoSoA::slice_E() { E = Cabana::slice<0>( aosoa_E ); }
