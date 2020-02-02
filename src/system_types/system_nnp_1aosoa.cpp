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

#include <system_nnp_1aosoa.h>

System_NNP_1AoSoA::System_NNP_1AoSoA() { AoSoA_NNP_1 aosoa_0( "All", 0 ); }

void System_NNP_1AoSoA::resize( T_INT N_new ) { aosoa_0.resize( N_new ); }

void System_NNP_1AoSoA::slice_G() { G = Cabana::slice<0>( aosoa_0 ); }
void System_NNP_1AoSoA::slice_dEdG() { dEdG = Cabana::slice<1>( aosoa_0 ); }
void System_NNP_1AoSoA::slice_E() { E = Cabana::slice<2>( aosoa_0 ); }
