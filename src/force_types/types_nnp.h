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

//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//************************************************************************

#ifndef TYPES_NNP_H
#define TYPES_NNP_H

#include <types.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

// TODO: hardcoded
#define MAX_SF 30
constexpr double CFLENGTH = 1.889726;
constexpr double CFENERGY = 0.036749;
constexpr double CFFORCE = CFLENGTH / CFENERGY;

enum ScalingType {
  ST_NONE,
  ST_SCALE,
  ST_CENTER,
  ST_SCALECENTER,
  ST_SCALESIGMA
};

typedef ExecutionSpace::array_layout array_layout; // TODO: check this
using h_t_mass = Kokkos::View<T_V_FLOAT *, array_layout, Kokkos::HostSpace>;
using d_t_SF = Kokkos::View<T_FLOAT * * [15]>;
using t_SF = Kokkos::View<T_FLOAT * * [15], array_layout, Kokkos::HostSpace>;
using d_t_SFscaling = Kokkos::View<T_FLOAT * * [8]>;
using t_SFscaling =
    Kokkos::View<T_FLOAT * * [8], array_layout, Kokkos::HostSpace>;
using d_t_SFGmemberlist =
    Kokkos::View<T_INT *
                 [MAX_SF + 1][MAX_SF + 1]>; //+1 to store size of memberlist
using t_SFGmemberlist = Kokkos::View<T_INT * [MAX_SF + 1][MAX_SF + 1],
                                     array_layout, Kokkos::HostSpace>;

using d_t_bias = Kokkos::View<T_FLOAT ***>;
using t_bias = Kokkos::View<T_FLOAT ***, array_layout, Kokkos::HostSpace>;
using d_t_weights = Kokkos::View<T_FLOAT ****>;
using t_weights = Kokkos::View<T_FLOAT ****, array_layout, Kokkos::HostSpace>;
using d_t_NN = Kokkos::View<T_FLOAT ***>;

#ifdef CabanaMD_LAYOUT_NNP_1AoSoA
using t_tuple_NNP =
    Cabana::MemberTypes<T_FLOAT[MAX_SF], T_FLOAT[MAX_SF], T_FLOAT>;
using AoSoA_NNP_1 =
    Cabana::AoSoA<t_tuple_NNP, MemorySpace, CabanaMD_VECTORLENGTH_NNP>;
using t_G = AoSoA_NNP_1::member_slice_type<0>;
using t_dEdG = AoSoA_NNP_1::member_slice_type<1>;
using t_E = AoSoA_NNP_1::member_slice_type<2>;

#elif defined(CabanaMD_LAYOUT_NNP_3AoSoA)
using t_tuple_NNP_SF = Cabana::MemberTypes<T_FLOAT[MAX_SF]>;
using t_tuple_NNP_fl = Cabana::MemberTypes<T_FLOAT>;
using AoSoA_NNP_SF =
    Cabana::AoSoA<t_tuple_NNP_SF, MemorySpace, CabanaMD_VECTORLENGTH_NNP>;
using AoSoA_NNP_fl =
    Cabana::AoSoA<t_tuple_NNP_fl, MemorySpace, CabanaMD_VECTORLENGTH_NNP>;
using t_G = AoSoA_NNP_SF::member_slice_type<0>;
using t_dEdG = AoSoA_NNP_SF::member_slice_type<0>;
using t_E = AoSoA_NNP_fl::member_slice_type<0>;
#endif

#endif // TYPES_NNP_H
