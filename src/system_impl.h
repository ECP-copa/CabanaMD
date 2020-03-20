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

#include <system.h>

template <class t_layout>
SystemCommon<t_layout>::SystemCommon()
{
    N = 0;
    N_max = 0;
    N_local = 0;
    N_ghost = 0;
    ntypes = 1;
    atom_style = "atomic";

    mass = t_mass();
    domain_x = domain_y = domain_z = 0.0;
    sub_domain_x = sub_domain_y = sub_domain_z = 0.0;
    domain_lo_x = domain_lo_y = domain_lo_z = 0.0;
    domain_hi_x = domain_hi_y = domain_hi_z = 0.0;
    sub_domain_hi_x = sub_domain_hi_y = sub_domain_hi_z = 0.0;
    sub_domain_lo_x = sub_domain_lo_y = sub_domain_lo_z = 0.0;
    mvv2e = boltz = dt = 0.0;

    print_lammps = false;

    mass = t_mass( "System::mass", ntypes );

    int proc_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &proc_rank );
    do_print = proc_rank == 0;
}

template <class t_layout>
const char *SystemCommon<t_layout>::name()
{
    return "SystemNone";
}

template <class t_layout>
void SystemCommon<t_layout>::slice_all()
{
    slice_x();
    slice_v();
    slice_f();
    slice_type();
    slice_id();
    slice_q();
}

template <class t_layout>
void SystemCommon<t_layout>::slice_integrate()
{
    slice_x();
    slice_v();
    slice_f();
    slice_type();
}

template <class t_layout>
void SystemCommon<t_layout>::slice_force()
{
    slice_x();
    slice_f();
    slice_type();
}

template <class t_layout>
void SystemCommon<t_layout>::slice_properties()
{
    slice_v();
    slice_type();
}
