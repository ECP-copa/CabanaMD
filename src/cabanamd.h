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

#ifndef CABANAMD_H
#define CABANAMD_H

#include <binning_cabana.h>
#include <comm_mpi.h>
#include <force.h>
#include <inputCL.h>
#include <inputFile.h>
#include <integrator_nve.h>
#include <types.h>

#ifdef CabanaMD_ENABLE_LB
#include <Cajita_LoadBalancer.hpp>
#include <Cajita_Types.hpp>
#endif

class CabanaMD
{
  public:
    bool _print_lammps = false;
    int nsteps;

    virtual void init( InputCL cl ) = 0;
    virtual void run() = 0;

    virtual void dump_binary( int ) = 0;
    virtual void check_correctness( int ) = 0;
};

template <class t_System, class t_Neighbor>
class CbnMD : public CabanaMD
{
  public:
    t_System *system;
    t_Neighbor *neighbor;
    Force<t_System, t_Neighbor> *force;
    Integrator<t_System> *integrator;
    Comm<t_System> *comm;
    Binning<t_System> *binning;
    InputFile<t_System> *input;
#ifdef CabanaMD_ENABLE_LB
    Cajita::Experimental::LoadBalancer<Cajita::UniformMesh<double>> *lb;
#endif

    void init( InputCL cl ) override;
    void run() override;

    void dump_binary( int ) override;
    void check_correctness( int ) override;
};

#include <cabanamd_impl.h>
#endif
