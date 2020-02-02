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

#ifndef SYSTEM_NNP_H
#define SYSTEM_NNP_H

#include <types_nnp.h>

#include <types.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

class System_NNP
{
  public:
    // Per Particle Property
    t_G G;
    t_dEdG dEdG;
    t_E E;

    virtual void resize( T_INT new_N ) {}

    virtual void slice_G() {}
    virtual void slice_dEdG() {}
    virtual void slice_E() {}
};

#include <modules_system_nnp.h>
#endif
