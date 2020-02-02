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

#ifndef SYSTEM_NNP_3AOSOA_H
#define SYSTEM_NNP_3AOSOA_H

#include <system_nnp.h>
#include <types_nnp.h>

#include <types.h>

class System_NNP_3AoSoA : public System_NNP
{
  public:
    System_NNP_3AoSoA();
    ~System_NNP_3AoSoA(){};

    void resize( T_INT new_N );

    void slice_G();
    void slice_dEdG();
    void slice_E();

  private:
    AoSoA_NNP_SF aosoa_G;
    AoSoA_NNP_SF aosoa_dEdG;
    AoSoA_NNP_fl aosoa_E;
};
#endif
