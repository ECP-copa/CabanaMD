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

#ifndef SYSTEM_2AOSOA_H
#define SYSTEM_2AOSOA_H

#include <system.h>

class System2AoSoA : public System
{
  public:
    using System::System;

    void init();

    void resize( T_INT new_N );

    void slice_x();
    void slice_v();
    void slice_f();
    void slice_type();
    void slice_id();
    void slice_q();

    void permute( t_linkedcell linkedcell );
    void migrate( t_distributor distributor );
    void gather( t_halo halo );

  private:
    AoSoA_2_0 aosoa_0;
    AoSoA_2_1 aosoa_1;
};
#endif
