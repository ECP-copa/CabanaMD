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

#ifndef SYSTEM_6AOSOA_H
#define SYSTEM_6AOSOA_H

#include <system.h>

class System6AoSoA : public System
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
    void migrate( std::shared_ptr<t_distributor> distributor );
    void gather( std::shared_ptr<t_halo> halo );

  private:
    AoSoA_x aosoa_x;
    AoSoA_x aosoa_v;
    AoSoA_x aosoa_f;
    AoSoA_int aosoa_id;
    AoSoA_int aosoa_type;
    AoSoA_fl aosoa_q;
};
#endif
