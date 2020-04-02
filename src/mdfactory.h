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

#include <cabanamd.h>
#include <inputCL.h>
#include <system.h>

class MDfactory
{
  public:
    static CabanaMD *create( InputCL commandline )
    {
        if ( commandline.layout_type == AOSOA_1 )
            return new CbnMD<System<AoSoA1>>;
        else if ( commandline.layout_type == AOSOA_2 )
            return new CbnMD<System<AoSoA2>>;
        else if ( commandline.layout_type == AOSOA_6 )
            return new CbnMD<System<AoSoA6>>;
        return nullptr;
    }
};
