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

// Include Module header files for system
#ifdef CabanaMD_LAYOUT_1AoSoA
#include <system_1aosoa.h>
#elif defined( CabanaMD_LAYOUT_2AoSoA )
#include <system_2aosoa.h>
#elif defined( CabanaMD_LAYOUT_6AoSoA )
#include <system_6aosoa.h>
#endif
