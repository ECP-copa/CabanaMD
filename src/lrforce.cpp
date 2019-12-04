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

#include<lrforce.h>

LRForce::LRForce(System*, bool half_neigh_):half_neigh(half_neigh_) {}

void LRForce::init_coeff(T_X_FLOAT, char**) {}
void LRForce::create_neigh_list(System*) {}
void LRForce::compute(System*) {}
const char* LRForce::name() { return "LRForceNone"; }

