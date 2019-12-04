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

#ifndef FORCE_H
#define FORCE_H
#include<types.h>
#include<system.h>

class LRForce {
public:
  bool half_neigh, comm_newton;
  Force(System* system, bool half_neigh_);

  virtual void init_coeff(T_X_FLOAT neigh_cut, char** args);
  virtual void create_neigh_list(System* system);

  virtual void compute(System* system);
  virtual T_F_FLOAT compute_energy(System*){return 0.0;} // Only needed for thermo output

  virtual const char* name();
};

#include<modules_force.h>
#endif
