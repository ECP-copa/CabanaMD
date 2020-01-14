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

#include <force_spme_cabana_neigh_impl.h>

template class ForceSPME<t_verletlist_half_2D>;
template class ForceSPME<t_verletlist_full_2D>;
template class ForceSPME<t_verletlist_half_CSR>;
template class ForceSPME<t_verletlist_full_CSR>;
