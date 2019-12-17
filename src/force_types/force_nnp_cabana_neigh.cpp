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

#include<force_nnp_cabana_neigh_impl.h>

template class ForceNNP<t_verletlist_full_2D, t_neighborop_serial, t_neighborop_serial>;
template class ForceNNP<t_verletlist_full_CSR, t_neighborop_serial, t_neighborop_serial>;
template class ForceNNP<t_verletlist_full_2D, t_neighborop_team, t_neighborop_team>;
template class ForceNNP<t_verletlist_full_CSR, t_neighborop_team, t_neighborop_team>;
template class ForceNNP<t_verletlist_full_2D, t_neighborop_team, t_neighborop_vector>;
template class ForceNNP<t_verletlist_full_CSR, t_neighborop_team, t_neighborop_vector>;
