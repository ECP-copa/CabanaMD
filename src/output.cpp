/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <output.h>

#include <mpi.h>

bool print_rank()
{
    int proc_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &proc_rank );
    return proc_rank == 0;
}
