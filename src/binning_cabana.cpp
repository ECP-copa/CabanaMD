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

//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright notice,
//       this list of conditions and the following disclaimer in the documentation
//       and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//************************************************************************

#include<binning_cabana.h>

Binning::Binning(System* s):system(s) {}

void Binning::create_binning(T_X_FLOAT dx_in, T_X_FLOAT dy_in, T_X_FLOAT dz_in, int halo_depth, bool do_local, bool do_ghost, bool sort) {
  if(do_local||do_ghost) {
    nhalo = halo_depth;
    int begin = do_local?0:system->N_local;
    int end = do_ghost?system->N_local+system->N_ghost:system->N_local;

    nbinx = T_INT(system->sub_domain_x/dx_in);
    nbiny = T_INT(system->sub_domain_y/dy_in);
    nbinz = T_INT(system->sub_domain_z/dz_in);

    if(nbinx == 0) nbinx = 1;
    if(nbiny == 0) nbiny = 1;
    if(nbinz == 0) nbinz = 1;

    T_X_FLOAT dx = system->sub_domain_x/nbinx;
    T_X_FLOAT dy = system->sub_domain_y/nbiny;
    T_X_FLOAT dz = system->sub_domain_z/nbinz;

    T_X_FLOAT eps = dx/1000;
    minx = -dx * halo_depth - eps + system->sub_domain_lo_x;
    maxx =  dx * halo_depth + eps + system->sub_domain_hi_x;
    miny = -dy * halo_depth - eps + system->sub_domain_lo_y;
    maxy =  dy * halo_depth + eps + system->sub_domain_hi_y;
    minz = -dz * halo_depth - eps + system->sub_domain_lo_z;
    maxz =  dz * halo_depth + eps + system->sub_domain_hi_z;

    T_X_FLOAT delta[3] = {dx,dy,dz};
    T_X_FLOAT min[3] = {minx,miny,minz};
    T_X_FLOAT max[3] = {maxx,maxy,maxz};

    x = Cabana::slice<0>(system->aosoa_x);

    t_linkedcell cell_list(x, begin, end, delta, min, max );

    if(sort) {
      Cabana::permute( cell_list, system->aosoa_x );
      Cabana::permute( cell_list, system->aosoa_v );
      Cabana::permute( cell_list, system->aosoa_f );
      Cabana::permute( cell_list, system->aosoa_id );
      Cabana::permute( cell_list, system->aosoa_type );
      Cabana::permute( cell_list, system->aosoa_q );
    }
  }
}

const char* Binning::name() { return "Binning:CabanaLinkedCell"; }

