/****************************************************************************
 * Copyright (c) 2018 by the Cabana authors                                 *
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

#include <input.h>
#include <property_temperature.h>

Input::Input(System* p):system(p) {

  //#ifdef CABANAMD_ENABLE_MPI
  //comm_type = COMM_MPI;
  //#else
  comm_type = COMM_SERIAL;
  //#endif
  integrator_type = INTEGRATOR_NVE;
  neighbor_type = NEIGH_CABANA_VERLET;
  force_type = FORCE_LJ_CABANA_NEIGH;
  force_iteration_type = FORCE_ITER_NEIGH_FULL;
  binning_type = BINNING_CABANA;

  // set defaults (matches ExaMiniMD LJ example)
  thermo_rate = 0;
  dumpbinary_rate = 0;
  correctness_rate = 0;
  dumpbinaryflag = false;
  correctnessflag = false;
  timestepflag = false;

  lattice_offset_x = 0.0;
  lattice_offset_y = 0.0;
  lattice_offset_z = 0.0;
  box[0] = 0;
  box[2] = 0;
  box[4] = 0;
  box[1] = 40;
  box[3] = 40;
  box[5] = 40;

  units_style = UNITS_LJ;
  lattice_style = LATTICE_FCC;
  lattice_constant = 0.8442;

  temperature_target = 1.4;
  temperature_seed = 87287;

  nsteps = 100;
  thermo_rate = 10;
  //timestep = 1;

  neighbor_skin = 0.3;
  comm_exchange_rate = 20;
  comm_newton = 0;

  ntypes = 1;
  mass_vec = {2.0};
  force_types = {{1, 1}};
  force_coeff = {{1.0, 1.0, 2.5}};
  force_cutoff = 2.5;
}

void Input::read_command_line_args(int argc, char* argv[]) {
  for(int i = 1; i < argc; i++) {
    // Help command
    if( (strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0) ) {
      if(system->do_print) {
        printf("CabanaMD 0.1 \n\n");
        printf("Options:\n");
        printf("  --units-style [TYPE]:       Specify units \n");
        printf("                              (metal, real, lj)\n");
        printf("  --lattice [TYPE] [CONSTANT] [NX] [NY] [NZ]:    Specify lattice \n");
        printf("                              (TYPE = fcc)\n");
        printf("                              (CONSTANT = float lattice constant)\n");
        printf("                              (NX, NY, NZ = integer unit cells per dimension)\n");
        printf("  --temperature [TARGET] [SEED]:    System temperature \n");
        printf("                              (TARGET = float temperature (K))\n");
        printf("                              (SEED = random seed)\n");
        printf("  --run [STEPS] [PRINT]:      simulation run parameters \n");
        printf("                              (STEPS = number of timesteps)\n");
        printf("                              (PRINT = thermo print frequency)\n");
        printf("  --neighbor [SKIN] [EXCHANGE] [NEWTON]:    neighbor list parameters \n");
        printf("                              (SKIN = neighbor list skin distance)\n");
        printf("                              (EXCHANGE = neighbor list build frequency)\n");
        printf("                              (NEWTON = (0 or 1) full or half neighbor lists)\n");
        printf("  --pair-coeff [N] [CUTOFF] [MASS]xN [COEFF]x(N**2+N)/2x4:    Per type/pair parameters \n");
        printf("                              (N = number of types)\n");
        printf("                              (CUTOFF = force distance cutoff)\n");
        printf("                              (MASS = One mass per type)\n");
	printf("                              (COEFF = Two types and two force coefficients per type (type1, type2, epsilon, sigma))\n");
	printf("  --dumpbinary [N] [PATH]:    Request that binary output files PATH/output* be generated every N steps\n");
        printf("                              (N = positive integer)\n");
        printf("                              (PATH = location of directory)\n");
        printf("  --correctness [N] [PATH] [FILE]:   Request that correctness check against files PATH/output* be performed every N steps, correctness data written to FILE\n");
        printf("                              (N = positive integer)\n");
        printf("                              (PATH = location of directory)\n");
      }
    }

    // Dump Binary
    else if( (strcmp(argv[i], "--dumpbinary") == 0) ) {
      dumpbinary_rate = atoi(argv[i+1]);
      dumpbinary_path = argv[i+2];
      dumpbinaryflag = true;
      i += 2;
    }
    
    // Correctness Check
    else if( (strcmp(argv[i], "--correctness") == 0) ) {
      correctness_rate = atoi(argv[i+1]);
      reference_path = argv[i+2];
      correctness_file = argv[i+3];
      correctnessflag = true;
      i += 3;
    }

    else if( (strcmp(argv[i],"--units-type")==0)) {
      if(strcmp(argv[i+1],"metal")==0) {
	units_style = UNITS_METAL;
      } else if(strcmp(argv[i+1],"real")==0) {
        units_style = UNITS_REAL;
      } else if(strcmp(argv[i+1],"lj")==0) {
        units_style = UNITS_LJ;
      }
      ++i;
    }

    else if( (strcmp(argv[i],"--lattice")==0)) {
      if(strcmp(argv[i+1],"fcc")==0) {
        lattice_style = LATTICE_FCC;
      } else if(strcmp(argv[i+1],"sc")==0) {
        lattice_style = LATTICE_SC;
      }
      lattice_constant = atof(argv[i+2]);

      box[1] = atoi(argv[i+3]);
      box[3] = atoi(argv[i+4]);
      box[5] = atoi(argv[i+5]);
      i += 5;
    }

    else if( (strcmp(argv[i],"--temperature")==0)) {
      temperature_target = atof(argv[i+1]);
      temperature_seed = atoi(argv[i+2]);
      i += 2;
    }

    else if( (strcmp(argv[i],"--run")==0)) {
      nsteps = atoi(argv[i+1]);
      thermo_rate = atoi(argv[i+2]);
      //timestep = atof(argv[i+3]); // determined by units_type
      i += 2;
    }

    else if( (strcmp(argv[i],"--neighbor")==0)) {
      neighbor_skin = atof(argv[i+1]);
      comm_exchange_rate = atoi(argv[i+2]);
      comm_newton = atoi(argv[i+3]);
      if (comm_newton == 0)
        force_iteration_type = FORCE_ITER_NEIGH_FULL;
      else
	force_iteration_type = FORCE_ITER_NEIGH_HALF;
      i += 3;
    }

    else if( (strcmp(argv[i],"--pair-coeff")==0)) {
      ntypes = atoi(argv[i+1]);
      force_cutoff = atof(argv[i+2]);
      i += 2;

      int nforce = (pow(ntypes, 2.0) + ntypes)/2;
      force_types.resize(nforce, std::vector<int>(2));
      force_coeff.resize(nforce, std::vector<double>(2));

      for (int type = 0; type < ntypes; type++){
        mass_vec[type] = atof(argv[i]);
        ++i;
      }
      for (int force = 0; force < nforce; force++){
        for (int type = 0; type < 2; type++){
          i++;
          force_types[force][type] = atoi(argv[i]);
        }
        for (int coeff = 0; coeff < 2; coeff++){
          i++;
          force_coeff[force][coeff] = atof(argv[i]);
        }
      }
    }
  
    else if( (strstr(argv[i], "--kokkos-") == NULL) ) {
      if(system->do_print)
        printf("ERROR: Unknown command line argument: %s\n",argv[i]);
      exit(1);
    }

  }

  // Constants dependent on units
  if (units_style == UNITS_METAL) {
    system->boltz = 8.617343e-5;
    //hplanck = 95.306976368;
    system->mvv2e = 1.0364269e-4;
    if (!timestepflag)
      system->dt = 0.001;
  } else if (units_style == UNITS_REAL) {
    system->boltz = 0.0019872067;
    //hplanck = 95.306976368;
    system->mvv2e = 48.88821291 * 48.88821291;
    if (!timestepflag)
      system->dt = 1.0;
  } else if (units_style == UNITS_LJ) {
    system->boltz = 1.0;
    //hplanck = 0.18292026;
    system->mvv2e = 1.0;
    if (!timestepflag)
      system->dt = 0.005;
  }

  lattice_nx = box[1];
  lattice_ny = box[3];
  lattice_nz = box[5];
  if(lattice_style == LATTICE_FCC)
    lattice_constant = std::pow((4.0 / lattice_constant), (1.0 / 3.0));

  //if (timestepflag)
  //  system->dt = timestep;

  system->ntypes = ntypes;
  system->mass = t_mass("System::mass",system->ntypes);
  for (int type = 0; type < ntypes; type++){
    Kokkos::View<T_V_FLOAT> mass_one(system->mass,type);
    T_V_FLOAT mass = mass_vec[type];
    Kokkos::deep_copy(mass_one,mass);
  }
}

void Input::create_lattice(Comm* comm) {

  System s = *system;

  t_x::HostMirror h_x = Kokkos::create_mirror_view(s.x);
  t_v::HostMirror h_v = Kokkos::create_mirror_view(s.v);
  t_q::HostMirror h_q = Kokkos::create_mirror_view(s.q);
  t_mass::HostMirror h_mass = Kokkos::create_mirror_view(s.mass);
  t_type::HostMirror h_type = Kokkos::create_mirror_view(s.type);
  t_id::HostMirror h_id = Kokkos::create_mirror_view(s.id);

  Kokkos::deep_copy(h_mass,s.mass);

  // Create Simple Cubic Lattice
  if(lattice_style == LATTICE_SC) {
    system->domain_x = lattice_constant * lattice_nx; 
    system->domain_y = lattice_constant * lattice_ny; 
    system->domain_z = lattice_constant * lattice_nz; 

    comm->create_domain_decomposition();
    s = *system;

    T_INT ix_start = s.sub_domain_lo_x/s.domain_x * lattice_nx - 0.5;
    T_INT iy_start = s.sub_domain_lo_y/s.domain_y * lattice_ny - 0.5;
    T_INT iz_start = s.sub_domain_lo_z/s.domain_z * lattice_nz - 0.5;

    T_INT ix_end = s.sub_domain_hi_x/s.domain_x * lattice_nx + 0.5;
    T_INT iy_end = s.sub_domain_hi_y/s.domain_y * lattice_ny + 0.5;
    T_INT iz_end = s.sub_domain_hi_z/s.domain_z * lattice_nz + 0.5;

    T_INT n = 0;

    for(T_INT iz=iz_start; iz<=iz_end; iz++) {
      T_FLOAT ztmp = lattice_constant * (iz+lattice_offset_z);
      for(T_INT iy=iy_start; iy<=iy_end; iy++) {
        T_FLOAT ytmp = lattice_constant * (iy+lattice_offset_y);
        for(T_INT ix=ix_start; ix<=ix_end; ix++) {
          T_FLOAT xtmp = lattice_constant * (ix+lattice_offset_x);
          if((xtmp >= s.sub_domain_lo_x) &&
             (ytmp >= s.sub_domain_lo_y) &&
             (ztmp >= s.sub_domain_lo_z) &&
             (xtmp <  s.sub_domain_hi_x) &&
             (ytmp <  s.sub_domain_hi_y) &&
             (ztmp <  s.sub_domain_hi_z) ) {
            n++;
          }
        }
      }
    }
    system->N_local = n;
    system->N = n;
    system->grow(n);
    s = *system;
    h_x = Kokkos::create_mirror_view(s.x);
    h_v = Kokkos::create_mirror_view(s.v);
    h_q = Kokkos::create_mirror_view(s.q);
    h_type = Kokkos::create_mirror_view(s.type);
    h_id = Kokkos::create_mirror_view(s.id);

    // Initialize system using the equivalent of the LAMMPS
    // velocity geom option, i.e. uniform random kinetic energies.
    // zero out momentum of the whole system afterwards, to eliminate
    // drift (bad for energy statistics)

    for(T_INT iz=iz_start; iz<=iz_end; iz++) {
      T_FLOAT ztmp = lattice_constant * (iz+lattice_offset_z);
      for(T_INT iy=iy_start; iy<=iy_end; iy++) {
        T_FLOAT ytmp = lattice_constant * (iy+lattice_offset_y);
        for(T_INT ix=ix_start; ix<=ix_end; ix++) {
          T_FLOAT xtmp = lattice_constant * (ix+lattice_offset_x);
          if((xtmp >= s.sub_domain_lo_x) &&
             (ytmp >= s.sub_domain_lo_y) &&
             (ztmp >= s.sub_domain_lo_z) &&
             (xtmp <  s.sub_domain_hi_x) &&
             (ytmp <  s.sub_domain_hi_y) &&
             (ztmp <  s.sub_domain_hi_z) ) {
            n++;
          }
        }
      }
    }
    system->grow(n);
    System s = *system;
    h_x = Kokkos::create_mirror_view(s.x);
    h_v = Kokkos::create_mirror_view(s.v);
    h_q = Kokkos::create_mirror_view(s.q);
    h_type = Kokkos::create_mirror_view(s.type);
    h_id = Kokkos::create_mirror_view(s.id);

    n = 0;

    // Initialize system using the equivalent of the LAMMPS
    // velocity geom option, i.e. uniform random kinetic energies.
    // zero out momentum of the whole system afterwards, to eliminate
    // drift (bad for energy statistics)

    for(T_INT iz=iz_start; iz<=iz_end; iz++) {
      T_FLOAT ztmp = lattice_constant * (iz+lattice_offset_z);
      for(T_INT iy=iy_start; iy<=iy_end; iy++) {
        T_FLOAT ytmp = lattice_constant * (iy+lattice_offset_y);
        for(T_INT ix=ix_start; ix<=ix_end; ix++) {
          T_FLOAT xtmp = lattice_constant * (ix+lattice_offset_x);
          if((xtmp >= s.sub_domain_lo_x) &&
             (ytmp >= s.sub_domain_lo_y) &&
             (ztmp >= s.sub_domain_lo_z) &&
             (xtmp <  s.sub_domain_hi_x) &&
             (ytmp <  s.sub_domain_hi_y) &&
             (ztmp <  s.sub_domain_hi_z) ) {
            h_x(n,0) = xtmp;
            h_x(n,1) = ytmp;
            h_x(n,2) = ztmp;
            h_type(n) = rand()%s.ntypes;
            h_id(n) = n+1;
            n++;
          }
        }
      }
    }
    comm->reduce_int(&system->N,1);

    // Make ids unique over all processes
    T_INT N_local_offset = n;
    comm->scan_int(&N_local_offset,1);
    for(T_INT i = 0; i<n; i++)
      h_id(i) += N_local_offset - n;

    if(system->do_print)
      printf("Atoms: %i %i\n",system->N,system->N_local);
  }

  // Create Face Centered Cubic (FCC) Lattice
  if(lattice_style == LATTICE_FCC) {
    system->domain_x = lattice_constant * lattice_nx;
    system->domain_y = lattice_constant * lattice_ny;
    system->domain_z = lattice_constant * lattice_nz;

    comm->create_domain_decomposition();
    s = *system;

    double basis[4][3];
    basis[0][0] = 0.0; basis[0][1] = 0.0; basis[0][2] = 0.0;
    basis[1][0] = 0.5; basis[1][1] = 0.5; basis[1][2] = 0.0;
    basis[2][0] = 0.5; basis[2][1] = 0.0; basis[2][2] = 0.5;
    basis[3][0] = 0.0; basis[3][1] = 0.5; basis[3][2] = 0.5;

    for (int i = 0; i < 4; i++) {
      basis[i][0] += lattice_offset_x;
      basis[i][1] += lattice_offset_y;
      basis[i][2] += lattice_offset_z;
    }

    T_INT ix_start = s.sub_domain_lo_x/s.domain_x * lattice_nx - 0.5;
    T_INT iy_start = s.sub_domain_lo_y/s.domain_y * lattice_ny - 0.5;
    T_INT iz_start = s.sub_domain_lo_z/s.domain_z * lattice_nz - 0.5;

    T_INT ix_end = s.sub_domain_hi_x/s.domain_x * lattice_nx + 0.5;
    T_INT iy_end = s.sub_domain_hi_y/s.domain_y * lattice_ny + 0.5;
    T_INT iz_end = s.sub_domain_hi_z/s.domain_z * lattice_nz + 0.5;

    T_INT n = 0;

    for(T_INT iz=iz_start; iz<=iz_end; iz++) {
      for(T_INT iy=iy_start; iy<=iy_end; iy++) {
        for(T_INT ix=ix_start; ix<=ix_end; ix++) {
          for(int k = 0; k<4; k++) {
            T_FLOAT xtmp = lattice_constant * (1.0*ix+basis[k][0]);
            T_FLOAT ytmp = lattice_constant * (1.0*iy+basis[k][1]);
            T_FLOAT ztmp = lattice_constant * (1.0*iz+basis[k][2]);
            if((xtmp >= s.sub_domain_lo_x) &&
               (ytmp >= s.sub_domain_lo_y) &&
               (ztmp >= s.sub_domain_lo_z) &&
               (xtmp <  s.sub_domain_hi_x) &&
               (ytmp <  s.sub_domain_hi_y) &&
               (ztmp <  s.sub_domain_hi_z) ) {
              n++;
            }
          }
        }
      }
    }

    system->N_local = n;
    system->N = n;
    system->grow(n);
    s = *system;
    h_x = Kokkos::create_mirror_view(s.x);
    h_v = Kokkos::create_mirror_view(s.v);
    h_q = Kokkos::create_mirror_view(s.q);
    h_type = Kokkos::create_mirror_view(s.type);
    h_id = Kokkos::create_mirror_view(s.id);

    // Initialize system using the equivalent of the LAMMPS
    // velocity geom option, i.e. uniform random kinetic energies.
    // zero out momentum of the whole system afterwards, to eliminate
    // drift (bad for energy statistics)

    for(T_INT iz=iz_start; iz<=iz_end; iz++) {
      for(T_INT iy=iy_start; iy<=iy_end; iy++) {
        for(T_INT ix=ix_start; ix<=ix_end; ix++) {
          for(int k = 0; k<4; k++) {
            T_FLOAT xtmp = lattice_constant * (1.0*ix+basis[k][0]);
            T_FLOAT ytmp = lattice_constant * (1.0*iy+basis[k][1]);
            T_FLOAT ztmp = lattice_constant * (1.0*iz+basis[k][2]);
            if((xtmp >= s.sub_domain_lo_x) &&
               (ytmp >= s.sub_domain_lo_y) &&
               (ztmp >= s.sub_domain_lo_z) &&
               (xtmp <  s.sub_domain_hi_x) &&
               (ytmp <  s.sub_domain_hi_y) &&
               (ztmp <  s.sub_domain_hi_z) ) {
              n++;
            }
          }
        }
      }
    }
    system->grow(n);
    System s = *system;
    h_x = Kokkos::create_mirror_view(s.x);
    h_v = Kokkos::create_mirror_view(s.v);
    h_q = Kokkos::create_mirror_view(s.q);
    h_type = Kokkos::create_mirror_view(s.type);
    h_id = Kokkos::create_mirror_view(s.id);

    n = 0;

    // Initialize system using the equivalent of the LAMMPS
    // velocity geom option, i.e. uniform random kinetic energies.
    // zero out momentum of the whole system afterwards, to eliminate
    // drift (bad for energy statistics)

    for(T_INT iz=iz_start; iz<=iz_end; iz++) {
      for(T_INT iy=iy_start; iy<=iy_end; iy++) {
        for(T_INT ix=ix_start; ix<=ix_end; ix++) {
          for(int k = 0; k<4; k++) {
            T_FLOAT xtmp = lattice_constant * (1.0*ix+basis[k][0]);
            T_FLOAT ytmp = lattice_constant * (1.0*iy+basis[k][1]);
            T_FLOAT ztmp = lattice_constant * (1.0*iz+basis[k][2]);
            if((xtmp >= s.sub_domain_lo_x) &&
               (ytmp >= s.sub_domain_lo_y) &&
               (ztmp >= s.sub_domain_lo_z) &&
               (xtmp <  s.sub_domain_hi_x) &&
               (ytmp <  s.sub_domain_hi_y) &&
               (ztmp <  s.sub_domain_hi_z) ) {
              h_x(n,0) = xtmp;
              h_x(n,1) = ytmp;
              h_x(n,2) = ztmp;
              h_type(n) = rand()%s.ntypes;
              h_id(n) = n+1;
              n++;
            }
          }
        }
      }
    }

    // Make ids unique over all processes
    T_INT N_local_offset = n;
    comm->scan_int(&N_local_offset,1);
    for(T_INT i = 0; i<n; i++)
      h_id(i) += N_local_offset - n;

    comm->reduce_int(&system->N,1);

    if(system->do_print)
      printf("Atoms: %i %i\n",system->N,system->N_local);
  }
  // Initialize velocity using the equivalent of the LAMMPS
  // velocity geom option, i.e. uniform random kinetic energies.
  // zero out momentum of the whole system afterwards, to eliminate
  // drift (bad for energy statistics)
  
  {  // Scope s
    System s = *system;
    T_FLOAT total_mass = 0.0;
    T_FLOAT total_momentum_x = 0.0;
    T_FLOAT total_momentum_y = 0.0;
    T_FLOAT total_momentum_z = 0.0;

    for(T_INT i=0; i<system->N_local; i++) {
      LAMMPS_RandomVelocityGeom random;
      double x[3] = {h_x(i,0),h_x(i,1),h_x(i,2)};
      random.reset(temperature_seed,x);

      T_FLOAT mass_i = h_mass(h_type(i));
      T_FLOAT vx = random.uniform()-0.5;
      T_FLOAT vy = random.uniform()-0.5;
      T_FLOAT vz = random.uniform()-0.5;

      h_v(i,0) = vx/sqrt(mass_i);
      h_v(i,1) = vy/sqrt(mass_i);
      h_v(i,2) = vz/sqrt(mass_i);

      h_q(i) = 0.0;

      total_mass += mass_i;
      total_momentum_x += mass_i * h_v(i,0);
      total_momentum_y += mass_i * h_v(i,1);
      total_momentum_z += mass_i * h_v(i,2);

    }
    comm->reduce_float(&total_momentum_x,1);
    comm->reduce_float(&total_momentum_y,1);
    comm->reduce_float(&total_momentum_z,1);
    comm->reduce_float(&total_mass,1);

    T_FLOAT system_vx = total_momentum_x / total_mass;
    T_FLOAT system_vy = total_momentum_y / total_mass;
    T_FLOAT system_vz = total_momentum_z / total_mass;

    for(T_INT i=0; i<system->N_local; i++) {
      h_v(i,0) -= system_vx;
      h_v(i,1) -= system_vy;
      h_v(i,2) -= system_vz;
    }

    Kokkos::deep_copy(s.v,h_v);
    Temperature temp(comm);
    T_V_FLOAT T = temp.compute(system);

    T_V_FLOAT T_init_scale = sqrt(temperature_target/T);

    for(T_INT i=0; i<system->N_local; i++) {
      h_v(i,0) *= T_init_scale;
      h_v(i,1) *= T_init_scale;
      h_v(i,2) *= T_init_scale;
    }
    Kokkos::deep_copy(s.x,h_x);
    Kokkos::deep_copy(s.v,h_v);
    Kokkos::deep_copy(s.q,h_q);
    Kokkos::deep_copy(s.type,h_type);
    Kokkos::deep_copy(s.id,h_id);
  }
}
