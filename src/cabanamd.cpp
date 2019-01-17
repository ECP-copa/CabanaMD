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

#include <cabanamd.h>
#include <property_temperature.h>
#include <property_kine.h>
#include <property_pote.h>

#define MAXPATHLEN 1024

CabanaMD::CabanaMD() {
  // First we need to create the System data structures
  // They are used by input
  system = new System();
  system->init();

  // Create the Input System, no modules for that,
  // so we can init it in constructor
  input = new Input(system);

  neighbor = NULL;
}

void CabanaMD::init(int argc, char* argv[]) {

  if(system->do_print)
    Kokkos::DefaultExecutionSpace::print_configuration(std::cout);

  // Lets parse the command line arguments
  input->read_command_line_args(argc,argv);

  // Now we know which integrator type to use
  integrator = new Integrator(system);
  
  // Fill some binning
  binning = new Binning(system);

  // Create Force Type
  bool half_neigh = input->force_iteration_type == FORCE_ITER_NEIGH_HALF;
  force = new Force(system,half_neigh);
  int nforce = (pow(input->ntypes, 2.0) + input->ntypes)/2;
  for(int line = 0; line < nforce; line++) {
    force->init_coeff(input->force_types[line], input->force_coeff[line]);
  }

  // Create Neighbor Instance
  if (false) {}
#define NEIGHBOR_MODULES_INSTANTIATION
#include<modules_neighbor.h>
#undef NEIGHBOR_MODULES_INSTANTIATION
  else comm->error("Invalid NeighborType");

  // Create Communication Submodule
  comm = new Comm(system,input->force_cutoff + input->neighbor_skin);

  // Do some additional settings
  force->comm_newton = input->comm_newton;
  if(neighbor)
    neighbor->comm_newton = input->comm_newton;

  // system->print_particles();
  if(system->do_print) {
    printf("Using: %s %s %s %s %s\n",force->name(),neighbor->name(),comm->name(),binning->name(),integrator->name());
  }

  // Ok lets go ahead and create the particles if that didn't happen yet
  if(system->N == 0)
    input->create_lattice(comm);

  // Create the Halo
  comm->exchange(); 

  // Sort particles
  T_F_FLOAT neigh_cutoff = input->force_cutoff + input->neighbor_skin;
  binning->create_binning(neigh_cutoff,neigh_cutoff,neigh_cutoff,1,true,false,true);

  // Set up particles
  comm->exchange_halo();

  // Create binning for neighborlist construction
  binning->create_binning(neigh_cutoff,neigh_cutoff,neigh_cutoff,1,true,true,false);

  // Compute NeighList
  if(neighbor)
    neighbor->create_neigh_list(system,binning,force->half_neigh,false);

  // Compute initial forces
  Kokkos::deep_copy(system->f,0.0);
  force->compute(system,neighbor);

  if(input->comm_newton) {
    // Reverse Communicate Force Update on Halo
    comm->update_force();
  }
  
  // Initial output
  int step = 0;
  if(input->thermo_rate > 0) {
    Temperature temp(comm);
    PotE pote(comm);
    KinE kine(comm);
    T_FLOAT T = temp.compute(system);
    T_FLOAT PE = pote.compute(system,binning,neighbor,force)/system->N;
    T_FLOAT KE = kine.compute(system)/system->N;
    if(system->do_print) {
      if (!system->print_lammps) {
        printf("\n");
        printf("#Timestep Temperature PotE ETot Time Atomsteps/s\n");
        printf("%i %lf %lf %lf %lf %e\n",step,T,PE,PE+KE,0.0,0.0);
      } else {
        printf("\n");
        printf("Step Temp E_pair TotEng CPU\n");
        printf("     %i %lf %lf %lf %lf %e\n",step,T,PE,PE+KE,0.0);
      }
    }
  }

  if(input->dumpbinaryflag)
    dump_binary(step);

  if(input->correctnessflag)
    check_correctness(step);

}

void CabanaMD::run(int nsteps) {
  T_F_FLOAT neigh_cutoff = input->force_cutoff + input->neighbor_skin;

  Temperature temp(comm);
  PotE pote(comm);
  KinE kine(comm);

  double force_time = 0;
  double comm_time  = 0;
  double neigh_time = 0;
  double other_time = 0;

  double last_time;
  Kokkos::Timer timer,force_timer,comm_timer,neigh_timer,other_timer;

  // Timestep Loop
  for(int step = 1; step <= nsteps; step++ ) {
    
    // Do first part of the verlet time step integration 
    other_timer.reset();
    integrator->initial_integrate();
    other_time += other_timer.seconds();

    if(step%input->comm_exchange_rate==0 && step >0) {
      // Exchange particles
      comm_timer.reset();
      comm->exchange(); 
      comm_time += comm_timer.seconds();

      // Sort particles
      other_timer.reset();
      binning->create_binning(neigh_cutoff,neigh_cutoff,neigh_cutoff,1,true,false,true);
      other_time += other_timer.seconds();

      // Exchange Halo
      comm_timer.reset();
      comm->exchange_halo();
      comm_time += comm_timer.seconds();
      
      // Create binning for neighborlist construction
      neigh_timer.reset();
      binning->create_binning(neigh_cutoff,neigh_cutoff,neigh_cutoff,1,true,true,false);

      // Compute Neighbor List if necessary
      if(neighbor)
        neighbor->create_neigh_list(system,binning,force->half_neigh,false);
      neigh_time += neigh_timer.seconds();
    } else {
      // Exchange Halo
      comm_timer.reset();
      comm->update_halo();
      comm_time += comm_timer.seconds();
    }

    // Zero out forces 
    force_timer.reset();
    Kokkos::deep_copy(system->f,0.0);
   
    // Compute Short Range Force
    force->compute(system,neighbor);
    force_time += force_timer.seconds();

    // This is where Bonds, Angles and KSpace should go eventually 
    
    // Reverse Communicate Force Update on Halo
    if(input->comm_newton) {
      comm_timer.reset();
      comm->update_force();
      comm_time += comm_timer.seconds();
    }

    // Do second part of the verlet time step integration 
    other_timer.reset();
    integrator->final_integrate();

    // On output steps print output
    if(step%input->thermo_rate==0) {
      T_FLOAT T = temp.compute(system);
      T_FLOAT PE = pote.compute(system,binning,neighbor,force)/system->N;
      T_FLOAT KE = kine.compute(system)/system->N;
      if(system->do_print) {
        if (!system->print_lammps) {
          double time = timer.seconds();
          printf("%i %lf %lf %lf %lf %e\n",step, T, PE, PE+KE, timer.seconds(),1.0*system->N*input->thermo_rate/(time-last_time));
          last_time = time;
        } else {
          double time = timer.seconds();
          printf("     %i %lf %lf %lf %lf\n",step, T, PE, PE+KE, timer.seconds());
          last_time = time;
        }
      }
    }

    if(input->dumpbinaryflag)
      dump_binary(step);
 
    if(input->correctnessflag)
      check_correctness(step);

    other_time += other_timer.seconds();
  }

  double time = timer.seconds();
  T_FLOAT T = temp.compute(system);
  T_FLOAT PE = pote.compute(system,binning,neighbor,force)/system->N;
  T_FLOAT KE = kine.compute(system)/system->N;

  if(system->do_print) {
    if (!system->print_lammps) {
      printf("\n");
      printf("#Procs Particles | Time T_Force T_Neigh T_Comm T_Other | Steps/s Atomsteps/s Atomsteps/(proc*s)\n");
      printf("%i %i | %lf %lf %lf %lf %lf | %lf %e %e PERFORMANCE\n",comm->num_processes(),system->N,time,
        force_time,neigh_time,comm_time,other_time,
        1.0*nsteps/time,1.0*system->N*nsteps/time,1.0*system->N*nsteps/time/comm->num_processes());
    } else {
      printf("Loop time of %f on %i procs for %i steps with %i atoms\n",time,comm->num_processes(),nsteps,system->N);
    }
  }
}

void CabanaMD::dump_binary(int step) {

  // On dump steps print configuration

  if(step%input->dumpbinary_rate) return;
  
  FILE* fp;
  T_INT n = system->N_local;
  
  char* filename = new char[MAXPATHLEN];
  sprintf(filename,"%s%s.%010d.%03d",input->dumpbinary_path,
          "/output",step,comm->process_rank());
  fp = fopen(filename,"wb");
  if (fp == NULL) {
    char str[MAXPATHLEN];
    sprintf(str,"Cannot open dump file %s",filename);
    //comm->error(str);
  }

  System s = *system;
  auto h_id = s.xvf.slice<IDs>();
  auto h_type = s.xvf.slice<Types>();
  auto h_q = s.xvf.slice<Charges>();
  auto h_x = s.xvf.slice<Positions>();
  auto h_v = s.xvf.slice<Velocities>();
  auto h_f = s.xvf.slice<Forces>();

  fwrite(&n,sizeof(T_INT),1,fp);
  fwrite(h_id.data(),sizeof(T_INT),n,fp);
  fwrite(h_type.data(),sizeof(T_INT),n,fp);
  fwrite(h_q.data(),sizeof(T_FLOAT),n,fp);
  fwrite(h_x.data(),sizeof(T_X_FLOAT),3*n,fp);
  fwrite(h_v.data(),sizeof(T_V_FLOAT),3*n,fp);
  fwrite(h_f.data(),sizeof(T_F_FLOAT),3*n,fp);
    
  fclose(fp);
}

// TODO: 1. Add path to Reference [DONE]
//       2. Add MPI Rank file ids in Reference [DONE]
//       3. Move to separate class
//       4. Add pressure to thermo output
//       5. basis_offset [DONE]
//       6. correctness output to file [DONE]

void CabanaMD::check_correctness(int step) {

  if(step%input->correctness_rate) return;

  FILE* fpref;
  T_INT n = system->N_local;
  T_INT ntmp;

  auto x = system->xvf.slice<Positions>();
  auto v = system->xvf.slice<Velocities>();
  auto f = system->xvf.slice<Forces>();
  auto id = system->xvf.slice<IDs>();
  auto type = system->xvf.slice<Types>();
  auto q = system->xvf.slice<Charges>();

  char* filename = new char[MAXPATHLEN];
  sprintf(filename,"%s%s.%010d.%03d",input->reference_path,
          "/output",step,comm->process_rank());
  fpref = fopen(filename,"rb");
  if (fpref == NULL) {
    char str[MAXPATHLEN];
    sprintf(str,"Cannot open input file %s",filename);
    //comm->error(str);
  }
  
  fread(&ntmp,sizeof(T_INT),1,fpref);
  if (ntmp != n) {
    //comm->error("Mismatch in current and reference atom counts");
    printf("Mismatch in current and reference atom counts\n");
  }
  
  AoSoA xvf_ref( n );
  auto xref = xvf_ref.slice<Positions>();
  auto vref = xvf_ref.slice<Velocities>();
  auto fref = xvf_ref.slice<Forces>();
  auto idref = xvf_ref.slice<IDs>();
  auto typeref = xvf_ref.slice<Types>();
  auto qref = xvf_ref.slice<Charges>();
  
  fread(idref.data(),sizeof(T_INT),n,fpref);
  fread(typeref.data(),sizeof(T_INT),n,fpref); 
  fread(qref.data(),sizeof(T_FLOAT),n,fpref);
  fread(xref.data(),sizeof(T_X_FLOAT),3*n,fpref);
  fread(vref.data(),sizeof(T_V_FLOAT),3*n,fpref);
  fread(fref.data(),sizeof(T_F_FLOAT),3*n,fpref);
  
  xref = xvf_ref.slice<Positions>();
  vref = xvf_ref.slice<Velocities>();
  fref = xvf_ref.slice<Forces>();
  idref = xvf_ref.slice<IDs>();
  typeref = xvf_ref.slice<Types>();
  qref = xvf_ref.slice<Charges>();

  T_FLOAT sumdelrsq = 0.0;
  T_FLOAT sumdelvsq = 0.0;
  T_FLOAT sumdelfsq = 0.0;
  T_FLOAT maxdelr = 0.0;
  T_FLOAT maxdelv = 0.0;
  T_FLOAT maxdelf = 0.0;
  for (int i = 0; i < n; i++) {
    int ii = -1;
    if (id(i) != idref(i)) 
      for (int j = 0; j < n; j++) {
        if (id(j) == idref(i)) {
          ii = j;
          break;
        }
      }
    else
      ii = i;
    
    if (ii == -1)
      printf("Unable to find current id matchinf reference id %d \n",idref(i));
    else {
      T_FLOAT delx, dely, delz, delrsq;
      delx = x(ii,0)-xref(i,0);
      dely = x(ii,1)-xref(i,1);
      delz = x(ii,2)-xref(i,2);
      delrsq = delx*delx + dely*dely + delz*delz;
      sumdelrsq += delrsq;
      maxdelr = MAX(fabs(delx),maxdelr);
      maxdelr = MAX(fabs(dely),maxdelr);
      maxdelr = MAX(fabs(delz),maxdelr);
      
      delx = v(ii,0)-vref(i,0);
      dely = v(ii,1)-vref(i,1);
      delz = v(ii,2)-vref(i,2);
      delrsq = delx*delx + dely*dely + delz*delz;
      sumdelvsq += delrsq;
      maxdelv = MAX(fabs(delx),maxdelv);
      maxdelv = MAX(fabs(dely),maxdelv);
      maxdelv = MAX(fabs(delz),maxdelv);
      
      delx = f(ii,0)-fref(i,0);
      dely = f(ii,1)-fref(i,1);
      delz = f(ii,2)-fref(i,2);
      delrsq = delx*delx + dely*dely + delz*delz;
      sumdelfsq += delrsq;
      maxdelf = MAX(fabs(delx),maxdelf);
      maxdelf = MAX(fabs(dely),maxdelf);
      maxdelf = MAX(fabs(delz),maxdelf);
    }
  }

  fclose(fpref);

  // Can't use this with current CommMPI::reduce_float()
  // T_FLOAT buf[3];
  // buf[0] = sumdelrsq;
  // buf[1] = sumdelvsq;
  // buf[2] = sumdelfsq;
  // comm->reduce_float(&buf[0],3);
  // sumdelrsq = buf[0];
  // sumdelrsq = buf[1];
  // sumdelrsq = buf[2];
  // buf[0] = maxdelr;
  // buf[1] = maxdelv;
  // buf[2] = maxdelf;
  // comm->reduce_max_float(buf,3);
  // maxdelr =   buf[0];
  // maxdelv =   buf[1];
  // maxdelf =   buf[2];

  comm->reduce_float(&sumdelrsq,1);
  comm->reduce_float(&sumdelvsq,1);
  comm->reduce_float(&sumdelfsq,1);
  comm->reduce_max_float(&maxdelr,1);
  comm->reduce_max_float(&maxdelv,1);
  comm->reduce_max_float(&maxdelf,1);

  if (system->do_print) {
    if (step == 0) {
      FILE* fpout = fopen(input->correctness_file,"w");
      fprintf(fpout, "# timestep deltarnorm maxdelr deltavnorm maxdelv deltafnorm maxdelf\n");
      fprintf(fpout, "%d %g %g %g %g %g %g\n",step,sqrt(sumdelrsq),maxdelr,sqrt(sumdelvsq),
              maxdelv,sqrt(sumdelfsq),maxdelf);
      fclose(fpout);
    } else {
      FILE* fpout = fopen(input->correctness_file,"a");
      fprintf(fpout, "%d %g %g %g %g %g %g\n",step,sqrt(sumdelrsq),maxdelr,sqrt(sumdelvsq),
              maxdelv,sqrt(sumdelfsq),maxdelf);
      fclose(fpout);
    }
  }
}

void CabanaMD::print_performance() {}

void CabanaMD::shutdown() {
  system->destroy();
}
