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

#include <cabanamd.h>
#include <property_temperature.h>
#include <property_kine.h>
#include <property_pote.h>
#include "read_data.h"

#define MAXPATHLEN 1024

CabanaMD::CabanaMD() {
  // First we need to create the System data structures
  // They are used by input
  system = new System();
  system->init();

  // Create the Input System, no modules for that,
  // so we can init it in constructor
  input = new Input(system);

}

void CabanaMD::init(int argc, char* argv[]) {

  if(system->do_print) {
     Kokkos::DefaultExecutionSpace::print_configuration(std::cout);
  }

  // Lets parse the command line arguments
  input->read_command_line_args(argc,argv);
  printf("Read command line arguments\n");
  // Read input file
  input->read_file();
  printf("Read input file\n");
  T_X_FLOAT neigh_cutoff = input->force_cutoff + input->neighbor_skin;
  
  // Now we know which integrator type to use
  integrator = new Integrator(system);

  // Fill some binning
  binning = new Binning(system);

  // Create Force Type
  if(false) {}
  #define FORCE_MODULES_INSTANTIATION
  #include<modules_force.h>
  #undef FORCE_MODULES_INSTANTIATION
  else comm->error("Invalid ForceType");
  for(std::size_t line = 0; line < input->force_coeff_lines.dimension_0(); line++) {
   force->init_coeff(neigh_cutoff,
                    input->input_data.words[input->force_coeff_lines(line)]);
  }
  // Create Communication Submodule
  comm = new Comm(system, neigh_cutoff);

  // Do some additional settings
  force->comm_newton = input->comm_newton;

  // system->print_particles();
  if(system->do_print) {
    printf("Using: %s %s %s %s\n",force->name(),comm->name(),binning->name(),integrator->name());
  }

  // Ok lets go ahead and create the particles if that didn't happen yet
  if(system->N == 0 && input->read_data_flag==true)
    read_lammps_data_file(input->lammps_data_file, system, comm);
  else if(system->N == 0)
    input->create_lattice(comm);

  // Create the Halo
  comm->exchange(); 

  // Sort particles
  binning->create_binning(neigh_cutoff,neigh_cutoff,neigh_cutoff,1,true,false,true);

  // Set up particles
  comm->exchange_halo();

  // Compute NeighList
  force->create_neigh_list(system);

  // Compute initial forces
  auto f = Cabana::slice<Forces>(system->xvf);
  Cabana::deep_copy(f, 0.0);
  force->compute(system);

  if(input->comm_newton || input->force_type == 2) { //update force if nnp pair style
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
    T_FLOAT PE = pote.compute(system,force)/system->N;
    T_FLOAT KE = kine.compute(system)/system->N;
    if(system->do_print) {
      if (!system->print_lammps) {
        printf("\n");
        printf("#Timestep Temperature PotE ETot Time Atomsteps/s\n");
        printf("%i %lf %lf %lf %lf %e\n",step,T,PE,PE+KE,0.0,0.0);
      } else {
        printf("\n");
        printf("Step Temp E_pair TotEng CPU\n");
        printf("     %i %lf %lf %lf %e\n",step,T,PE,PE+KE,0.0);
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
  double integrate_time = 0;
  double other_time = 0;

  double last_time = 0;
  Kokkos::Timer timer,force_timer,comm_timer,neigh_timer,integrate_timer,other_timer;

  // Timestep Loop
  for(int step = 1; step <= nsteps; step++ ) {
    
    // Do first part of the verlet time step integration 
    integrate_timer.reset();
    integrator->initial_integrate();
    integrate_time += integrate_timer.seconds();
    
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
      
      // Compute Neighbor List 
      neigh_timer.reset();
      force->create_neigh_list(system);
      neigh_time += neigh_timer.seconds();
    } else {
      // Exchange Halo
      comm_timer.reset();
      comm->update_halo();
      comm_time += comm_timer.seconds();
    }

    // Zero out forces
    force_timer.reset();
    auto f = Cabana::slice<Forces>(system->xvf);
    Cabana::deep_copy(f, 0.0);

    // Compute Short Range Force
    force->compute(system);
    force_time += force_timer.seconds();

    // This is where Bonds, Angles and KSpace should go eventually 
    
    // Reverse Communicate Force Update on Halo
    if(input->comm_newton or input->force_type == 2) { //update force if nnp pair style
      comm_timer.reset();
      comm->update_force();
      comm_time += comm_timer.seconds();
    }

    // Do second part of the verlet time step integration 
    integrate_timer.reset();
    integrator->final_integrate();
    integrate_time += integrate_timer.seconds();
    
    other_timer.reset();
    // On output steps print output
    if(step%input->thermo_rate==0) {
      T_FLOAT T = temp.compute(system);
      T_FLOAT PE = pote.compute(system,force)/system->N;
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
    auto f = Cabana::slice<Forces>(system->xvf);
    auto id = Cabana::slice<IDs>(system->xvf);
 
    printf("TXXXX: \n");
    printf("%d %f %f %f\n", id(0), f(0,0), f(0,1), f(0,2));    
    printf("%d %f %f %f\n", id(1), f(1,0), f(1,1), f(1,2));    
    printf("%d %f %f %f\n", id(2), f(2,0), f(2,1), f(2,2));    
    printf("%d %f %f %f\n", id(3), f(3,0), f(3,1), f(3,2));    
    printf("%d %f %f %f\n", id(4), f(4,0), f(4,1), f(4,2));    
    printf("%d %f %f %f\n", id(5), f(5,0), f(5,1), f(5,2));    
  double time = timer.seconds();

  if(system->do_print) {
    if (!system->print_lammps) {
      printf("\n");
      printf("#Procs Particles | Time T_Force T_Neigh T_Comm T_Int T_Other | Steps/s Atomsteps/s Atomsteps/(proc*s)\n");
      printf("%i %i | %lf %lf %lf %lf %lf %lf | %lf %e %e PERFORMANCE\n",comm->num_processes(),system->N,time,
        force_time,neigh_time,comm_time,integrate_time,other_time,
        1.0*nsteps/time,1.0*system->N*nsteps/time,1.0*system->N*nsteps/time/comm->num_processes());
      printf("%i %i | %lf %lf %lf %lf %lf %lf | FRACTION\n",comm->num_processes(),system->N,1.0,
        force_time/time,neigh_time/time,comm_time/time,integrate_time/time,other_time/time);
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
  auto h_id = Cabana::slice<IDs>(s.xvf);
  auto h_type = Cabana::slice<Types>(s.xvf);
  auto h_q = Cabana::slice<Charges>(s.xvf);
  auto h_x = Cabana::slice<Positions>(s.xvf);
  auto h_v = Cabana::slice<Velocities>(s.xvf);
  auto h_f = Cabana::slice<Forces>(s.xvf);

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
//     2. Add MPI Rank file ids in Reference [DONE]
//     3. Move to separate class
//     4. Add pressure to thermo output
//     5. basis_offset [DONE]
//     6. correctness output to file [DONE]

void CabanaMD::check_correctness(int step) {

  if(step%input->correctness_rate) return;

  FILE* fpref;
  T_INT n = system->N_local;
  T_INT ntmp;

  auto x = Cabana::slice<Positions>(system->xvf);
  auto v = Cabana::slice<Velocities>(system->xvf);
  auto f = Cabana::slice<Forces>(system->xvf);
  auto id = Cabana::slice<IDs>(system->xvf);
  auto type = Cabana::slice<Types>(system->xvf);
  auto q = Cabana::slice<Charges>(system->xvf);

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
  auto xref = Cabana::slice<Positions>(xvf_ref);
  auto vref = Cabana::slice<Velocities>(xvf_ref);
  auto fref = Cabana::slice<Forces>(xvf_ref);
  auto idref = Cabana::slice<IDs>(xvf_ref);
  auto typeref = Cabana::slice<Types>(xvf_ref);
  auto qref = Cabana::slice<Charges>(xvf_ref);

  fread(idref.data(),sizeof(T_INT),n,fpref);
  fread(typeref.data(),sizeof(T_INT),n,fpref); 
  fread(qref.data(),sizeof(T_FLOAT),n,fpref);
  fread(xref.data(),sizeof(T_X_FLOAT),3*n,fpref);
  fread(vref.data(),sizeof(T_V_FLOAT),3*n,fpref);
  fread(fref.data(),sizeof(T_F_FLOAT),3*n,fpref);

  xref = Cabana::slice<Positions>(xvf_ref);
  vref = Cabana::slice<Velocities>(xvf_ref);
  fref = Cabana::slice<Forces>(xvf_ref);
  idref = Cabana::slice<IDs>(xvf_ref);
  typeref = Cabana::slice<Types>(xvf_ref);
  qref = Cabana::slice<Charges>(xvf_ref);

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
