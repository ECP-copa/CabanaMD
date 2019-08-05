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

/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying LICENSE_MINIMD file.
---------------------------------------------------------------------- */
#include <cstring>
#include <string>
#include <comm_serial.h>
#include <iostream>
#include <fstream>
#include <types.h>

using namespace std;

string read_lammps_parse_keyword(ifstream &file)
{
  // TODO: error checking. Look at setup.cpp
  // proc 0 reads upto non-blank line plus 1 following line
  // eof is set to 1 if any read hits end-of-file
  string line;
  string keyword = "";
  while(1) {
    //exit if end-of-file is encountered
    if (file.eof())
      break;
    getline(file, line);
    //ignore anything after # by getting substring till # (takes care of comment lines too)
    line = line.substr(0, line.find('#'));
    //for non blank lines, skip leading whitespaces and read word
    std::size_t pos = line.find_first_not_of(" \r\t\n");
    if (pos != string::npos) {
      keyword = line;
      if ((keyword.compare("Atoms ") == 0) || (keyword.compare("Velocities ") == 0) || 
          (keyword.compare("Atoms") == 0) || (keyword.compare("Velocities") == 0)) 
        return keyword;
      else
       {
         printf("ERROR: Unknown identifier in data file: %s\n", keyword.data());
         return keyword;
       }
    }
    else
      continue;
  }
  return keyword;
}

void read_lammps_header(ifstream &file, System* s)
{
  string line;
  // skip 1st line of file
  if (!getline(file, line))
    printf("ERROR: could not read line from file. Please check for a valid file and ensure that file path is less than 32 characters\n");
  while(1) {
    
    getline(file, line);
    //skip blank lines
    std::size_t pos = line.find_first_not_of(" \r\t\n");
    if (pos == string::npos)
      continue;
    
    //ignore anything after # by getting substring till # (takes care of comment lines too)
    line = line.substr(0, line.find('#'));
    
    int natoms, ntypes;
    double xlo, xhi, ylo, yhi, zlo, zhi;
    const char* temp = line.data(); //convert to C-string for sscanf utility
    // search line for header keyword and set corresponding variable
    if(line.find("atoms") != string::npos) {
      sscanf(temp, "%i", &natoms);
      s->N = natoms;
    }
    else if(line.find("atom types") != string::npos) {
      sscanf(temp, "%i", &ntypes);
      s->ntypes = ntypes;
    }
    else if(line.find("xlo xhi") != string::npos) {
      sscanf(temp, "%lg %lg", &xlo, &xhi);
      s->domain_lo_x = xlo;
      s->domain_hi_x = xhi;
      s->domain_x = xhi - xlo;
    }
    else if(line.find("ylo yhi") != string::npos) {
      sscanf(temp, "%lg %lg", &ylo, &yhi);
      s->domain_lo_y = ylo;
      s->domain_hi_y = yhi;
      s->domain_y = yhi - ylo;
    }
    else if(line.find("zlo zhi") != string::npos) {
      sscanf(temp, "%lg %lg", &zlo, &zhi);
      s->domain_lo_z = zlo;
      s->domain_hi_z = zhi;
      s->domain_z = zhi - zlo;
      break;
    }
  }
}
    

void read_lammps_atoms(ifstream &file, System* s)
{
  string line;
  auto x = Cabana::slice<0>(s->aosoa_x);
  auto id = Cabana::slice<0>(s->aosoa_id);
  auto type = Cabana::slice<0>(s->aosoa_type);
  auto q = Cabana::slice<0>(s->aosoa_q);
  
  //skip any empty lines before reading in data
  while (1) {
    getline(file, line);
    if (!line.empty())
      break;
  }
  
  
  T_INT id_tmp, type_tmp;
  T_FLOAT x_tmp, y_tmp, z_tmp, q_tmp; 
  T_INT counter = 0;
  for (int n=0; n < s->N; n++) {
    const char* temp = line.data();
    if (s->atom_style == "atomic") {
      sscanf(temp, "%i %i %lg %lg %lg", &id_tmp, &type_tmp, &x_tmp, &y_tmp, &z_tmp);
      if((x_tmp >= s->sub_domain_lo_x) &&
         (y_tmp >= s->sub_domain_lo_y) &&
         (z_tmp >= s->sub_domain_lo_z) &&
         (x_tmp <  s->sub_domain_hi_x) &&
         (y_tmp <  s->sub_domain_hi_y) &&
         (z_tmp <  s->sub_domain_hi_z) ) { 
        id(n) = id_tmp; type(n) = type_tmp-1; x(n,0) = x_tmp; x(n,1) = y_tmp; x(n,2) = z_tmp;
        q(n) = 0;
        counter++;
      }
    getline(file, line); 
    }
    if (s->atom_style == "charge") {
      sscanf(temp, "%i %i %lg %lg %lg %lg", &id_tmp, &type_tmp, &q_tmp, &x_tmp, &y_tmp, &z_tmp);
      if((x_tmp >= s->sub_domain_lo_x) &&
         (y_tmp >= s->sub_domain_lo_y) &&
         (z_tmp >= s->sub_domain_lo_z) &&
         (x_tmp <  s->sub_domain_hi_x) &&
         (y_tmp <  s->sub_domain_hi_y) &&
         (z_tmp <  s->sub_domain_hi_z) ) { 
        id(n) = id_tmp; type(n) = type_tmp-1; q(n) = q_tmp; x(n,0) = x_tmp; x(n,1) = y_tmp; x(n,2) = z_tmp;
        counter++;
      }
    //getline pushed to the end of loop because line already stores the 1st non-blank line
    //after exiting while loop
    getline(file, line); 
    }
  }
  s->N_local = counter;
  s->N = counter;
}


void read_lammps_velocities(ifstream &file, System* s)
{
  string line;
  auto v = Cabana::slice<0>(s->aosoa_v);
  
  //skip any empty lines before reading in data
  while (1) {
    getline(file, line);
    if (!line.empty())
      break;
  }
  
  T_INT id_tmp;
  T_FLOAT vx_tmp, vy_tmp, vz_tmp; 
  for (int n=0; n < s->N; n++) {
    const char* temp = line.data();
    sscanf(temp, "%i %lg %lg %lg", &id_tmp, &vx_tmp, &vy_tmp, &vz_tmp);
    v(n,0) = vx_tmp; v(n,1) = vy_tmp; v(n,2) = vz_tmp;
    getline(file, line);
  }
}


void read_lammps_data_file(const char* filename, System* s, Comm* comm) {
  
  string keyword;
  ifstream file(filename);
  //TODO: error checking bad files
  //read header information
  read_lammps_header(file, s);
  
  s->resize(s->N);
  
  //perform domain decomposition and get access to subdomains
  comm->create_domain_decomposition();
  
  // check that exiting string is a valid section keyword
  keyword = read_lammps_parse_keyword(file);
  
  //read atom information
  read_lammps_atoms(file, s);
  //TODO: catch this error: printf("Must read Atoms before Velocities\n");
  
  keyword = read_lammps_parse_keyword(file);
  //read velocities
  if(keyword.compare("Velocities") == 0)
    read_lammps_velocities(file, s);
  
  //TODO: read masses from data file
  /*else if(strcmp(keyword, "Masses") == 0) {
    fgets(line, MAXLINE, fp);

    #if PRECISION==1
            sscanf(line, "%i %g", &tmp, &atom.mass);
    #else
            sscanf(line, "%i %lg", &tmp, &atom.mass);
    #endif
  }
  */
  #ifdef CabanaMD_ENABLE_MPI
  int me;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  // check that correct # of atoms were created 
  int natoms;
  MPI_Allreduce(s->N_local, &natoms, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if(natoms != s->N) {
    if(me == 0 && system->do_print)
        printf("ERROR: Created incorrect # of atoms\n");
  }
  #endif
}

