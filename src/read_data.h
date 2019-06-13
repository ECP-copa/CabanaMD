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

using namespace std;

string read_lammps_parse_keyword(ifstream &file, System* s)
{
  // TODO: error checking. Look at setup.cpp
  // proc 0 reads upto non-blank line plus 1 following line
  // eof is set to 1 if any read hits end-of-file
  string line, keyword;
  while(1) {
    getline(file, line);
    //for non blank lines, skip leading whitespaces and read word
    int pos = line.find_first_not_of(" \r\t\n");
    if (pos != string::npos) {
      keyword = line;
      if (keyword != "Atoms" || keyword != "Velocities")
        printf("ERROR: Unknown identifier in data file: %s", keyword);
      return keyword;
    }
    else
      //skip blank lines
      continue;
  }
}

void read_lammps_header(ifstream &file, System* s)
{
  string line;

  // skip 1st line of file
  string firstline;
  getline(file, firstline);

  // skip blank lines and trim anything after # (takes care of comment lines)
  while(1) {
    getline(file, line);
    int pos = line.find_first_not_of(" \r\t\n");
    if (pos == string::npos)
      continue;
    //int hash_loc = line.find('#');
    const char* temp = line.data();
    //temp[hash_loc] = '\0';

    double xlo, xhi, ylo, yhi, zlo, zhi;
    // search line for header keyword and set corresponding variable
    if(line.find("atoms") != string::npos)
      sscanf(temp, "%i", s->N);
    else if(line.find("atom types") != string::npos)
      sscanf(temp, "%i", s->ntypes);

    else if(line.find("xlo xhi") != string::npos) {
      sscanf(temp, "%lg %lg", &xlo, &xhi);
      s->domain_x = xhi - xlo;
    }
    else if(line.find("ylo yhi") != string::npos) {
      sscanf(temp, "%lg %lg", &ylo, &yhi);
      s->domain_y = yhi - ylo;
    }
    else if(line.find("zlo zhi") != string::npos) {
      sscanf(temp, "%lg %lg", &zlo, &zhi);
      s->domain_z = zhi - zlo;
    }
    else
      break;

  }

}
    

void read_lammps_atoms(ifstream &file, System* s)
{
  string line;
  auto x = Cabana::slice<Positions>(s->xvf);
  auto id = Cabana::slice<IDs>(s->xvf);
  auto type = Cabana::slice<Types>(s->xvf);
  auto q = Cabana::slice<Charges>(s->xvf);
  
  for (int n=0; n < s->N; n++) {
    getline(file, line);
    const char* temp = line.data();
    if (s->atom_style == "atomic") {
      sscanf(temp, "%i %i %lg %lg %lg", id(n), type(n), x(n,0), x(n,1), x(n,2));
      q(n) = 0;
    }
    if (s->atom_style == "charge") {
      sscanf(temp, "%i %i %lg %lg %lg %lg", id(n), type(n), q(n), x(n,0), x(n,1), x(n,2));
    }
  }
}


void read_lammps_velocities(ifstream &file, System* s)
{
  string line;
  auto v = Cabana::slice<Velocities>(s->xvf);
  auto id = Cabana::slice<IDs>(s->xvf);
  
  for (int n=0; n < s->N; n++) {
    getline(file, line);
    const char* temp = line.data();
    sscanf(temp, "%i %lg %lg %lg", id(n), v(n,0), v(n,1), v(n,2));
  }
}


void read_lammps_data_file(string filename, System* s) {
  
  string keyword;
  ifstream file(filename);
  //TODO: error checking bad files
  //read header information
  read_lammps_header(file, s);
  //perform domain decomposition and get access to subdomains
  //comm->create_domain_decomposition(); //TODO: Look into this 
  // check that exiting string is a valid section keyword
  keyword = read_lammps_parse_keyword(file, s);
  //read atom information
  read_lammps_atoms(file, s);
  //TODO: catch this error: printf("Must read Atoms before Velocities\n");
  keyword = read_lammps_parse_keyword(file, s);
  //read velocities
  if(keyword.compare("Velocities") == 0) {
    read_lammps_velocities(file, s);
  }
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

