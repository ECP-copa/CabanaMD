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

#ifndef VTK_DOMAIN_WRITER_H
#define VTK_DOMAIN_WRITER_H
#include <mpi.h>

#include <sstream>
#include <string>

namespace VTKWriter
{
// Write PVTU
void writeDomainParallelFile( MPI_Comm comm, int time_step,
                              std::string &basename )
{
    // Should only be called from a single rank
    int size;
    MPI_Comm_size( comm, &size );
    std::stringstream filename;
    filename << basename << "_" << time_step << ".pvtu";
    FILE *file = fopen( filename.str().c_str(), "w" );
    fprintf( file, "<?xml version=\"1.0\"?>\n" );
    fprintf( file, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" "
                   "byte_order=\"LittleEndian\" header_type=\"UInt32\">\n" );
    fprintf( file, "<PUnstructuredGrid>\n" );
    fprintf( file, "\t<PCellData>\n" );
    fprintf( file, "\t\t<PDataArray type=\"Int32\" Name=\"rank\"/>\n" );
    fprintf( file, "\t\t<PDataArray type=\"Float64\" Name=\"work\"/>\n" );
    fprintf( file, "\t</PCellData>\n" );
    fprintf( file, "\t<PPoints>\n" );
    fprintf( file, "\t\t<PDataArray type=\"Float64\" Name=\"Points\" "
                   "NumberOfComponents=\"3\"/>\n" );
    fprintf( file, "\t</PPoints>\n" );
    for ( int i = 0; i < size; ++i )
        fprintf( file, "\t<Piece Source=\"%s_%d_%d.vtu\"/>\n", basename.c_str(),
                 time_step, i );
    fprintf( file, "</PUnstructuredGrid>\n" );
    fprintf( file, "</VTKFile>\n" );
    fclose( file );
}

// Write VTU for domain (low corner, high corner)
// basename will be appended with the corresponding time step, rank and
// extension
void writeDomain( MPI_Comm comm, int time_step,
                  std::array<double, 6> &domain_vertices, double work,
                  std::string &basename )
{
    int rank;
    MPI_Comm_rank( comm, &rank );
    if ( rank == 1 )
        writeDomainParallelFile( comm, time_step, basename );
    std::stringstream filename;
    // todo(sschulz): properly format, according to max rank
    filename << basename << "_" << time_step << "_" << rank << ".vtu";
    FILE *file = fopen( filename.str().c_str(), "w" );
    fprintf( file, "<?xml version=\"1.0\"?>\n" );
    fprintf( file, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
                   "byte_order=\"LittleEndian\" header_type=\"UInt32\">\n" );
    fprintf( file, "<UnstructuredGrid>\n" );
    std::array<double, 3 * 8> vertices;
    vertices[0 * 3 + 0] = domain_vertices[0];
    vertices[2 * 3 + 0] = domain_vertices[0];
    vertices[4 * 3 + 0] = domain_vertices[0];
    vertices[6 * 3 + 0] = domain_vertices[0];
    vertices[0 * 3 + 1] = domain_vertices[1];
    vertices[1 * 3 + 1] = domain_vertices[1];
    vertices[4 * 3 + 1] = domain_vertices[1];
    vertices[5 * 3 + 1] = domain_vertices[1];
    vertices[0 * 3 + 2] = domain_vertices[2];
    vertices[1 * 3 + 2] = domain_vertices[2];
    vertices[2 * 3 + 2] = domain_vertices[2];
    vertices[3 * 3 + 2] = domain_vertices[2];
    vertices[1 * 3 + 0] = domain_vertices[3];
    vertices[3 * 3 + 0] = domain_vertices[3];
    vertices[5 * 3 + 0] = domain_vertices[3];
    vertices[7 * 3 + 0] = domain_vertices[3];
    vertices[2 * 3 + 1] = domain_vertices[4];
    vertices[3 * 3 + 1] = domain_vertices[4];
    vertices[6 * 3 + 1] = domain_vertices[4];
    vertices[7 * 3 + 1] = domain_vertices[4];
    vertices[4 * 3 + 2] = domain_vertices[5];
    vertices[5 * 3 + 2] = domain_vertices[5];
    vertices[6 * 3 + 2] = domain_vertices[5];
    vertices[7 * 3 + 2] = domain_vertices[5];
    std::array<int, 8> connectivity = { 0, 1, 2, 3, 4, 5, 6, 7 };
    fprintf( file, "<Piece NumberOfPoints=\"8\" NumberOfCells=\"1\">\n" );
    fprintf( file, "\t<PointData>\n" );
    fprintf( file, "\t</PointData>\n" );
    fprintf( file, "\t<CellData>\n" );
    fprintf( file, "\t\t<DataArray type=\"Int32\" Name=\"rank\" "
                   "NumberOfComponents=\"1\" format=\"ascii\">\n" );
    fprintf( file, "%d", rank );
    fprintf( file, "\n\t\t</DataArray>\n" );
    fprintf( file, "\t\t<DataArray type=\"Float64\" Name=\"work\" "
                   "NumberOfComponents=\"1\" format=\"ascii\">\n" );
    fprintf( file, "%g", work );
    fprintf( file, "\n\t\t</DataArray>\n" );
    fprintf( file, "\t</CellData>\n" );
    fprintf( file, "\t<Points>\n" );
    fprintf( file, "\t\t<DataArray type=\"Float64\" Name=\"Points\" "
                   "NumberOfComponents=\"3\" format=\"ascii\">\n" );
    for ( const double &vert : vertices )
        fprintf( file, "%g ", vert );
    fprintf( file, "\n\t\t</DataArray>\n" );
    fprintf( file, "\t</Points>\n" );
    fprintf( file, "\t<Cells>\n" );
    fprintf( file, "\t\t<DataArray type=\"Int32\" Name=\"connectivity\" "
                   "NumberOfComponents=\"1\" format=\"ascii\">\n" );
    for ( const int &conn : connectivity )
        fprintf( file, "%d ", conn );
    fprintf( file, "\n\t\t</DataArray>\n" );
    fprintf( file, "\t\t<DataArray type=\"Int32\" Name=\"offsets\" "
                   "NumberOfComponents=\"1\" format=\"ascii\">\n" );
    fprintf( file, "8\n" );
    fprintf( file, "\n\t\t</DataArray>\n" );
    fprintf( file, "\t\t<DataArray type=\"UInt8\" Name=\"types\" "
                   "NumberOfComponents=\"1\" format=\"ascii\">\n" );
    fprintf( file, "11\n" );
    fprintf( file, "\n\t\t</DataArray>\n" );
    fprintf( file, "\t</Cells>\n" );
    fprintf( file, "</Piece>\n" );
    fprintf( file, "</UnstructuredGrid>\n" );
    fprintf( file, "</VTKFile>\n" );
    fclose( file );
}

void writeParticlesParallelFile( MPI_Comm comm, const int step,
                                 std::string filename )
{
    // Should only be called from a single rank
    int size;
    MPI_Comm_size( comm, &size );
    // Prepare actual filename
    // todo(sschulz): Also separate filename construction into function
    size_t pos = 0;
    pos = filename.find( "*", pos );
    std::stringstream time_string;
    time_string << step;
    filename.replace( pos, 1, time_string.str() );
    std::string parallel_filename( filename );
    pos = 0;
    pos = parallel_filename.find( "%", pos );
    std::string empty_string( "" );
    parallel_filename.replace( pos, 1, empty_string );
    pos = 0;
    pos = parallel_filename.find( ".vtu", pos );
    parallel_filename.replace( pos, 4, ".pvtu" );
    FILE *file = fopen( parallel_filename.c_str(), "w" );
    fprintf( file, "<?xml version=\"1.0\"?>\n" );
    fprintf( file, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" "
                   "byte_order=\"LittleEndian\" header_type=\"UInt32\">\n" );
    fprintf( file, "<PUnstructuredGrid>\n" );
    fprintf( file, "\t<PPointData>\n" );
    fprintf( file, "\t\t<PDataArray type=\"Float64\" Name=\"Velocity\"/>\n" );
    fprintf( file, "\t\t<PDataArray type=\"Int32\" Name=\"Id\"/>\n" );
    fprintf( file, "\t\t<PDataArray type=\"Int32\" Name=\"Type\"/>\n" );
    fprintf( file, "\t</PPointData>\n" );
    fprintf( file, "\t<PCellData>\n" );
    fprintf( file, "\t</PCellData>\n" );
    fprintf( file, "\t<PPoints>\n" );
    fprintf( file, "\t\t<PDataArray type=\"Float64\" Name=\"Points\" "
                   "NumberOfComponents=\"3\"/>\n" );
    fprintf( file, "\t</PPoints>\n" );
    for ( int i = 0; i < size; ++i )
    {
        std::string piece_filename( filename );
        pos = 0;
        pos = piece_filename.find( "%", pos );
        std::stringstream rank_string;
        rank_string << "_" << i;
        piece_filename.replace( pos, 1, rank_string.str() );
        fprintf( file, "\t<Piece Source=\"%s\"/>\n", piece_filename.c_str() );
    }
    fprintf( file, "</PUnstructuredGrid>\n" );
    fprintf( file, "</VTKFile>\n" );
    fclose( file );
}

// Write particles to vtu file
// filename must contain * and % which will be replaced by time step and _rank.
// The filename dump_*%.vtu will be create the file dump_43_325.vtu in time
// step 43 on rank 325.
template <class t_System>
void writeParticles( MPI_Comm comm, const int step, t_System *system,
                     std::string filename, std::ofstream &err )
{
    int rank;
    MPI_Comm_rank( comm, &rank );
    // Write parallel file
    if ( rank == 1 )
        writeParticlesParallelFile( comm, step, filename );
    // Prepare actual filename
    // todo(sschulz): Separate filename construction into function
    size_t pos = 0;
    pos = filename.find( "*", pos );
    if ( std::string::npos == pos )
        log_err( err, "VTK output file does not contain required '*'" );
    std::stringstream time_string;
    time_string << step;
    filename.replace( pos, 1, time_string.str() );
    pos = 0;
    pos = filename.find( "%", pos );
    if ( std::string::npos == pos )
        log_err( err, "VTK output file does not contain required '%'" );
    std::stringstream rank_string;
    rank_string << "_" << rank;
    filename.replace( pos, 1, rank_string.str() );
    // Prepare data
    System<Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>,
           CabanaMD_LAYOUT>
        host_system;
    system->slice_x();
    auto x = system->x;
    host_system.resize( x.size() );
    host_system.slice_x();
    auto host_x = host_system.x;
    host_system.deep_copy( *system );
    host_system.slice_all();
    host_x = host_system.x;
    auto host_id = host_system.id;
    auto host_type = host_system.type;
    auto host_v = host_system.v;
    FILE *file = fopen( filename.c_str(), "w" );
    fprintf( file, "<?xml version=\"1.0\"?>\n" );
    fprintf( file, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" "
                   "byte_order=\"LittleEndian\" header_type=\"UInt32\">\n" );
    fprintf( file, "<UnstructuredGrid>\n" );
    fprintf( file, "<Piece NumberOfPoints=\"%d\" NumberOfCells=\"0\">\n",
             system->N_local );
    fprintf( file, "\t<PointData>\n" );
    fprintf( file, "\t\t<DataArray type=\"Float64\" Name=\"Velocity\" "
                   "NumberOfComponents=\"3\" format=\"ascii\">\n" );
    for ( int n = 0; n < system->N_local; ++n )
        fprintf( file, "%g %g %g ", host_v( n, 0 ), host_v( n, 1 ),
                 host_v( n, 2 ) );
    fprintf( file, "\n\t\t</DataArray>\n" );
    fprintf( file, "\t\t<DataArray type=\"Int32\" Name=\"Id\" "
                   "NumberOfComponents=\"1\" format=\"ascii\">\n" );
    for ( int n = 0; n < system->N_local; ++n )
        fprintf( file, "%d ", host_id( n ) );
    fprintf( file, "\n\t\t</DataArray>\n" );
    fprintf( file, "\t\t<DataArray type=\"Int32\" Name=\"Type\" "
                   "NumberOfComponents=\"1\" format=\"ascii\">\n" );
    for ( int n = 0; n < system->N_local; ++n )
        fprintf( file, "%d ", host_type( n ) );
    fprintf( file, "\n\t\t</DataArray>\n" );
    fprintf( file, "\t</PointData>\n" );
    fprintf( file, "\t<CellData>\n" );
    fprintf( file, "\t</CellData>\n" );
    fprintf( file, "\t<Points>\n" );
    fprintf( file, "\t\t<DataArray type=\"Float64\" Name=\"Points\" "
                   "NumberOfComponents=\"3\" format=\"ascii\">\n" );
    for ( int n = 0; n < system->N_local; ++n )
        fprintf( file, "%g %g %g ", host_x( n, 0 ), host_x( n, 1 ),
                 host_x( n, 2 ) );
    fprintf( file, "\n\t\t</DataArray>\n" );
    fprintf( file, "\t</Points>\n" );
    fprintf( file, "\t<Cells>\n" );
    fprintf( file, "\t\t<DataArray type=\"Int32\" Name=\"connectivity\" "
                   "NumberOfComponents=\"1\" format=\"ascii\">\n" );
    fprintf( file, "\n\t\t</DataArray>\n" );
    fprintf( file, "\t\t<DataArray type=\"Int32\" Name=\"offsets\" "
                   "NumberOfComponents=\"1\" format=\"ascii\">\n" );
    fprintf( file, "\n\t\t</DataArray>\n" );
    fprintf( file, "\t\t<DataArray type=\"UInt8\" Name=\"types\" "
                   "NumberOfComponents=\"1\" format=\"ascii\">\n" );
    fprintf( file, "\n\t\t</DataArray>\n" );
    fprintf( file, "\t</Cells>\n" );
    fprintf( file, "</Piece>\n" );
    fprintf( file, "</UnstructuredGrid>\n" );
    fprintf( file, "</VTKFile>\n" );
    fclose( file );
}

} // end namespace VTKWriter
#endif
