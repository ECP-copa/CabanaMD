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

namespace VTKDomainWriter
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
    fprintf( file, "\t</PCellData>\n" );
    fprintf( file, "\t<PPoints>\n" );
    fprintf( file, "\t\t<PDataArray type=\"Float64\" Name=\"Points\" "
                   "NumberOfComponents=\"3\"/>\n" );
    fprintf( file, "\t</PPoints>\n" );
    for ( std::size_t i = 0; i < size; ++i )
        fprintf( file, "\t<Piece Source=\"%s_%d_%lu.vtu\"/>\n",
                 basename.c_str(), time_step, i );
    fprintf( file, "</PUnstructuredGrid>\n" );
    fprintf( file, "</VTKFile>\n" );
    fclose( file );
}
// Write VTU for domain (low corner, high corner)
void writeDomain( MPI_Comm comm, int time_step,
                  std::array<double, 6> &domain_vertices,
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

} // end namespace VTKDomainWriter
#endif
