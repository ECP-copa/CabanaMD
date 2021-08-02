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

#ifndef LOADBALANCER_H
#define LOADBALANCER_H

#include <types.h>
#include <vtk_domain_writer.h>

#include <ALL.hpp>

#include <string>

#include <mpi.h>

template <class t_System>
class LoadBalancer
{
  public:
    LoadBalancer( t_System *system )
        : _comm( MPI_COMM_WORLD )
        , _system( system )
    {
        MPI_Comm_rank( _comm, &_rank );
        _liball =
            std::make_shared<ALL::ALL<double, double>>( ALL::TENSOR, 3, 0 );
        std::vector<int> rank_dim_pos( _system->rank_dim_pos.begin(),
                                       _system->rank_dim_pos.end() );
        std::vector<int> ranks_per_dim( _system->ranks_per_dim.begin(),
                                        _system->ranks_per_dim.end() );
        _liball->setProcGridParams( rank_dim_pos, ranks_per_dim );
        // todo(sschulz): Do we need a minimum domain size?
        // _liball->setMinimumDomainSize(..);
        _liball->setCommunicator( _comm );
        _liball->setProcTag( _rank );
        _liball->setup();

        std::vector<ALL::Point<double>> lb_vertices( 2,
                                                     ALL::Point<double>( 3 ) );
        lb_vertices.at( 0 )[0] = _system->local_mesh_lo_x;
        lb_vertices.at( 0 )[1] = _system->local_mesh_lo_y;
        lb_vertices.at( 0 )[2] = _system->local_mesh_lo_z;
        lb_vertices.at( 1 )[0] = _system->local_mesh_hi_x;
        lb_vertices.at( 1 )[1] = _system->local_mesh_hi_y;
        lb_vertices.at( 1 )[2] = _system->local_mesh_hi_z;
        _liball->setVertices( lb_vertices );
    }

    void balance()
    {
        int rank;
        MPI_Comm_rank( _comm, &rank );
        double work = ( _system->N_local + _system->N_ghost ) / ( rank + 1 );
        printf( ">> Work: %g\n", work );
        _liball->setWork( work );
        _liball->balance();
        std::vector<ALL::Point<double>> updated_vertices =
            _liball->getVertices();
        printf( ">> New Vertices: %g %g %g %g %g %g\n",
                updated_vertices.at( 0 )[0], updated_vertices.at( 0 )[1],
                updated_vertices.at( 0 )[2], updated_vertices.at( 1 )[0],
                updated_vertices.at( 1 )[1], updated_vertices.at( 1 )[2] );
        std::array<double, 3> low_corner = { updated_vertices.at( 0 )[0],
                                             updated_vertices.at( 0 )[1],
                                             updated_vertices.at( 0 )[2] };
        std::array<double, 3> high_corner = { updated_vertices.at( 1 )[0],
                                              updated_vertices.at( 1 )[1],
                                              updated_vertices.at( 1 )[2] };
        _system->update_domain( low_corner, high_corner );
    }

    void output( const int t ) const
    {
        std::string vtk_actual_domain_basename( "domain_act" );
        std::string vtk_lb_domain_basename( "domain_lb" );
        // _liball->printVTKoutlines( t );

        std::array<double, 6> vertices = {
            _system->local_mesh_lo_x, _system->local_mesh_lo_y,
            _system->local_mesh_lo_z, _system->local_mesh_hi_x,
            _system->local_mesh_hi_y, _system->local_mesh_hi_z };
        VTKDomainWriter::writeDomain( _comm, t, vertices,
                                      vtk_actual_domain_basename );

        std::vector<ALL::Point<double>> updated_vertices =
            _liball->getVertices();
        for ( std::size_t d = 0; d < 3; ++d )
            vertices[d] = updated_vertices.at( 0 )[d];
        for ( std::size_t d = 3; d < 6; ++d )
            vertices[d] = updated_vertices.at( 1 )[d - 3];
        VTKDomainWriter::writeDomain( _comm, t, vertices,
                                      vtk_lb_domain_basename );
    }

  private:
    MPI_Comm _comm;
    t_System *_system;
    std::shared_ptr<ALL::ALL<double, double>> _liball;
    int _rank;
};

#endif
