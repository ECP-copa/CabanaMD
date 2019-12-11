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

#include<force_ewald_cabana_neigh.h>

constexpr double PI(3.141592653589793238462643);
constexpr double PI_SQRT(1.772453850905516);
constexpr double PI_SQ(PI*PI);// 9.869604401089359
constexpr double PI_DIV_SQ(1.0/PI_SQ);//0.101321183642338


template<class t_neighbor>
ForceEwald<t_neighbor>::ForceEwald(System* system, bool half_neigh_):Force(system,half_neigh_) {
    half_neigh = half_neigh_;
    assert( half_neigh == true );

    std::vector<int> dims( 3 );
    std::vector<int> periods( 3 );

    dims.at( 0 ) = dims.at( 1 ) = dims.at( 2 ) = 0;
    periods.at( 0 ) = periods.at( 1 ) = periods.at( 2 ) = 1;

    int n_ranks;
    MPI_Comm_size( MPI_COMM_WORLD, &n_ranks );

    MPI_Dims_create( n_ranks, 3, dims.data() );

    MPI_Comm cart_comm;
    MPI_Cart_create( MPI_COMM_WORLD, 3, dims.data(), periods.data(), 0,
                         &cart_comm );

    int rank;
    MPI_Comm_rank( cart_comm, &rank );

    std::vector<int> loc_coords( 3 );
    std::vector<int> cart_dims( 3 );
    std::vector<int> cart_periods( 3 );
    MPI_Cart_get( cart_comm, 3, cart_dims.data(), cart_periods.data(),
                  loc_coords.data() );


    //MPI_Topo_test( comm, &comm_type );
    //assert( comm_type == MPI_CART );
    this->comm = comm;
    
}

//initialize Ewald params if given from input deck
template<class t_neighbor>
void ForceEwald<t_neighbor>::init_coeff(char** args) {

  double alpha = atof(args[3]);
  double rmax = atof(args[4]);
  double kmax = atof(args[5]);
}
//TODO: overload initialization to include tuning when not given params

//TODO: Use create_neigh_list just like shortrange forces
template<class t_neighbor>
void ForceEwald<t_neighbor>::create_neigh_list(System* system) {
//  N_local = system->N_local;
//
//  double grid_min[3] = {system->sub_domain_lo_x - system->sub_domain_x,
//                        system->sub_domain_lo_y - system->sub_domain_y,
//                        system->sub_domain_lo_z - system->sub_domain_z};
//  double grid_max[3] = {system->sub_domain_hi_x + system->sub_domain_x,
//                        system->sub_domain_hi_y + system->sub_domain_y,
//                        system->sub_domain_hi_z + system->sub_domain_z};
//
//  auto x = Cabana::slice<Positions>(system->xvf);
//
//  t_neighbor list( x, 0, N_local, neigh_cut, 1.0, grid_min, grid_max );
//  neigh_list = list;
}

template<class t_neighbor>
void ForceEwald<t_neighbor>::compute(System* system) {

  double Ur = 0.0, Uk = 0.0, Uself = 0.0, Udip = 0.0;
  double Udip_vec[3];
  double N_local, N_max;

  N_local = system->N_local;
  N_max = system->N_max;
  x = Cabana::slice<Positions>(system->xvf);
  f = Cabana::slice<Forces>(system->xvf);
  //f_a = Cabana::slice<Forces>(system->xvf);
  id = Cabana::slice<IDs>(system->xvf);
  //type = Cabana::slice<Types>(system->xvf);
  q = Cabana::slice<Charges>(system->xvf);
  p = Cabana::slice<Potentials>(system->xvf);//TODO: Add potentials as part of the system AoSoA?

  //This seems awkward. Could remove?
  Kokkos::View<double *, MemorySpace> domain_length( "domain length", 6 );
  domain_length( 0 ) = system->sub_domain_lo_x;
  domain_length( 1 ) = system->sub_domain_hi_x;
  domain_length( 2 ) = system->sub_domain_lo_y;
  domain_length( 3 ) = system->sub_domain_hi_y;
  domain_length( 4 ) = system->sub_domain_lo_z;
  domain_length( 5 ) = system->sub_domain_hi_z;

  // compute subdomain size and make it available in the kernels
  Kokkos::View<double *, MemorySpace> sub_domain_size( "sub_domain size", 3);
  sub_domain_size( 0 ) = system->sub_domain_x;
  sub_domain_size( 1 ) = system->sub_domain_y;
  sub_domain_size( 2 ) = system->sub_domain_z;
  //sub_domain_size( 0 ) = (system->sub_domain_hi_x + system->sub_domain_x) - (system_sub_domain_lo_x - system_sub_domain_x);
  //sub_domain_size( 1 ) = (system->sub_domain_hi_y + system->sub_domain_y) - (system_sub_domain_lo_y - system_sub_domain_y);
  //sub_domain_size( 2 ) = (system->sub_domain_hi_z + system->sub_domain_z) - (system_sub_domain_lo_z - system_sub_domain_z);
  //sub_domain_size( 1 ) = system->subdomain_hi_y - system_subdomain_y;
  //sub_domain_size( 2 ) = system->subdomain_hi_z - system_subdomain_z;
 
  // compute domain size and make it available in the kernels
  Kokkos::View<double *, MemorySpace> domain_size( "domain size", 3 );
  domain_size( 0 ) = system->domain_x;//TODO: Seems very nnecessary. Remove
  domain_size( 1 ) = system->domain_y;
  domain_size( 2 ) = system->domain_z;
  //domain_size( 0 ) = (system->domain_hi_x + system->domain_x) - (system->domain_lo_x - system->domain_x);
  //domain_size( 1 ) = (system->domain_hi_y + system->domain_y) - (system->domain_lo_y - system->domain_y);
  //domain_size( 2 ) = (system->domain_hi_z + system->domain_z) - (system->domain_lo_z - system->domain_z);
 
  //get the solver parameters
  double alpha = _alpha;
  double r_max = _r_max;
  double eps_r = _eps_r;
  double k_max = _k_max;

  // store MPI information
  int rank, n_ranks;
  std::vector<int> loc_dims( 3 );
  std::vector<int> cart_dims( 3 );
  std::vector<int> cart_periods( 3 );
  MPI_Comm_rank( comm, &rank );
  MPI_Comm_size( comm, &n_ranks );
  MPI_Cart_get( comm, 3, cart_dims.data(), cart_periods.data(),
                loc_dims.data() );

  // neighbor information
  std::vector<int> neighbor_low( 3 );
  std::vector<int> neighbor_up( 3 );

  // get neighbors in parallel decomposition
  for ( int dim = 0; dim < 3; ++dim )
  {
      MPI_Cart_shift( comm, dim, 1, &neighbor_low.at( dim ),
                      &neighbor_up.at( dim ) );
  }

  // initialize potential and force to zero
  auto init_parameters = KOKKOS_LAMBDA( const int idx )
    {
        p( idx ) = 0.0;
        f( idx, 0 ) = 0.0;
        f( idx, 1 ) = 0.0;
        f( idx, 2 ) = 0.0;
    }; 
  Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, N_max ), init_parameters );
  Kokkos::fence();

  // In order to compute the k-space contribution in parallel
  // first the following sums need to be created for each
  // k-vector:
  //              sum(1<=i<=N_part) sin/cos (dot(k,r_i))
  // This can be achieved by computing partial sums on each
  // MPI process, reducing them over all processes and
  // afterward using the pre-computed values to compute
  // the forces and potentials acting on the particles
  // in parallel independently again.
 
  // determine number of required sine / cosine values  
  int k_int = std::ceil( k_max ) + 1;
  int n_kvec = ( 2 * k_int + 1 ) * ( 2 * k_int ) * ( 2 * k_int + 1 );

  //allocate View to store them
  Kokkos::View<double *, MemorySpace> U_trigonometric(
      "sine and cosine contributions", 2 * n_kvec );

  //set all values to zero
  Kokkos::parallel_for( 2 * n_kvec, KOKKOS_LAMBDA( const int idx ) {
      U_trigonometric( idx ) = 0.0;
  } );
  Kokkos::fence();
  
  double lx = domain_size(0);
  double ly = domain_size(1);
  double lz = domain_size(2);

  //Compute partial sums
  auto partial_sums = KOKKOS_LAMBDA( const int idx )
  {
        for ( int kz = -k_int; kz <= k_int; ++kz )
        {
            // compute wave vector component
            double _kz = 2.0 * PI / lz * (double)kz;
            for ( int ky = -k_int; ky <= k_int; ++ky )
            {
                // compute wave vector component
                double _ky = 2.0 * PI / ly * (double)ky;
                for ( int kx = -k_int; kx <= k_int; ++kx )
                {
                    // no values required for the central box
                    if ( kx == 0 && ky == 0 && kz == 0 )
                       continue;
                    // compute index in contribution array
                    int kidx =
                        ( kz + k_int ) * ( 2 * k_int + 1 ) * ( 2 * k_int + 1 ) * ( 2 * k_int + 1 ) +
                        ( ky + k_int ) * ( 2 * k_int + 1 ) + ( kx + k_int );
                    // compute wave vector component
                    double _kx = 2.0 * PI / lx * (double)kx;
                    // compute dot product with local particle and wave
                    // vector
                    double kr = _kz * x( idx, 0 ) + _ky * x( idx, 1 ) + _kz * x( idx, 2 );
                    //add contributions
                    Kokkos::atomic_add( &U_trigonometric( 2 * kidx ), q( idx ) * cos( kr ) );
                    Kokkos::atomic_add( &U_trigonometric( 2 * kidx + 1 ), q( idx ) * sin( kr ) );
                }
            }
        }
    };
    Kokkos::parallel_for( N_max, partial_sums );
    Kokkos::fence();

    //reduce the partial results

    double *U_trigon_array = new double[2 * n_kvec];
    for ( int idx = 0; idx < 2 * n_kvec; ++idx )
        U_trigon_array[idx] = U_trigonometric( idx );

    MPI_Allreduce( MPI_IN_PLACE, U_trigon_array, 2 * n_kvec, MPI_DOUBLE,
                   MPI_SUM, comm );

    for ( int idx = 0; idx < 2 * n_kvec; ++idx )
        U_trigonometric( idx ) = U_trigon_array[idx];

    delete[] U_trigon_array;

    // In orig Ewald this was reduction to Uk
    // Now, it's a parallel_for to update each p(idx)
    auto kspace_potential = KOKKOS_LAMBDA( const int idx ) {
        // general coefficient
        double coeff = 4.0 * PI / ( lx * ly * lz );
        double k[3];
        
        for ( int kz = -k_int; kz <= k_int; ++kz )
        {
            // compute wave vector component
            k[2] = 2.0 * PI / lz * (double)kz;
            for ( int ky = -k_int; ky <= k_int; ++ky )
            {
                // compute wave vector component
                k[1] = 2.0 * PI / ly * (double)ky;
                for ( int kx = -k_int; kx <= k_int; ++kx )
                {
                    // no values required for the central box
                    if ( kx == 0 && ky == 0 && kz == 0 )
                        continue;
                        // compute index in contribution array
                        int kidx = ( kz + k_int ) * ( 2 * k_int + 1 ) *
                                       ( 2 * k_int + 1 ) +
                                   ( ky + k_int ) * ( 2 * k_int + 1 ) +
                                   ( kx + k_int );
                        // compute wave vector component
                        k[0] = 2.0 * PI / lx * (double)kx;
                        // compute dot product of wave vector with itself
                        double kk = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                        ;
                        // compute dot product with local particle and wave
                        // vector
                        double kr = k[0] * x( idx, 0 ) + k[1] * x( idx, 1 ) +
                                    k[2] * x( idx, 2 );

                        // coefficient dependent on wave vector
                        double k_coeff =
                            exp( -kk / ( 4 * alpha * alpha ) ) / kk;

                        // contribution to potential energy
                        double contrib =
                            coeff * k_coeff *
                            ( U_trigonometric( 2 * kidx ) *
                                  U_trigonometric( 2 * kidx ) +
                              U_trigonometric( 2 * kidx + 1 ) *
                                  U_trigonometric( 2 * kidx + 1 ) );
                        p( idx ) += contrib;
                        //Uk_part += contrib;

                        for ( int dim = 0; dim < 3; ++dim )
                            f( idx, dim ) +=
                                k_coeff * 2.0 * q( idx ) * k[dim] *
                                ( U_trigonometric( 2 * kidx + 1 ) * cos( kr ) -
                                  U_trigonometric( 2 * kidx ) * sin( kr ) );
                }
            }
        }
    };
    Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, N_max ), kspace_potential );
    Kokkos::fence();

    //MPI_Allreduce( MPI_IN_PLACE, &Uk, 1, MPI_DOUBLE, MPI_SUM, comm );

    // computation real-space contribution
    //
    // In order to compute the real-space contribution to potentials and
    // forces the Cabana implementation of halos and Verlet lists is
    // used. The halos are used to communicate particles along the
    // borders of MPI domains to their respective neighbors, so that
    // complete Verlet lists can be created. To save computation time
    // the half shell variant is used, that means that Newton's third
    // law of motion is used: F(i,j) = -F(j,i). The downside of this
    // is that the computed partial forces and potentials of the
    // ghost particles need to be communicated back to the source
    // process, which is done by using the 'scatter' implementation
    // of Cabana.
    // TODO: Enable re-use of CabanaMD neighborlist from short-range forces

    
    // lower end of system
    double grid_min[3] = {system->sub_domain_lo_x - system->sub_domain_x,
                          system->sub_domain_lo_y - system->sub_domain_y,
                          system->sub_domain_lo_z - system->sub_domain_z};
    // upper end of system
    double grid_max[3] = {system->sub_domain_hi_x + system->sub_domain_x,
                          system->sub_domain_hi_y + system->sub_domain_y,
                          system->sub_domain_hi_z + system->sub_domain_z};

    //double grid_min[3] = {-r_max + domain_length( 0 ),
    //                      -r_max + domain_length( 2 ),
    //                      -r_max + domain_length( 4 )};

    using ListAlgorithm = Cabana::HalfNeighborTag;
    using ListType =
        Cabana::VerletList<DeviceType, ListAlgorithm, Cabana::VerletLayoutCSR>;

    // store the number of local particles
    //int n_local = n_max;
    // offset due to the previously received particles
    int offset = 0;

    // communicate particles along the edges of the system

    // six halo regions required for transport in all directions
    std::vector<int> n_export = {0, 0, 0, 0, 0, 0};
    std::vector<Cabana::Halo<DeviceType> *> halos( 6 );

    // do three-step communication, x -> y -> z
    for ( int dim = 0; dim < 3; ++dim )
    {
        // check if the cut-off is not larger then two times the system size
        assert( r_max <= 2.0 * domain_size( dim ) );

        // find out how many particles are close to the -dim border
        Kokkos::parallel_reduce( N_max,
                                 KOKKOS_LAMBDA( const int idx, int &low ) {
                                     low += ( x( idx, dim ) <=
                                              domain_length( 2 * dim ) + r_max )
                                                ? 1
                                                : 0;
                                 },
                                 n_export.at( 2 * dim ) );
        Kokkos::fence();

        // find out how many particles are close to the +dim border
        Kokkos::parallel_reduce(
            N_max,
            KOKKOS_LAMBDA( const int idx, int &up ) {
                up += ( x( idx, dim ) >= domain_length( 2 * dim + 1 ) - r_max )
                          ? 1
                          : 0;
            },
            n_export.at( 2 * dim + 1 ) );
        Kokkos::fence();
        // list with the ranks and target processes for the halo
        Kokkos::View<int *, DeviceType> export_ranks_low(
            "export_ranks_low", n_export.at( 2 * dim ) );
        Kokkos::View<int *, DeviceType> export_ranks_up(
            "export_ranks_up", n_export.at( 2 * dim + 1 ) );

        Kokkos::View<int *, DeviceType> export_ids_low(
            "export_ids_low", n_export.at( 2 * dim ) );
        Kokkos::View<int *, DeviceType> export_ids_up(
            "export_ids_up", n_export.at( 2 * dim + 1 ) );

        // fill the export arrays for the halo construction
        int idx_up = 0, idx_low = 0;
        for ( int idx = 0; idx < N_max; ++idx )
        {
            if ( x( idx, dim ) <= domain_length( 2 * dim ) + r_max )
            {
                export_ranks_low( idx_low ) = neighbor_low.at( dim );
                export_ids_low( idx_low ) = idx;
                ++idx_low;
            }
            if ( x( idx, dim ) >= domain_length( 2 * dim + 1 ) - r_max )
            {
                export_ranks_up( idx_up ) = neighbor_up.at( dim );
                export_ids_up( idx_up ) = idx;
                ++idx_up;
            }
        }

        // create neighbor list
        std::vector<int> neighbors = {neighbor_low.at( dim ), rank,
                                      neighbor_up.at( dim )};
        std::sort( neighbors.begin(), neighbors.end() );
        auto unique_end = std::unique( neighbors.begin(), neighbors.end() );
        neighbors.resize( std::distance( neighbors.begin(), unique_end ) );
        halos.at( 2 * dim ) = new Cabana::Halo<DeviceType>(
            comm, N_max, export_ids_low, export_ranks_low, neighbors );
        N_max += halos.at( 2 * dim )->numGhost();

        // resize particle list to contain all particles
        system->xvf.resize( N_max );

        // update slices
        x = Cabana::slice<Positions>(system->xvf);
        f = Cabana::slice<Forces>(system->xvf);
        //f_a = Cabana::slice<Forces>(system->xvf);
        id = Cabana::slice<IDs>(system->xvf);
        //type = Cabana::slice<Types>(system->xvf);
        q = Cabana::slice<Charges>(system->xvf);
        p = Cabana::slice<Potentials>(system->xvf);

        // gather data for halo regions
        Cabana::gather( *( halos.at( 2 * dim ) ), x );
        Cabana::gather( *( halos.at( 2 * dim ) ), q );
        Cabana::gather( *( halos.at( 2 * dim ) ), p );
        Cabana::gather( *( halos.at( 2 * dim ) ), f );
        Cabana::gather( *( halos.at( 2 * dim ) ), id );

        // do periodic corrections and reset partial forces
        // and potentials of ghost particles
        // (they are accumulated during the scatter step)
        for ( int idx = N_local + offset;
              (std::size_t)idx <
              N_local + offset + halos.at( 2 * dim )->numGhost();
              ++idx )
        {
            p( idx ) = 0.0;
            f( idx, 0 ) = 0.0;
            f( idx, 1 ) = 0.0;
            f( idx, 2 ) = 0.0;
            if ( loc_dims.at( dim ) == cart_dims.at( dim ) - 1 )
            {
                x( idx, dim ) += domain_size( dim );
            }
        }

        offset += halos.at( 2 * dim )->numGhost();

        // do transfer of particles in upward direction
        halos.at( 2 * dim + 1 ) = new Cabana::Halo<DeviceType>(
            comm, N_max, export_ids_up, export_ranks_up, neighbors );
        N_max += halos.at( 2 * dim + 1 )->numGhost();

        // resize particle list to contain all particles
        system->xvf.resize( N_max );

        // update slices
        x = Cabana::slice<Positions>(system->xvf);
        f = Cabana::slice<Forces>(system->xvf);
        //f_a = Cabana::slice<Forces>(system->xvf);
        id = Cabana::slice<IDs>(system->xvf);
        //type = Cabana::slice<Types>(system->xvf);
        q = Cabana::slice<Charges>(system->xvf);
        p = Cabana::slice<Potentials>(system->xvf);

        // gather data for halo regions

        Cabana::gather( *( halos.at( 2 * dim + 1 ) ), x );
        Cabana::gather( *( halos.at( 2 * dim + 1 ) ), q );
        Cabana::gather( *( halos.at( 2 * dim + 1 ) ), p );
        Cabana::gather( *( halos.at( 2 * dim + 1 ) ), f );
        Cabana::gather( *( halos.at( 2 * dim + 1 ) ), id );

        // do periodic corrections and reset partial forces
        // and potentials of ghost particles
        // (they are accumulated during the scatter step)
        for ( int idx = N_local + offset;
              (std::size_t)idx <
              N_local + offset + halos.at( 2 * dim + 1 )->numGhost();
              ++idx )
        {
            p( idx ) = 0.0;
            f( idx, 0 ) = 0.0;
            f( idx, 1 ) = 0.0;
            f( idx, 2 ) = 0.0;
            if ( loc_dims.at( dim ) == 0 )
            {
                x( idx, dim ) -= domain_size( dim );
            }
        }

        offset += halos.at( 2 * dim + 1 )->numGhost();
    }

    // create VerletList to iterate over

    double min_coords[3];
    double max_coords[3];

    for ( int dim = 0; dim < 3; ++dim )
    {
        max_coords[dim] = -2.0 * r_max;
        min_coords[dim] = domain_size( dim ) + 2.0 * r_max;
    }

    for ( int idx = 0; idx < N_max; ++idx )
    {
        for ( int dim = 0; dim < 3; ++dim )
        {
            if ( x( idx, dim ) < min_coords[dim] )
                min_coords[dim] = x( idx, dim );
            if ( x( idx, dim ) > max_coords[dim] )
                max_coords[dim] = x( idx, dim );
        }
    }

    // compute cell size in a way that not too many cells are used
    double cell_size;
    cell_size =
        std::max( std::min( ( grid_max[0] - grid_min[0] ) / 20.0,
                            std::min( ( grid_max[1] - grid_min[1] ) / 20.0,
                                      ( grid_max[2] - grid_min[2] ) / 20.0 ) ),
                  1.0 );

    ListType verlet_list( x, 0, N_local, r_max, cell_size, grid_min, grid_max );

    auto force_contribs = KOKKOS_LAMBDA( const int idx ) 
    { 
        int num_n =
            Cabana::NeighborList<ListType>::numNeighbor( verlet_list, idx );

        double rx = x( idx, 0 );
        double ry = x( idx, 1 );
        double rz = x( idx, 2 );

        for ( int ij = 0; ij < num_n; ++ij )
        {
            int j = Cabana::NeighborList<ListType>::getNeighbor( verlet_list,
                                                                 idx, ij );
            double dx = x( j, 0 ) - rx;
            double dy = x( j, 1 ) - ry;
            double dz = x( j, 2 ) - rz;
            double d = sqrt( dx * dx + dy * dy + dz * dz );

            // potential computation
            double contrib = 0.5 * q( idx ) * q( j ) * erfc( alpha * d ) / d;
            Kokkos::atomic_add( &p( idx ), contrib );
            Kokkos::atomic_add( &p( j ), contrib );

            // force computation
           double f_fact = q( idx ) * q( j ) *
                            ( 2.0 * sqrt( alpha / PI ) * exp( -alpha * d * d ) +
                              erfc( sqrt( alpha ) * d ) ) /
                            ( d * d );
            Kokkos::atomic_add( &f( idx, 0 ), f_fact * dx );
            Kokkos::atomic_add( &f( idx, 1 ), f_fact * dy );
            Kokkos::atomic_add( &f( idx, 2 ), f_fact * dz );
            Kokkos::atomic_add( &f( j, 0 ), -f_fact * dx );
            Kokkos::atomic_add( &f( j, 1 ), -f_fact * dy );
            Kokkos::atomic_add( &f( j, 2 ), -f_fact * dz );
        }
    };
    Kokkos::parallel_for( N_local, force_contribs );
    Kokkos::fence();


    // send the force and potential contributions of the
    // ghost particles back to the origin processes
    for ( int n_halo = 5; n_halo >= 0; --n_halo )
    {
        Cabana::scatter( *( halos.at( n_halo ) ), p );
        Cabana::scatter( *( halos.at( n_halo ) ), f );

        N_max -= halos.at( n_halo )->numGhost();
        system->xvf.resize( N_max );
        // update slices
        x = Cabana::slice<Positions>(system->xvf);
        f = Cabana::slice<Forces>(system->xvf);
        //f_a = Cabana::slice<Forces>(system->xvf);
        id = Cabana::slice<IDs>(system->xvf);
        //type = Cabana::slice<Types>(system->xvf);
        q = Cabana::slice<Charges>(system->xvf);
        p = Cabana::slice<Potentials>(system->xvf);
    }

    // check if the particle array was reduced to the correct size again
    assert( N_max == N_local );
    //TODO: Fix translations between N_max,N_local here and in Rene's code


    // computation of self-energy contribution
    auto calc_Uself = KOKKOS_LAMBDA( int idx )
    {
        p( idx ) += -alpha / PI_SQRT * q( idx ) * q( idx );
    };
    Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, N_max ), calc_Uself );
    Kokkos::fence();

    //Not including dipole correction (usually unnecessary)

}


template<class t_neighbor>
T_V_FLOAT ForceEwald<t_neighbor>::compute_energy(System* system) {

  //step++;
  return 0.0;
}

template<class t_neighbor>
const char* ForceEwald<t_neighbor>::name() {return "Ewald";}

