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

/* Ewald solver */

template <class t_System, class t_Neighbor>
ForceEwald<t_System, t_Neighbor>::ForceEwald( t_System *system )
    : Force<t_System, t_Neighbor>( system )
{
    _alpha = 0.0;
    _k_max = 0.0;
    _r_max = 0.0;

    std::vector<int> dims( 3 );
    std::vector<int> periods( 3 );

    dims.at( 0 ) = dims.at( 1 ) = dims.at( 2 ) = 0;
    periods.at( 0 ) = periods.at( 1 ) = periods.at( 2 ) = 1;

    int n_ranks;
    MPI_Comm_size( MPI_COMM_WORLD, &n_ranks );

    MPI_Dims_create( n_ranks, 3, dims.data() );

    MPI_Cart_create( MPI_COMM_WORLD, 3, dims.data(), periods.data(), 0,
                     &cart_comm );

    int rank;
    MPI_Comm_rank( cart_comm, &rank );

    std::vector<int> loc_coords( 3 );
    std::vector<int> cart_dims( 3 );
    std::vector<int> cart_periods( 3 );
    MPI_Cart_get( cart_comm, 3, cart_dims.data(), cart_periods.data(),
                  loc_coords.data() );

    std::vector<int> neighbor_low( 3 );
    std::vector<int> neighbor_up( 3 );
    for ( int dim = 0; dim < 3; ++dim )
        MPI_Cart_shift( cart_comm, dim, 1, &neighbor_low.at( dim ),
                        &neighbor_up.at( dim ) );
}

// TODO: allow user to specify parameters
template <class t_System, class t_Neighbor>
void ForceEwald<t_System, t_Neighbor>::init_coeff( t_System *system,
                                                   char **args )
{
    double accuracy = atof( args[2] );
    tune( system, accuracy );
}

template <class t_System, class t_Neighbor>
void ForceEwald<t_System, t_Neighbor>::tune( t_System *system, double accuracy )
{
    if ( system->domain_x != system->domain_y or
         system->domain_x != system->domain_z )
        throw std::runtime_error(
            "Ewald needs symmetric system size for now." );
    const int N = system->N_local;
    lx = system->domain_x;
    ly = system->domain_y;
    lz = system->domain_z;
    // Fincham 1994, Optimisation of the Ewald Sum for Large Systems
    // only valid for cubic systems (needs adjustement for non-cubic systems)
    constexpr double EXECUTION_TIME_RATIO_K_R = 2.0; // TODO: Change?
    double p = -log( accuracy );
    double tune_factor =
        pow( EXECUTION_TIME_RATIO_K_R, 1.0 / 6.0 ) * sqrt( p / PI );

    _r_max = tune_factor / pow( N, 1.0 / 6.0 ) * lx;
    // TODO: make the maximum cut_off dependent on the domain size
    double comp_size =
        ( 5.0 > 0.75 * system->domain_x ) ? 0.75 * system->domain_x : 5.0;
    comp_size = 20.0;
    if ( _r_max > comp_size )
        tune_factor = pow( N, 1.0 / 6.0 ) * 0.91 * comp_size / lx;

    _r_max = tune_factor / pow( N, 1.0 / 6.0 ) * lx;
    _alpha = tune_factor * pow( N, 1.0 / 6.0 ) / lx;
    _k_max = tune_factor * pow( N, 1.0 / 6.0 ) / lx * 2.0 * PI;
    _alpha = sqrt( p ) / _r_max;
    _k_max = 2.0 * sqrt( p ) * _alpha;
}

template <class t_System, class t_Neighbor>
void ForceEwald<t_System, t_Neighbor>::compute( t_System *system,
                                                t_Neighbor *neighbor )
{
    // Per-atom slices
    system->slice_force();
    auto x = system->x;
    auto f = system->f;
    auto type = system->type;
    system->slice_q();
    auto q = system->q;

    // Neighbor list
    auto neigh_list = neighbor->get();

    auto N_local = system->N_local;
    double alpha = _alpha;
    double k_max = _k_max;

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
    int n_kvec = ( 2 * k_int + 1 ) * ( 2 * k_int + 1 ) * ( 2 * k_int + 1 );

    U_trigonometric = Kokkos::View<T_F_FLOAT *, DeviceType>(
        "ForceEwald::U_trig", 2 * n_kvec );

    // Compute partial sums
    auto kspace_partial_sums = KOKKOS_LAMBDA( const int idx )
    {
        double qi = q( idx );

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
                        ( kz + k_int ) * ( 2 * k_int + 1 ) * ( 2 * k_int + 1 ) +
                        ( ky + k_int ) * ( 2 * k_int + 1 ) + ( kx + k_int );
                    // compute wave vector component
                    double _kx = 2.0 * PI / lx * (double)kx;
                    // compute dot product with local particle and wave
                    // vector
                    double kr = _kx * x( idx, 0 ) + _ky * x( idx, 1 ) +
                                _kz * x( idx, 2 );
                    // add contributions
                    Kokkos::atomic_add( &U_trigonometric( 2 * kidx ),
                                        qi * cos( kr ) );
                    Kokkos::atomic_add( &U_trigonometric( 2 * kidx + 1 ),
                                        qi * sin( kr ) );
                }
            }
        }
    };
    Kokkos::parallel_for( "ForceEwald::KspacePartial", N_local,
                          kspace_partial_sums );
    Kokkos::fence();

    // reduce the partial results
    MPI_Allreduce( MPI_IN_PLACE, U_trigonometric.data(), 2 * n_kvec, MPI_DOUBLE,
                   MPI_SUM, cart_comm );

    const double ELECTRON_CHARGE = 1.60217662E-19; // electron charge in
                                                   // Coulombs
    const double EPS_0 =
        8.8541878128E-22; // permittivity of free space in Farads/Angstrom
    const double ANGSTROMS = 1E-10; // convert meters to Angstroms
    const double FORCE_CONVERSION_FACTOR =
        ANGSTROMS * ELECTRON_CHARGE / ( 4.0 * PI * EPS_0 );

    auto kspace_force = KOKKOS_LAMBDA( const int idx )
    {
        double k[3];

        double qi = q( idx );

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
                    int kidx =
                        ( kz + k_int ) * ( 2 * k_int + 1 ) * ( 2 * k_int + 1 ) +
                        ( ky + k_int ) * ( 2 * k_int + 1 ) + ( kx + k_int );
                    // compute wave vector component
                    k[0] = 2.0 * PI / lx * (double)kx;
                    // compute dot product of wave vector with itself
                    double kk = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];

                    // compute dot product with local particle and wave
                    // vector
                    double kr = k[0] * x( idx, 0 ) + k[1] * x( idx, 1 ) +
                                k[2] * x( idx, 2 );

                    // coefficient dependent on wave vector
                    double k_coeff = exp( -kk / ( 4 * alpha * alpha ) ) / kk;

                    for ( int dim = 0; dim < 3; ++dim )
                        f( idx, dim ) +=
                            k_coeff * 2.0 * qi * k[dim] *
                            FORCE_CONVERSION_FACTOR *
                            ( U_trigonometric( 2 * kidx + 1 ) * cos( kr ) -
                              U_trigonometric( 2 * kidx ) * sin( kr ) );
                }
            }
        }
    };
    Kokkos::parallel_for( "ForceEwald::Kspace", N_local, kspace_force );
    Kokkos::fence();

    auto realspace_force = KOKKOS_LAMBDA( const int idx )
    {
        int num_n =
            Cabana::NeighborList<t_neigh_list>::numNeighbor( neigh_list, idx );

        double rx = x( idx, 0 );
        double ry = x( idx, 1 );
        double rz = x( idx, 2 );
        double qi = q( idx );

        for ( int ij = 0; ij < num_n; ++ij )
        {
            int j = Cabana::NeighborList<t_neigh_list>::getNeighbor( neigh_list,
                                                                     idx, ij );
            double dx = x( j, 0 ) - rx;
            double dy = x( j, 1 ) - ry;
            double dz = x( j, 2 ) - rz;
            double d = sqrt( dx * dx + dy * dy + dz * dz );
            double qj = q( j );

            // force computation
            double f_fact = qi * qj * FORCE_CONVERSION_FACTOR *
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
    Kokkos::parallel_for( N_local, realspace_force );
    Kokkos::fence();
}

template <class t_System, class t_Neighbor>
T_V_FLOAT
ForceEwald<t_System, t_Neighbor>::compute_energy( t_System *system,
                                                  t_Neighbor *neighbor )
{
    auto N_local = system->N_local;
    double alpha = _alpha;
    double k_max = _k_max;

    // Per-atom slices
    system->slice_x();
    auto x = system->x;
    system->slice_q();
    auto q = system->q;

    // Neighbor list
    auto neigh_list = neighbor->get();

    const double ELECTRON_CHARGE = 1.60217662E-19; // electron charge in
                                                   // Coulombs
    const double EPS_0 =
        8.8541878128E-22; // permittivity of free space in Farads/Angstrom
    const double ENERGY_CONVERSION_FACTOR =
        ELECTRON_CHARGE / ( 4.0 * PI * EPS_0 );

    int k_int = std::ceil( k_max ) + 1;

    T_V_FLOAT energy_k = 0.0;
    auto kspace_potential = KOKKOS_LAMBDA( const int idx, T_V_FLOAT &PE )
    {
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
                    int kidx =
                        ( kz + k_int ) * ( 2 * k_int + 1 ) * ( 2 * k_int + 1 ) +
                        ( ky + k_int ) * ( 2 * k_int + 1 ) + ( kx + k_int );
                    // compute wave vector component
                    k[0] = 2.0 * PI / lx * (double)kx;
                    // compute dot product of wave vector with itself
                    double kk = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];

                    // coefficient dependent on wave vector
                    double k_coeff = exp( -kk / ( 4 * alpha * alpha ) ) / kk;

                    // contribution to potential energy
                    PE += coeff * k_coeff *
                          ( U_trigonometric( 2 * kidx ) *
                                U_trigonometric( 2 * kidx ) +
                            U_trigonometric( 2 * kidx + 1 ) *
                                U_trigonometric( 2 * kidx + 1 ) ) *
                          ENERGY_CONVERSION_FACTOR;
                }
            }
        }
    };
    Kokkos::parallel_reduce( "ForceEwald::KspacePE", N_local, kspace_potential,
                             energy_k );
    Kokkos::fence();

    T_V_FLOAT energy_r = 0.0;
    auto realspace_potential = KOKKOS_LAMBDA( const int idx, T_V_FLOAT &PE )
    {
        int num_n =
            Cabana::NeighborList<t_neigh_list>::numNeighbor( neigh_list, idx );
        double rx = x( idx, 0 );
        double ry = x( idx, 1 );
        double rz = x( idx, 2 );
        double qi = q( idx );

        for ( int ij = 0; ij < num_n; ++ij )
        {
            int j = Cabana::NeighborList<t_neigh_list>::getNeighbor( neigh_list,
                                                                     idx, ij );
            double dx = x( j, 0 ) - rx;
            double dy = x( j, 1 ) - ry;
            double dz = x( j, 2 ) - rz;
            double d = sqrt( dx * dx + dy * dy + dz * dz );
            double qj = q( j );

            PE += ENERGY_CONVERSION_FACTOR * qi * qj * erfc( alpha * d ) / d;
        }

        // self-energy contribution
        PE += -alpha / PI_SQRT * qi * qi * ENERGY_CONVERSION_FACTOR;
    };
    Kokkos::parallel_reduce( "ForceEwald::RealSpacePE", N_local,
                             realspace_potential, energy_r );
    Kokkos::fence();
    // Not including dipole correction (usually unnecessary)
    T_V_FLOAT energy = energy_k + energy_r;
    return energy;
}

template <class t_System, class t_Neighbor>
const char *ForceEwald<t_System, t_Neighbor>::name()
{
    return "LongRangeForce:Ewald";
}
