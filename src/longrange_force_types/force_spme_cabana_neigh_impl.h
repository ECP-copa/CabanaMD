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

#include <Cajita.hpp>
#ifdef Cabana_ENABLE_Cuda
#include <cufft.h>
#include <cufftw.h>
#else
#include <fftw3.h>
#endif

#include <mpi.h>

/* Smooth particle mesh Ewald (SPME) solver
 * - This method, from Essman et al. (1995) computes long-range Coulombic forces
 *   with O(nlogN) scaling by using 3D FFT and interpolation to a mesh for the
 *   reciprocal space part of the Ewald sum.
 */

template <class t_System, class t_Neighbor>
ForceSPME<t_System, t_Neighbor>::ForceSPME( t_System *system )
    : Force<t_System, t_Neighbor>( system )
{
    _alpha = 0.0;
    _k_max = 0.0;
    _r_max = 0.0;
}

// TODO: allow user to specify parameters
template <class t_System, class t_Neighbor>
void ForceSPME<t_System, t_Neighbor>::init_coeff(
    std::vector<std::vector<std::string>> args )
{
    accuracy = std::stod( args[2] );
}

template <class t_System, class t_Neighbor>
void ForceSPME<t_System, t_Neighbor>::init_longrange( t_System *system,
                                                      double r_max )
{
    _r_max = r_max;
    tune( system );
    create_mesh( system );
}

// Tune to a given accuracy
template <class t_System, class t_Neighbor>
void ForceSPME<t_System, t_Neighbor>::tune( t_System *system )
{
    if ( system->domain_x != system->domain_y or
         system->domain_x != system->domain_z )
        throw std::runtime_error( "SPME needs symmetric system size for now." );

    // Fincham 1994, Optimisation of the Ewald Sum for Large Systems
    // only valid for cubic systems (needs adjustement for non-cubic systems)
    auto p = -log( accuracy );
    _k_max = 2 * p / _r_max;
    _alpha = sqrt( p ) / _r_max;
}

// Compute a 1D cubic cardinal B-spline value, used in spreading charge to mesh
//   Given the distance from the particle (x) in units of mesh spaces, this
//   computes the fraction of that charge to place at a mesh point x mesh spaces
//   away The cubic B-spline used here is shifted so that it is symmetric about
//   zero All cubic B-splines are smooth functions that go to zero and are
//   defined piecewise
// TODO: replace use of this with Cajita functions
template <class t_System, class t_Neighbor>
KOKKOS_INLINE_FUNCTION double
ForceSPME<t_System, t_Neighbor>::oneDspline( double x )
{
    if ( x >= 0.0 and x < 1.0 )
    {
        return ( 1.0 / 6.0 ) * x * x * x;
    }
    else if ( x >= 1.0 and x <= 2.0 )
    {
        return -( 1.0 / 2.0 ) * x * x * x + 2.0 * x * x - 2.0 * x +
               ( 2.0 / 3.0 );
    }
    // Using the symmetry here, only need to define function between 0 and 2
    // Beware: This means all input to this function should be made positive
    {
        return 0.0; // Zero if distance is >= 2 mesh spacings
    }
}

// Compute derivative of 1D cubic cardinal B-spline
// TODO: replace use of this with Cajita functions
template <class t_System, class t_Neighbor>
KOKKOS_INLINE_FUNCTION double
ForceSPME<t_System, t_Neighbor>::oneDsplinederiv( double origx )
{
    double x = 2.0 - std::abs( origx );
    double forcedir = 1.0;
    if ( origx < 0.0 )
    {
        forcedir = -1.0;
    }
    if ( x >= 0.0 and x < 1.0 )
    {
        return ( 1.0 / 2.0 ) * x * x * forcedir;
    }
    else if ( x >= 1.0 and x <= 2.0 )
    {
        return ( -( 3.0 / 2.0 ) * x * x + 4.0 * x - 2.0 ) * forcedir;
    }
    // Using the symmetry here, only need to define function between 0 and 2
    // Beware: This means all input to this function should be made positive
    else
    {
        return 0.0; // Zero if distance is >= 2 mesh spacings
    }
}

// Compute a 1-D Euler spline. This function is part of the "lattice structure
// factor" and is given by:
//   b(k, meshwidth) = exp(2*PI*i*3*k/meshwidth) / SUM_{l=0,2}(1Dspline(l+1) *
//   exp(2*PI*i*k*l/meshwidth)) when using a non-shifted cubic B-spline in the
//   charge spread, where meshwidth is the number of mesh points in that
//   dimension and k is the scaled fractional coordinate
template <class t_System, class t_Neighbor>
KOKKOS_INLINE_FUNCTION double
ForceSPME<t_System, t_Neighbor>::oneDeuler( int k, int meshwidth )
{
    double denomreal = 0.0;
    double denomimag = 0.0;
    // Compute the denominator sum first, splitting the complex exponential into
    // sin and cos
    for ( int l = 0; l < 3; l++ )
    {
        denomreal +=
            ForceSPME::oneDspline( fmin( 4.0 - ( l + 1.0 ), l + 1.0 ) ) *
            cos( 2.0 * PI * double( k ) * l / double( meshwidth ) );
        denomimag +=
            ForceSPME::oneDspline( fmin( 4.0 - ( l + 1.0 ), l + 1.0 ) ) *
            sin( 2.0 * PI * double( k ) * l / double( meshwidth ) );
    }
    // Compute the numerator, again splitting the complex exponential
    double numreal = cos( 2.0 * PI * 3.0 * double( k ) / double( meshwidth ) );
    double numimag = sin( 2.0 * PI * 3.0 * double( k ) / double( meshwidth ) );
    // Returning the norm of the 1-D Euler spline
    return ( numreal * numreal + numimag * numimag ) /
           ( denomreal * denomreal + denomimag * denomimag );
}

// Create uniform mesh for SPME method
template <class t_System, class t_Neighbor>
void ForceSPME<t_System, t_Neighbor>::create_mesh( t_System *system )
{
    // TODO: This should be configurable
    double cell_size = system->domain_x / 100.0;
    std::array<bool, 3> is_dim_periodic = {true, true, true};
    std::array<double, 3> global_low_corner = {
        system->domain_lo_x, system->domain_lo_y, system->domain_lo_z};
    std::array<double, 3> global_high_corner = {
        system->domain_hi_x, system->domain_hi_y, system->domain_hi_z};
    // Create the global mesh.
    auto uniform_global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );

    // Compute what proc_grid must be
    const int proc_grid_x = system->domain_x / system->sub_domain_x;
    const int proc_grid_y = system->domain_y / system->sub_domain_y;
    const int proc_grid_z = system->domain_z / system->sub_domain_z;

    const std::array<int, 3> proc_grid = {proc_grid_x, proc_grid_y,
                                          proc_grid_z};

    // Partition mesh into local grids.
    auto partitioner = Cajita::ManualPartitioner( proc_grid );
    auto global_grid = Cajita::createGlobalGrid(
        MPI_COMM_WORLD, uniform_global_mesh, is_dim_periodic, partitioner );

    // Create a local grid.
    const int halo_width = 3;
    local_grid = Cajita::createLocalGrid( global_grid, halo_width );

    // Create a scalar field for charge on the grid.
    // Using uppercase for grid/mesh variables and lowercase for particles
    auto scalar_layout =
        Cajita::createArrayLayout( local_grid, 1, Cajita::Node() );
    Q = Cajita::createArray<double, device_type>( "meshq", scalar_layout );
    Q_halo = Cajita::createHalo( *Q, Cajita::FullHaloPattern() );

    auto owned_space = local_grid->indexSpace( Cajita::Own(), Cajita::Node(),
                                               Cajita::Local() );

    int meshwidth_x = global_grid->globalNumEntity( Cajita::Node(), 0 );
    int meshwidth_y = global_grid->globalNumEntity( Cajita::Node(), 1 );
    int meshwidth_z = global_grid->globalNumEntity( Cajita::Node(), 2 );
    double alpha = _alpha;

    // Calculating the values of the BC array involves first shifting the
    // fractional coords then compute the B and C arrays as described in the
    // paper This can be done once at the start of a run if the mesh stays
    // constant
    BC_array =
        Cajita::createArray<double, device_type>( "BC_array", scalar_layout );
    auto BC_view = BC_array->view();

    auto BC_functor = KOKKOS_LAMBDA( const int kx, const int ky, const int kz )
    {
        int mx, my, mz;
        if ( kx + ky + kz > 0 )
        {
            // Shift the C array
            mx = kx;
            my = ky;
            mz = kz;
            if ( mx > meshwidth_x / 2.0 )
            {
                mx = kx - meshwidth_x;
            }
            if ( my > meshwidth_y / 2.0 )
            {
                my = ky - meshwidth_y;
            }
            if ( mz > meshwidth_z / 2.0 )
            {
                mz = kz - meshwidth_z;
            }
            auto m2 = ( mx * mx + my * my + mz * mz );
            // Calculate BC.
            BC_view( kx, ky, kz, 0 ) =
                ForceSPME::oneDeuler( kx, meshwidth_x ) *
                ForceSPME::oneDeuler( ky, meshwidth_y ) *
                ForceSPME::oneDeuler( kz, meshwidth_z ) *
                exp( -PI * PI * m2 / ( alpha * alpha ) ) /
                ( PI * system->domain_x * system->domain_y * system->domain_z *
                  m2 );
        }
        else
        {
            BC_view( kx, ky, kz, 0 ) = 0.0;
        }
    };
    Kokkos::parallel_for(
        Cajita::createExecutionPolicy( owned_space, exe_space() ), BC_functor );
    Kokkos::fence();
    // TODO: check these indices
}

// Compute the forces
template <class t_System, class t_Neighbor>
void ForceSPME<t_System, t_Neighbor>::compute( t_System *system,
                                               t_Neighbor *neighbor )
{
    // For now, force symmetry
    if ( system->domain_x != system->domain_y or
         system->domain_x != system->domain_z )
        throw std::runtime_error( "SPME needs symmetric system size for now." );

    auto N_local = system->N_local;
    auto alpha = _alpha;

    // Per-atom properties
    system->slice_force();
    auto x = system->x;
    auto f = system->f;
    auto type = system->type;
    system->slice_q();
    auto q = system->q;

    // Neighbor list
    auto neigh_list = neighbor->get();

    /* reciprocal-space contribution */
    auto owned_space = local_grid->indexSpace( Cajita::Own(), Cajita::Node(),
                                               Cajita::Local() );
    auto grid_policy =
        Cajita::createExecutionPolicy( owned_space, exe_space() );

    // First, spread the charges onto the mesh
    Cajita::ArrayOp::assign( *Q, 0.0, Cajita::Ghost() );
    auto scalar_p2g = Cajita::createScalarValueP2G( q, 1.0 );
    Cajita::p2g( scalar_p2g, x, N_local, Cajita::Spline<3>(), *Q_halo, *Q );

    // Copy mesh charge into complex view for FFT work
    auto vector_layout =
        Cajita::createArrayLayout( local_grid, 1, Cajita::Node() );
    Qcomplex = Cajita::createArray<Kokkos::complex<double>, device_type>(
        "Qcomplex", vector_layout );
    auto Qcomplex_view = Qcomplex->view();
    auto Q_view = Q->view();

    auto copy_charge = KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        Qcomplex_view( i, j, k, 0 ).real( Q_view( i, j, k, 0 ) );
        Qcomplex_view( i, j, k, 0 ).imag( 0.0 );
    };
    Kokkos::parallel_for( grid_policy, copy_charge );
    Kokkos::fence();

    // Next, solve Poisson's equation taking some FFTs of charges on mesh grid

    // Using default FastFourierTransformParams
    auto fft =
        Cajita::Experimental::createFastFourierTransform<double, device_type>(
            *vector_layout,
            Cajita::Experimental::FastFourierTransformParams() );

    fft->reverse( *Qcomplex );

    // update Q for later force calcs
    auto BC_view = BC_array->view();
    auto mult_BC_Qr = KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        Qcomplex_view( i, j, k, 0 ) *= BC_view( i, j, k, 0 );
    };
    Kokkos::parallel_for( grid_policy, mult_BC_Qr );
    Kokkos::fence();

    fft->forward( *Qcomplex );

    // Copy real part of complex Qr to Q
    auto copy_back_charge =
        KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        Q_view( i, j, k, 0 ) = Qcomplex_view( i, j, k, 0 ).real();
    };
    Kokkos::parallel_for( grid_policy, copy_back_charge );
    Kokkos::fence();

    // Now, compute forces on each particle
    auto scalar_gradient_g2p = Cajita::createScalarGradientG2P( f, 1.0 );
    Cajita::g2p( *Q, *Q_halo, x, N_local, Cajita::Spline<3>(),
                 scalar_gradient_g2p );

    // Now, multiply each value by particle's charge to get force
    // TODO: This needs to only be applied to the longrange portion
    auto mult_q = KOKKOS_LAMBDA( const int idx )
    {
        f( idx, 0 ) *= q( idx );
        f( idx, 1 ) *= q( idx );
        f( idx, 2 ) *= q( idx );
    };
    Kokkos::RangePolicy<exe_space> policy( 0, N_local );
    Kokkos::parallel_for( policy, mult_q );
    Kokkos::fence();

    /* real-space contribution */
    auto realspace_force = KOKKOS_LAMBDA( const int idx )
    {
        int num_n =
            Cabana::NeighborList<t_neigh_list>::numNeighbor( neigh_list, idx );

        auto rx = x( idx, 0 );
        auto ry = x( idx, 1 );
        auto rz = x( idx, 2 );
        auto qi = q( idx );

        for ( int ij = 0; ij < num_n; ++ij )
        {
            int j = Cabana::NeighborList<t_neigh_list>::getNeighbor( neigh_list,
                                                                     idx, ij );
            auto dx = x( j, 0 ) - rx;
            auto dy = x( j, 1 ) - ry;
            auto dz = x( j, 2 ) - rz;
            auto d = sqrt( dx * dx + dy * dy + dz * dz );
            auto qj = q( j );

            // force computation
            auto f_fact = qi * qj *
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
    Kokkos::parallel_for( policy, realspace_force );
    Kokkos::fence();
}

template <class t_System, class t_Neighbor>
T_V_FLOAT
ForceSPME<t_System, t_Neighbor>::compute_energy( t_System *system,
                                                 t_Neighbor *neighbor )
{
    N_local = system->N_local;
    auto alpha = _alpha;

    // Per-atom properties
    system->slice_x();
    auto x = system->x;
    system->slice_q();
    auto q = system->q;

    // Neighbor list
    auto neigh_list = neighbor->get();

    auto owned_space = local_grid->indexSpace( Cajita::Own(), Cajita::Node(),
                                               Cajita::Local() );
    auto BC_view = BC_array->view();
    auto Qcomplex_view = Qcomplex->view();

    const double ELECTRON_CHARGE = 1.60217662E-19; // electron charge in
                                                   // Coulombs
    const double EPS_0 =
        8.8541878128E-22; // permittivity of free space in Farads/Angstrom
    const double ENERGY_CONVERSION_FACTOR =
        ELECTRON_CHARGE / ( 4.0 * PI * EPS_0 );

    T_V_FLOAT energy_k = 0.0;
    auto kspace_potential =
        KOKKOS_LAMBDA( const int i, const int j, const int k, T_V_FLOAT &PE )
    {
        PE += ENERGY_CONVERSION_FACTOR * BC_view( i, j, k, 0 ) *
              ( ( Qcomplex_view( i, j, k, 0 ).real() *
                  Qcomplex_view( i, j, k, 0 ).real() ) +
                ( Qcomplex_view( i, j, k, 0 ).imag() *
                  Qcomplex_view( i, j, k, 0 ).imag() ) );
    };
    Kokkos::parallel_reduce(
        "ForceSPME::KspacePE",
        Cajita::createExecutionPolicy( owned_space, exe_space() ),
        kspace_potential, energy_k );
    Kokkos::fence();

    T_V_FLOAT energy_r = 0.0;
    auto realspace_potential = KOKKOS_LAMBDA( const int idx, T_V_FLOAT &PE )
    {
        int num_n =
            Cabana::NeighborList<t_neigh_list>::numNeighbor( neigh_list, idx );

        auto rx = x( idx, 0 );
        auto ry = x( idx, 1 );
        auto rz = x( idx, 2 );
        auto qi = q( idx );

        for ( int ij = 0; ij < num_n; ++ij )
        {
            int j = Cabana::NeighborList<t_neigh_list>::getNeighbor( neigh_list,
                                                                     idx, ij );
            auto dx = x( j, 0 ) - rx;
            auto dy = x( j, 1 ) - ry;
            auto dz = x( j, 2 ) - rz;
            auto d = sqrt( dx * dx + dy * dy + dz * dz );
            auto qj = q( j );

            PE += ENERGY_CONVERSION_FACTOR * qi * qj * erfc( alpha * d ) / d;
        }

        // self-energy contribution
        PE += -alpha / PI_SQRT * qi * qi * ENERGY_CONVERSION_FACTOR;
    };
    Kokkos::RangePolicy<exe_space> policy( 0, N_local );
    Kokkos::parallel_reduce( "ForceSPME::RealSpacePE", policy,
                             realspace_potential, energy_r );
    Kokkos::fence();

    T_V_FLOAT energy = energy_k + energy_r;
    return energy;
}

template <class t_System, class t_Neighbor>
const char *ForceSPME<t_System, t_Neighbor>::name()
{
    return "LongRangeForce:SPME";
}
