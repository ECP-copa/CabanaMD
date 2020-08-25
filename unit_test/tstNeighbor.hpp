/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <neighbor.h>
#include <system.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace Test
{
//---------------------------------------------------------------------------//
// Test list implementation.
template <class... Params>
struct TestNeighborList
{
    Kokkos::View<int *, Params...> counts;
    Kokkos::View<int **, Params...> neighbors;
};
template <class KokkosMemorySpace>
TestNeighborList<typename TEST_EXECSPACE::array_layout, Kokkos::HostSpace>
createTestListHostCopy( const TestNeighborList<KokkosMemorySpace> &test_list )
{
    using data_layout = typename decltype( test_list.counts )::array_layout;
    TestNeighborList<data_layout, Kokkos::HostSpace> list_host;
    Kokkos::resize( list_host.counts, test_list.counts.extent( 0 ) );
    Kokkos::deep_copy( list_host.counts, test_list.counts );
    Kokkos::resize( list_host.neighbors, test_list.neighbors.extent( 0 ),
                    test_list.neighbors.extent( 1 ) );
    Kokkos::deep_copy( list_host.neighbors, test_list.neighbors );
    return list_host;
}

//---------------------------------------------------------------------------//
// Copy into a host test list, extracted with the neighbor list interface.
template <class ListType>
TestNeighborList<typename TEST_EXECSPACE::array_layout, Kokkos::HostSpace>
copyListToHost( const ListType &list, const int total_atoms, const int max_n )
{
    TestNeighborList<TEST_MEMSPACE> list_host;
    list_host.counts =
        Kokkos::View<int *, TEST_MEMSPACE>( "counts", total_atoms );
    list_host.neighbors =
        Kokkos::View<int **, TEST_MEMSPACE>( "neighbors", total_atoms, max_n );
    Kokkos::parallel_for(
        "copy list", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, total_atoms ),
        KOKKOS_LAMBDA( const int p ) {
            list_host.counts( p ) =
                Cabana::NeighborList<ListType>::numNeighbor( list, p );
            for ( int n = 0; n < list_host.counts( p ); ++n )
                list_host.neighbors( p, n ) =
                    Cabana::NeighborList<ListType>::getNeighbor( list, p, n );
        } );
    Kokkos::fence();
    return createTestListHostCopy( list_host );
}

//-------------------------------------------------------------------------//
// Build a neighbor list with a brute force n^2 implementation.
template <class PositionSlice>
TestNeighborList<TEST_MEMSPACE>
computeFullNeighborList( const PositionSlice &position, const double cutoff )
{
    // Count first.
    TestNeighborList<TEST_MEMSPACE> list;
    int total_atoms = position.size();
    double rsqr = cutoff * cutoff;
    list.counts = Kokkos::View<int *, TEST_MEMSPACE>( "test_neighbor_count",
                                                      total_atoms );
    Kokkos::deep_copy( list.counts, 0 );
    auto count_op = KOKKOS_LAMBDA( const int i )
    {
        for ( int j = 0; j < total_atoms; ++j )
        {
            if ( i != j )
            {
                double dsqr = 0.0;
                for ( int d = 0; d < 3; ++d )
                    dsqr += ( position( i, d ) - position( j, d ) ) *
                            ( position( i, d ) - position( j, d ) );
                if ( dsqr <= rsqr )
                    list.counts( i ) += 1;
            }
        }
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> exec_policy( 0, total_atoms );
    Kokkos::parallel_for( exec_policy, count_op );
    Kokkos::fence();

    // Allocate.
    auto max_op = KOKKOS_LAMBDA( const int i, int &max_val )
    {
        if ( max_val < list.counts( i ) )
            max_val = list.counts( i );
    };
    int max_n;
    Kokkos::parallel_reduce( exec_policy, max_op, Kokkos::Max<int>( max_n ) );
    Kokkos::fence();
    list.neighbors = Kokkos::View<int **, TEST_MEMSPACE>( "test_neighbors",
                                                          total_atoms, max_n );

    // Fill.
    auto fill_op = KOKKOS_LAMBDA( const int i )
    {
        int n_count = 0;
        for ( int j = 0; j < total_atoms; ++j )
        {
            if ( i != j )
            {
                double dsqr = 0.0;
                for ( int d = 0; d < 3; ++d )
                    dsqr += ( position( i, d ) - position( j, d ) ) *
                            ( position( i, d ) - position( j, d ) );
                if ( dsqr <= rsqr )
                {
                    list.neighbors( i, n_count ) = j;
                    ++n_count;
                }
            }
        }
    };
    Kokkos::parallel_for( exec_policy, fill_op );
    Kokkos::fence();
    return list;
}

//---------------------------------------------------------------------------//
template <class ListType, class PositionSlice>
void checkFullNeighborListPartialRange( const ListType &list,
                                        const PositionSlice &position,
                                        const double cutoff,
                                        const int local_atoms )
{
    // Build a full list to test with.
    auto N2_list = computeFullNeighborList( position, cutoff );
    auto N2_list_host = createTestListHostCopy( N2_list );

    // Copy to a consistent list format on the host.
    auto list_host = copyListToHost( list, N2_list.neighbors.extent( 0 ),
                                     N2_list.neighbors.extent( 1 ) );

    // Check the results.
    int total_atoms = position.size();
    for ( int p = 0; p < total_atoms; ++p )
    {
        if ( p < local_atoms )
        {
            // First check that the number of neighbors are the same.
            EXPECT_EQ( list_host.counts( p ), N2_list_host.counts( p ) );

            // Now extract the neighbors.
            std::vector<int> computed_neighbors( N2_list_host.counts( p ) );
            std::vector<int> actual_neighbors( N2_list_host.counts( p ) );
            for ( int n = 0; n < N2_list_host.counts( p ); ++n )
            {
                computed_neighbors[n] = list_host.neighbors( p, n );
                actual_neighbors[n] = N2_list_host.neighbors( p, n );
            }

            // Sort them because we have no guarantee of order.
            std::sort( computed_neighbors.begin(), computed_neighbors.end() );
            std::sort( actual_neighbors.begin(), actual_neighbors.end() );

            // Now compare directly.
            for ( int n = 0; n < N2_list_host.counts( p ); ++n )
                EXPECT_EQ( computed_neighbors[n], actual_neighbors[n] );
        }
        else
        {
            // Ghost atoms should have no neighbors
            EXPECT_EQ( list_host.counts( p ), 0 );
        }
    }
}

//---------------------------------------------------------------------------//
template <class ListType, class PositionSlice>
void checkHalfNeighborListPartialRange( const ListType &list,
                                        const PositionSlice &position,
                                        const double cutoff,
                                        const int local_atoms )
{
    // First build a full list.
    auto N2_list = computeFullNeighborList( position, cutoff );
    auto N2_list_host = createTestListHostCopy( N2_list );

    // Copy to a consistent list format on the host.
    auto list_host = copyListToHost( list, N2_list.neighbors.extent( 0 ),
                                     N2_list.neighbors.extent( 1 ) );

    // Check that the full list is nearly twice the size of the half list.
    int total_atoms = position.size();
    int half_size = 0;
    int full_size = 0;
    for ( int p = 0; p < total_atoms; ++p )
    {
        half_size += list_host.counts( p );
        full_size += N2_list_host.counts( p );
    }
    // Ghost atoms mean this won't be exactly 2x.
    EXPECT_LE( half_size, full_size );

    for ( int p = 0; p < total_atoms; ++p )
    {
        if ( p < local_atoms )
        {
            // Atoms should have equal or fewer neighbors in half list.
            EXPECT_LE( list_host.counts( p ), N2_list_host.counts( p ) );

            // Check that there are no self neighbors.
            for ( int n = 0; n < list_host.counts( p ); ++n )
            {
                auto p_n = list_host.neighbors( p, n );
                for ( int m = 0; m < list_host.counts( p_n ); ++m )
                {
                    auto n_m = list_host.neighbors( p_n, m );
                    EXPECT_NE( n_m, p );
                }
            }
        }
        else
        {
            // Ghost atoms should have no neighbors
            EXPECT_EQ( list_host.counts( p ), 0 );
        }
    }
}

//---------------------------------------------------------------------------//
// Create atoms.
template <class t_System>
t_System createAtoms( const int num_atom, const int num_ghost,
                      const double box_min, const double box_max )
{
    t_System system;
    system.init();

    // Manually setup what would be done in input
    system.dt = 0.005;
    system.mvv2e = 1.0;

    // Set mass (device View)
    using h_t_mass = typename t_System::h_t_mass;
    h_t_mass h_mass = Kokkos::create_mirror_view( system.mass );
    h_mass( 0 ) = 1.0;
    Kokkos::deep_copy( system.mass, h_mass );

    system.resize( num_atom );
    system.N_local = num_atom - num_ghost;
    system.N_ghost = num_ghost;

    auto box = box_max - box_min;
    system.domain_x = system.domain_y = system.domain_z = box;
    system.domain_lo_x = system.domain_lo_y = system.domain_lo_z = box_min;
    system.domain_hi_x = system.domain_hi_y = system.domain_hi_z = box_max;
    system.sub_domain_lo_x = system.sub_domain_lo_y = system.sub_domain_lo_z =
        box_min;
    system.sub_domain_hi_x = system.sub_domain_hi_y = system.sub_domain_hi_z =
        box_max;

    // Create random atom positions within the box.
    system.slice_x();
    auto position = system.x;
    using PoolType = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    using RandomType = Kokkos::Random_XorShift64<TEST_EXECSPACE>;
    PoolType pool( 342343901 );
    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        auto gen = pool.get_state();
        for ( int d = 0; d < 3; ++d )
            position( p, d ) =
                Kokkos::rand<RandomType, double>::draw( gen, box_min, box_max );
        pool.free_state( gen );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> exec_policy( 0, num_atom );
    Kokkos::parallel_for( exec_policy, random_coord_op );
    Kokkos::fence();

    return system;
}

//---------------------------------------------------------------------------//
template <class t_System, class t_Neighbor>
void testNeighborListPartialRange( bool half_neigh )
{
    // Create the AoSoA and fill with random atom positions.
    int num_atom = 1e3;
    int num_ghost = 200;
    int num_local = num_atom - num_ghost;
    double cutoff = 2.32;
    double box_min = -5.3 * cutoff;
    double box_max = 4.7 * cutoff;

    t_System system =
        createAtoms<t_System>( num_atom, num_ghost, box_min, box_max );

    // Create the neighbor list.
    t_Neighbor neighbor( cutoff, half_neigh );
    neighbor.create( &system );

    // Check the neighbor list.
    system.slice_x();
    auto position = system.x;
    auto neigh_list = neighbor.get();

    if ( half_neigh )
        checkHalfNeighborListPartialRange( neigh_list, position, cutoff,
                                           num_local );
    else
        checkFullNeighborListPartialRange( neigh_list, position, cutoff,
                                           num_local );
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, verlet_full_test )
{
    using DeviceType = Kokkos::Device<TEST_EXECSPACE, TEST_MEMSPACE>;
    using t_System = System<DeviceType, AoSoA6>;
    {
        using t_Neigh = NeighborVerlet<t_System, Cabana::FullNeighborTag,
                                       Cabana::VerletLayout2D>;
        testNeighborListPartialRange<t_System, t_Neigh>( false );
    }
    {
        using t_Neigh = NeighborVerlet<t_System, Cabana::FullNeighborTag,
                                       Cabana::VerletLayoutCSR>;
        testNeighborListPartialRange<t_System, t_Neigh>( false );
    }
    {
#ifdef Cabana_ENABLE_ARBORX
        using t_Neigh = NeighborTree<t_System, Cabana::FullNeighborTag,
                                     Cabana::VerletLayoutCSR>;
        testNeighborListPartialRange<t_System, t_Neigh>( false );
#endif
    }
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, verlet_half_test )
{
    using DeviceType = Kokkos::Device<TEST_EXECSPACE, TEST_MEMSPACE>;
    using t_System = System<DeviceType, AoSoA6>;
    {
        using t_Neigh = NeighborVerlet<t_System, Cabana::HalfNeighborTag,
                                       Cabana::VerletLayout2D>;
        testNeighborListPartialRange<t_System, t_Neigh>( true );
    }
    {
        using t_Neigh = NeighborVerlet<t_System, Cabana::HalfNeighborTag,
                                       Cabana::VerletLayoutCSR>;
        testNeighborListPartialRange<t_System, t_Neigh>( true );
    }
    {
#ifdef Cabana_ENABLE_ARBORX
        using t_Neigh = NeighborTree<t_System, Cabana::HalfNeighborTag,
                                     Cabana::VerletLayoutCSR>;
        testNeighborListPartialRange<t_System, t_Neigh>( true );
#endif
    }
}

//---------------------------------------------------------------------------//

} // end namespace Test
