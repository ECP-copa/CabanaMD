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

#include <integrator_nve.h>
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
// Create particles.
template <class t_System>
t_System createParticles( const int num_particle, const int num_ghost,
                          const double box_min, const double box_max )
{
    t_System system;
    system.init();

    // Manually setup what would be done in input
    system.dt = 0.005;
    system.mvv2e = 1.0;
    system.mass( 0 ) = 1.0;

    system.resize( num_particle );
    system.N_local = num_particle - num_ghost;
    system.N_ghost = num_ghost;

    auto box = box_max - box_min;
    system.domain_x = system.domain_y = system.domain_z = box;
    system.domain_lo_x = system.domain_lo_y = system.domain_lo_z = box_min;
    system.domain_hi_x = system.domain_hi_y = system.domain_hi_z = box_max;

    system.slice_integrate();
    auto x = system.x;
    auto v = system.v;
    auto f = system.f;
    auto type = system.type;

    using PoolType = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    using RandomType = Kokkos::Random_XorShift64<TEST_EXECSPACE>;
    PoolType pool( 342343901 );
    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        auto gen = pool.get_state();
        for ( int d = 0; d < 3; ++d )
        {
            x( p, d ) =
                Kokkos::rand<RandomType, double>::draw( gen, box_min, box_max );
            v( p, d ) =
                Kokkos::rand<RandomType, double>::draw( gen, -1.0, 1.0 );
            f( p, d ) =
                Kokkos::rand<RandomType, double>::draw( gen, -1.0, 1.0 );
            type( p ) = 0;
        }
        pool.free_state( gen );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> exec_policy( 0, num_particle );
    Kokkos::parallel_for( exec_policy, random_coord_op );
    Kokkos::fence();

    return system;
}

//---------------------------------------------------------------------------//
template <class t_System>
void testIntegratorReversibility( int steps )
{
    // Create the AoSoA and fill with random particle positions
    int num_particle = 1e3;
    int num_ghost = 200;
    double test_radius = 2.32;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;

    t_System system =
        createParticles<t_System>( num_particle, num_ghost, box_min, box_max );
    Integrator<t_System> integrator( &system );

    // Keep a copy of initial positions on the host
    using DataTypes = Cabana::MemberTypes<double[3]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    AoSoA_t x_aosoa_init( "x_init_host", num_particle );
    auto x_init = Cabana::slice<0>( x_aosoa_init );
    system.slice_x();
    Cabana::deep_copy( x_init, system.x );

    // Integrate one step
    for ( int s = 0; s < steps; ++s )
    {
        integrator.initial_integrate( &system );
        integrator.final_integrate( &system );
    }

    // Reverse system
    system.slice_v();
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
            system.v( p, d ) *= -1.0;

    // Integrate back
    for ( int s = 0; s < steps; ++s )
    {
        integrator.initial_integrate( &system );
        integrator.final_integrate( &system );
    }

    // Make a copy of final results on the host
    AoSoA_t x_aosoa_final( "x_final_host", num_particle );
    auto x_final = Cabana::slice<0>( x_aosoa_final );
    Cabana::deep_copy( x_final, system.x );

    // Check the results
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
            EXPECT_FLOAT_EQ( x_final( p, d ), x_init( p, d ) );
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, reversibility_test )
{
    testIntegratorReversibility<System<AoSoA6>>( 100 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
