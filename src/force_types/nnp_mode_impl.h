// This file will soon be removed.
// See https://github.com/ECP-copa/CabanaMD/issues/37

// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include <nnp_mode.h>

#include <utility.h>

#include <algorithm> // std::min, std::max
#include <cstdlib>   // atoi, atof
#include <fstream>   // std::ifstream
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <string>

using namespace std;
using namespace nnpCbn;

template <class t_slice_x, class t_slice_type, class t_slice_G,
          class t_neigh_list, class t_neigh_parallel, class t_angle_parallel>
void Mode::calculateSymmetryFunctionGroups( t_slice_x x, t_slice_type type,
                                            t_slice_G G,
                                            t_neigh_list neigh_list,
                                            int N_local )
{
    Cabana::deep_copy( G, 0.0 );

    Kokkos::RangePolicy<ExecutionSpace> policy( 0, N_local );
    t_neigh_parallel neigh_op_tag;
    t_angle_parallel angle_op_tag;

    auto calc_radial_symm_op = KOKKOS_LAMBDA( const int i, const int j )
    {
        double pfcij = 0.0;
        double pdfcij = 0.0;
        double eta, rs;
        size_t nej;
        int memberindex, globalIndex;
        double rij, r2ij;
        T_F_FLOAT dxij, dyij, dzij;

        int attype = type( i );
        for ( int groupIndex = 0; groupIndex < numSFGperElem( attype );
              ++groupIndex )
        {
            if ( d_SF( attype, d_SFGmemberlist( attype, groupIndex, 0 ), 1 ) ==
                 2 )
            {
                size_t memberindex0 = d_SFGmemberlist( attype, groupIndex, 0 );
                size_t e1 = d_SF( attype, memberindex0, 2 );
                double rc = d_SF( attype, memberindex0, 7 );
                size_t size =
                    d_SFGmemberlist( attype, groupIndex, maxSFperElem );

                nej = type( j );
                dxij = ( x( i, 0 ) - x( j, 0 ) ) * CFLENGTH * convLength;
                dyij = ( x( i, 1 ) - x( j, 1 ) ) * CFLENGTH * convLength;
                dzij = ( x( i, 2 ) - x( j, 2 ) ) * CFLENGTH * convLength;
                r2ij = dxij * dxij + dyij * dyij + dzij * dzij;
                rij = sqrt( r2ij );
                if ( e1 == nej && rij < rc )
                {
                    compute_cutoff( cutoffType, pfcij, pdfcij, rij, rc, false );
                    for ( size_t k = 0; k < size; ++k )
                    {
                        memberindex = d_SFGmemberlist( attype, groupIndex, k );
                        globalIndex = d_SF( attype, memberindex, 14 );
                        eta = d_SF( attype, memberindex, 4 );
                        rs = d_SF( attype, memberindex, 8 );
                        G( i, globalIndex ) +=
                            exp( -eta * ( rij - rs ) * ( rij - rs ) ) * pfcij;
                    }
                }
            }
        }
    };
    Cabana::neighbor_parallel_for(
        policy, calc_radial_symm_op, neigh_list, Cabana::FirstNeighborsTag(),
        neigh_op_tag, "Mode::calculateRadialSymmetryFunctionGroups" );
    Kokkos::fence();

    auto calc_angular_symm_op =
        KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        double pfcij = 0.0;
        double pdfcij = 0.0;
        double pfcik, pdfcik, pfcjk, pdfcjk;
        size_t nej, nek;
        int memberindex, globalIndex;
        double rij, r2ij, rik, r2ik, rjk, r2jk;
        T_F_FLOAT dxij, dyij, dzij, dxik, dyik, dzik, dxjk, dyjk, dzjk;
        double eta, rs, lambda, zeta;

        int attype = type( i );
        for ( int groupIndex = 0; groupIndex < numSFGperElem( attype );
              ++groupIndex )
        {
            if ( d_SF( attype, d_SFGmemberlist( attype, groupIndex, 0 ), 1 ) ==
                 3 )
            {
                size_t memberindex0 = d_SFGmemberlist( attype, groupIndex, 0 );
                size_t e1 = d_SF( attype, memberindex0, 2 );
                size_t e2 = d_SF( attype, memberindex0, 3 );
                double rc = d_SF( attype, memberindex0, 7 );
                size_t size =
                    d_SFGmemberlist( attype, groupIndex, maxSFperElem );

                nej = type( j );
                dxij = ( x( i, 0 ) - x( j, 0 ) ) * CFLENGTH * convLength;
                dyij = ( x( i, 1 ) - x( j, 1 ) ) * CFLENGTH * convLength;
                dzij = ( x( i, 2 ) - x( j, 2 ) ) * CFLENGTH * convLength;
                r2ij = dxij * dxij + dyij * dyij + dzij * dzij;
                rij = sqrt( r2ij );
                if ( ( e1 == nej || e2 == nej ) && rij < rc )
                {
                    // Calculate cutoff function and derivative.
                    compute_cutoff( cutoffType, pfcij, pdfcij, rij, rc, false );

                    nek = type( k );

                    if ( ( e1 == nej && e2 == nek ) ||
                         ( e2 == nej && e1 == nek ) )
                    {
                        dxik =
                            ( x( i, 0 ) - x( k, 0 ) ) * CFLENGTH * convLength;
                        dyik =
                            ( x( i, 1 ) - x( k, 1 ) ) * CFLENGTH * convLength;
                        dzik =
                            ( x( i, 2 ) - x( k, 2 ) ) * CFLENGTH * convLength;
                        r2ik = dxik * dxik + dyik * dyik + dzik * dzik;
                        rik = sqrt( r2ik );
                        if ( rik < rc )
                        {
                            dxjk = dxik - dxij;
                            dyjk = dyik - dyij;
                            dzjk = dzik - dzij;
                            r2jk = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
                            if ( r2jk < rc * rc )
                            {
                                // Energy calculation.
                                compute_cutoff( cutoffType, pfcik, pdfcik, rik,
                                                rc, false );

                                rjk = sqrt( r2jk );
                                compute_cutoff( cutoffType, pfcjk, pdfcjk, rjk,
                                                rc, false );

                                double const rinvijik = 1.0 / rij / rik;
                                double const costijk =
                                    ( dxij * dxik + dyij * dyik +
                                      dzij * dzik ) *
                                    rinvijik;
                                double vexp = 0.0, rijs = 0.0, riks = 0.0,
                                       rjks = 0.0;
                                for ( size_t l = 0; l < size; ++l )
                                {
                                    globalIndex =
                                        d_SF( attype,
                                              d_SFGmemberlist( attype,
                                                               groupIndex, l ),
                                              14 );
                                    memberindex = d_SFGmemberlist(
                                        attype, groupIndex, l );
                                    eta = d_SF( attype, memberindex, 4 );
                                    lambda = d_SF( attype, memberindex, 5 );
                                    zeta = d_SF( attype, memberindex, 6 );
                                    rs = d_SF( attype, memberindex, 8 );
                                    if ( rs > 0.0 )
                                    {
                                        rijs = rij - rs;
                                        riks = rik - rs;
                                        rjks = rjk - rs;
                                        vexp = exp( -eta * ( rijs * rijs +
                                                             riks * riks +
                                                             rjks * rjks ) );
                                    }
                                    else
                                        vexp = exp( -eta *
                                                    ( r2ij + r2ik + r2jk ) );
                                    double const plambda =
                                        1.0 + lambda * costijk;
                                    double fg = vexp;
                                    if ( plambda <= 0.0 )
                                        fg = 0.0;
                                    else
                                        fg *= pow( plambda, ( zeta - 1.0 ) );
                                    G( i, globalIndex ) +=
                                        fg * plambda * pfcij * pfcik * pfcjk;
                                } // l
                            }     // rjk <= rc
                        }         // rik <= rc
                    }             // elem
                }                 // rij <= rc
            }
        }
    };
    Cabana::neighbor_parallel_for(
        policy, calc_angular_symm_op, neigh_list, Cabana::SecondNeighborsTag(),
        angle_op_tag, "Mode::calculateAngularSymmetryFunctionGroups" );
    Kokkos::fence();

    auto scale_symm_op = KOKKOS_LAMBDA( const int i )
    {
        int attype = type( i );

        int memberindex0;
        int memberindex, globalIndex;
        double raw_value = 0.0;
        for ( int groupIndex = 0; groupIndex < numSFGperElem( attype );
              ++groupIndex )
        {
            memberindex0 = d_SFGmemberlist( attype, groupIndex, 0 );

            size_t size = d_SFGmemberlist( attype, groupIndex, maxSFperElem );
            for ( size_t k = 0; k < size; ++k )
            {
                globalIndex = d_SF(
                    attype, d_SFGmemberlist( attype, groupIndex, k ), 14 );
                memberindex = d_SFGmemberlist( attype, groupIndex, k );

                if ( d_SF( attype, memberindex0, 1 ) == 2 )
                    raw_value = G( i, globalIndex );
                else if ( d_SF( attype, memberindex0, 1 ) == 3 )
                    raw_value =
                        G( i, globalIndex ) *
                        pow( 2, ( 1 - d_SF( attype, memberindex, 6 ) ) );

                G( i, globalIndex ) =
                    scale( attype, raw_value, memberindex, d_SFscaling );
            }
        }
    };
    Kokkos::parallel_for( "Mode::scaleSymmetryFunctionGroups", policy,
                          scale_symm_op );
    Kokkos::fence();
}

template <class t_slice_type, class t_slice_G, class t_slice_dEdG,
          class t_slice_E>
void Mode::calculateAtomicNeuralNetworks( t_slice_type type, t_slice_G G,
                                          t_slice_dEdG dEdG, t_slice_E E,
                                          int N_local )
{
    NN = d_t_NN( "Mode::NN", N_local, numLayers, maxNeurons );
    dfdx = d_t_NN( "Mode::dfdx", N_local, numLayers, maxNeurons );
    inner = d_t_NN( "Mode::inner", N_local, numHiddenLayers, maxNeurons );
    outer = d_t_NN( "Mode::outer", N_local, numHiddenLayers, maxNeurons );

    auto calc_nn_op = KOKKOS_LAMBDA( const int atomindex )
    {
        int attype = type( atomindex );
        // set input layer of NN
        int layer_0, layer_lminusone;
        layer_0 = (int)numSFperElem( attype );

        for ( int k = 0; k < layer_0; ++k )
            NN( atomindex, 0, k ) = G( atomindex, k );
        // forward propagation
        for ( int l = 1; l < numLayers; l++ )
        {
            if ( l == 1 )
                layer_lminusone = layer_0;
            else
                layer_lminusone = numNeuronsPerLayer[l - 1];
            double dtmp;
            for ( int i = 0; i < numNeuronsPerLayer[l]; i++ )
            {
                dtmp = 0.0;
                for ( int j = 0; j < layer_lminusone; j++ )
                    dtmp += weights( attype, l - 1, i, j ) *
                            NN( atomindex, l - 1, j );
                dtmp += bias( attype, l - 1, i );
                if ( AF( l ) == 0 )
                {
                    NN( atomindex, l, i ) = dtmp;
                    dfdx( atomindex, l, i ) = 1.0;
                }
                else if ( AF( l ) == 1 )
                {
                    dtmp = tanh( dtmp );
                    NN( atomindex, l, i ) = dtmp;
                    dfdx( atomindex, l, i ) = 1.0 - dtmp * dtmp;
                }
            }
        }

        E( atomindex ) = NN( atomindex, numLayers - 1, 0 );

        // derivative of network w.r.t NN inputs
        for ( int k = 0; k < numNeuronsPerLayer[0]; k++ )
        {
            for ( int i = 0; i < numNeuronsPerLayer[1]; i++ )
                inner( atomindex, 0, i ) =
                    weights( attype, 0, i, k ) * dfdx( atomindex, 1, i );

            for ( int l = 1; l < numHiddenLayers + 1; l++ )
            {
                for ( int i2 = 0; i2 < numNeuronsPerLayer( l + 1 ); i2++ )
                {
                    outer( atomindex, l - 1, i2 ) = 0.0;

                    for ( int i1 = 0; i1 < numNeuronsPerLayer( l ); i1++ )
                        outer( atomindex, l - 1, i2 ) +=
                            weights( attype, l, i2, i1 ) *
                            inner( atomindex, l - 1, i1 );
                    outer( atomindex, l - 1, i2 ) *=
                        dfdx( atomindex, l + 1, i2 );

                    if ( l < numHiddenLayers )
                        inner( atomindex, l, i2 ) =
                            outer( atomindex, l - 1, i2 );
                }
            }
            dEdG( atomindex, k ) = outer( atomindex, numHiddenLayers - 1, 0 );
        }
    };
    Kokkos::parallel_for( "Mode::calculateAtomicNeuralNetworks", N_local,
                          calc_nn_op );
    Kokkos::fence();
}

template <class t_slice_x, class t_slice_f, class t_slice_type,
          class t_slice_dEdG, class t_neigh_list, class t_neigh_parallel,
          class t_angle_parallel>
void Mode::calculateForces( t_slice_x x, t_slice_f f_a, t_slice_type type,
                            t_slice_dEdG dEdG, t_neigh_list neigh_list,
                            int N_local )
{
    double convForce = convLength / convEnergy;

    Kokkos::RangePolicy<ExecutionSpace> policy( 0, N_local );
    t_neigh_parallel neigh_op_tag;
    t_angle_parallel angle_op_tag;

    auto calc_radial_force_op = KOKKOS_LAMBDA( const int i, const int j )
    {
        double pfcij = 0.0;
        double pdfcij = 0.0;
        double rij, r2ij;
        T_F_FLOAT dxij, dyij, dzij;
        double eta, rs;
        int memberindex, globalIndex;

        int attype = type( i );

        for ( int groupIndex = 0; groupIndex < numSFGperElem( attype );
              ++groupIndex )
        {
            if ( d_SF( attype, d_SFGmemberlist( attype, groupIndex, 0 ), 1 ) ==
                 2 )
            {
                size_t memberindex0 = d_SFGmemberlist( attype, groupIndex, 0 );
                size_t e1 = d_SF( attype, memberindex0, 2 );
                double rc = d_SF( attype, memberindex0, 7 );
                size_t size =
                    d_SFGmemberlist( attype, groupIndex, maxSFperElem );

                size_t nej = type( j );
                dxij = ( x( i, 0 ) - x( j, 0 ) ) * CFLENGTH * convLength;
                dyij = ( x( i, 1 ) - x( j, 1 ) ) * CFLENGTH * convLength;
                dzij = ( x( i, 2 ) - x( j, 2 ) ) * CFLENGTH * convLength;
                r2ij = dxij * dxij + dyij * dyij + dzij * dzij;
                rij = sqrt( r2ij );
                if ( e1 == nej && rij < rc )
                {
                    // Energy calculation.
                    // Calculate cutoff function and derivative.
                    compute_cutoff( cutoffType, pfcij, pdfcij, rij, rc, true );
                    for ( size_t k = 0; k < size; ++k )
                    {
                        globalIndex = d_SF(
                            attype, d_SFGmemberlist( attype, groupIndex, k ),
                            14 );
                        memberindex = d_SFGmemberlist( attype, groupIndex, k );
                        eta = d_SF( attype, memberindex, 4 );
                        rs = d_SF( attype, memberindex, 8 );
                        double pexp = exp( -eta * ( rij - rs ) * ( rij - rs ) );
                        // Force calculation.
                        double const p1 =
                            d_SFscaling( attype, memberindex, 6 ) *
                            ( pdfcij - 2.0 * eta * ( rij - rs ) * pfcij ) *
                            pexp / rij;
                        f_a( i, 0 ) -= ( dEdG( i, globalIndex ) *
                                         ( p1 * dxij ) * CFFORCE * convForce );
                        f_a( i, 1 ) -= ( dEdG( i, globalIndex ) *
                                         ( p1 * dyij ) * CFFORCE * convForce );
                        f_a( i, 2 ) -= ( dEdG( i, globalIndex ) *
                                         ( p1 * dzij ) * CFFORCE * convForce );

                        f_a( j, 0 ) += ( dEdG( i, globalIndex ) *
                                         ( p1 * dxij ) * CFFORCE * convForce );
                        f_a( j, 1 ) += ( dEdG( i, globalIndex ) *
                                         ( p1 * dyij ) * CFFORCE * convForce );
                        f_a( j, 2 ) += ( dEdG( i, globalIndex ) *
                                         ( p1 * dzij ) * CFFORCE * convForce );
                    }
                }
            }
        }
    };
    Cabana::neighbor_parallel_for( policy, calc_radial_force_op, neigh_list,
                                   Cabana::FirstNeighborsTag(), neigh_op_tag,
                                   "Mode::calculateRadialForces" );
    Kokkos::fence();

    auto calc_angular_force_op =
        KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        double pfcij = 0.0;
        double pdfcij = 0.0;
        double pfcik, pdfcik, pfcjk, pdfcjk;
        size_t nej, nek;
        double rij, r2ij, rik, r2ik, rjk, r2jk;
        T_F_FLOAT dxij, dyij, dzij, dxik, dyik, dzik, dxjk, dyjk, dzjk;
        double eta, rs, lambda, zeta;
        int memberindex, globalIndex;

        int attype = type( i );
        for ( int groupIndex = 0; groupIndex < numSFGperElem( attype );
              ++groupIndex )
        {
            if ( d_SF( attype, d_SFGmemberlist( attype, groupIndex, 0 ), 1 ) ==
                 3 )
            {
                size_t memberindex0 = d_SFGmemberlist( attype, groupIndex, 0 );
                size_t e1 = d_SF( attype, memberindex0, 2 );
                size_t e2 = d_SF( attype, memberindex0, 3 );
                double rc = d_SF( attype, memberindex0, 7 );
                size_t size =
                    d_SFGmemberlist( attype, groupIndex, maxSFperElem );

                nej = type( j );
                dxij = ( x( i, 0 ) - x( j, 0 ) ) * CFLENGTH * convLength;
                dyij = ( x( i, 1 ) - x( j, 1 ) ) * CFLENGTH * convLength;
                dzij = ( x( i, 2 ) - x( j, 2 ) ) * CFLENGTH * convLength;
                r2ij = dxij * dxij + dyij * dyij + dzij * dzij;
                rij = sqrt( r2ij );
                if ( ( e1 == nej || e2 == nej ) && rij < rc )
                {
                    // Calculate cutoff function and derivative.
                    compute_cutoff( cutoffType, pfcij, pdfcij, rij, rc, true );

                    nek = type( k );
                    if ( ( e1 == nej && e2 == nek ) ||
                         ( e2 == nej && e1 == nek ) )
                    {
                        dxik =
                            ( x( i, 0 ) - x( k, 0 ) ) * CFLENGTH * convLength;
                        dyik =
                            ( x( i, 1 ) - x( k, 1 ) ) * CFLENGTH * convLength;
                        dzik =
                            ( x( i, 2 ) - x( k, 2 ) ) * CFLENGTH * convLength;
                        r2ik = dxik * dxik + dyik * dyik + dzik * dzik;
                        rik = sqrt( r2ik );
                        if ( rik < rc )
                        {
                            dxjk = dxik - dxij;
                            dyjk = dyik - dyij;
                            dzjk = dzik - dzij;
                            r2jk = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
                            if ( r2jk < rc * rc )
                            {
                                // Energy calculation.
                                compute_cutoff( cutoffType, pfcik, pdfcik, rik,
                                                rc, true );
                                rjk = sqrt( r2jk );

                                compute_cutoff( cutoffType, pfcjk, pdfcjk, rjk,
                                                rc, true );

                                double const rinvijik = 1.0 / rij / rik;
                                double const costijk =
                                    ( dxij * dxik + dyij * dyik +
                                      dzij * dzik ) *
                                    rinvijik;
                                double const pfc = pfcij * pfcik * pfcjk;
                                double const r2sum = r2ij + r2ik + r2jk;
                                double const pr1 = pfcik * pfcjk * pdfcij / rij;
                                double const pr2 = pfcij * pfcjk * pdfcik / rik;
                                double const pr3 = pfcij * pfcik * pdfcjk / rjk;
                                double vexp = 0.0, rijs = 0.0, riks = 0.0,
                                       rjks = 0.0;
                                for ( size_t l = 0; l < size; ++l )
                                {
                                    globalIndex =
                                        d_SF( attype,
                                              d_SFGmemberlist( attype,
                                                               groupIndex, l ),
                                              14 );
                                    memberindex = d_SFGmemberlist(
                                        attype, groupIndex, l );
                                    rs = d_SF( attype, memberindex, 8 );
                                    eta = d_SF( attype, memberindex, 4 );
                                    lambda = d_SF( attype, memberindex, 5 );
                                    zeta = d_SF( attype, memberindex, 6 );
                                    if ( rs > 0.0 )
                                    {
                                        rijs = rij - rs;
                                        riks = rik - rs;
                                        rjks = rjk - rs;
                                        vexp = exp( -eta * ( rijs * rijs +
                                                             riks * riks +
                                                             rjks * rjks ) );
                                    }
                                    else
                                        vexp = exp( -eta * r2sum );

                                    double const plambda =
                                        1.0 + lambda * costijk;
                                    double fg = vexp;
                                    if ( plambda <= 0.0 )
                                        fg = 0.0;
                                    else
                                        fg *= pow( plambda, ( zeta - 1.0 ) );

                                    fg *= pow( 2, ( 1 - zeta ) ) *
                                          d_SFscaling( attype, memberindex, 6 );
                                    double const pfczl = pfc * zeta * lambda;
                                    double factorDeriv =
                                        2.0 * eta / zeta / lambda;
                                    double const p2etapl =
                                        plambda * factorDeriv;
                                    double p1, p2, p3;
                                    if ( rs > 0.0 )
                                    {
                                        p1 = fg *
                                             ( pfczl *
                                                   ( rinvijik - costijk / r2ij -
                                                     p2etapl * rijs / rij ) +
                                               pr1 * plambda );
                                        p2 = fg *
                                             ( pfczl *
                                                   ( rinvijik - costijk / r2ik -
                                                     p2etapl * riks / rik ) +
                                               pr2 * plambda );
                                        p3 =
                                            fg *
                                            ( pfczl * ( rinvijik +
                                                        p2etapl * rjks / rjk ) -
                                              pr3 * plambda );
                                    }
                                    else
                                    {
                                        p1 = fg * ( pfczl * ( rinvijik -
                                                              costijk / r2ij -
                                                              p2etapl ) +
                                                    pr1 * plambda );
                                        p2 = fg * ( pfczl * ( rinvijik -
                                                              costijk / r2ik -
                                                              p2etapl ) +
                                                    pr2 * plambda );
                                        p3 = fg *
                                             ( pfczl * ( rinvijik + p2etapl ) -
                                               pr3 * plambda );
                                    }
                                    f_a( i, 0 ) -= ( dEdG( i, globalIndex ) *
                                                     ( p1 * dxij + p2 * dxik ) *
                                                     CFFORCE * convForce );
                                    f_a( i, 1 ) -= ( dEdG( i, globalIndex ) *
                                                     ( p1 * dyij + p2 * dyik ) *
                                                     CFFORCE * convForce );
                                    f_a( i, 2 ) -= ( dEdG( i, globalIndex ) *
                                                     ( p1 * dzij + p2 * dzik ) *
                                                     CFFORCE * convForce );

                                    f_a( j, 0 ) += ( dEdG( i, globalIndex ) *
                                                     ( p1 * dxij + p3 * dxjk ) *
                                                     CFFORCE * convForce );
                                    f_a( j, 1 ) += ( dEdG( i, globalIndex ) *
                                                     ( p1 * dyij + p3 * dyjk ) *
                                                     CFFORCE * convForce );
                                    f_a( j, 2 ) += ( dEdG( i, globalIndex ) *
                                                     ( p1 * dzij + p3 * dzjk ) *
                                                     CFFORCE * convForce );

                                    f_a( k, 0 ) += ( dEdG( i, globalIndex ) *
                                                     ( p2 * dxik - p3 * dxjk ) *
                                                     CFFORCE * convForce );
                                    f_a( k, 1 ) += ( dEdG( i, globalIndex ) *
                                                     ( p2 * dyik - p3 * dyjk ) *
                                                     CFFORCE * convForce );
                                    f_a( k, 2 ) += ( dEdG( i, globalIndex ) *
                                                     ( p2 * dzik - p3 * dzjk ) *
                                                     CFFORCE * convForce );
                                } // l
                            }     // rjk <= rc
                        }         // rik <= rc
                    }             // elem
                }                 // rij <= rc
            }
        }
    };
    Cabana::neighbor_parallel_for( policy, calc_angular_force_op, neigh_list,
                                   Cabana::SecondNeighborsTag(), angle_op_tag,
                                   "Mode::calculateAngularForces" );
    Kokkos::fence();

    return;
}
