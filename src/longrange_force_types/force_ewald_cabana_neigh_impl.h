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

template<class t_neighbor>
ForceEwald<t_neighbor>::ForceEwald(System* system, bool half_neigh_, MPI_Comm comm):LRForce(system,half_neigh_,comm) {
    half_neigh = half_neigh_;
    assert( half_neigh == true );
    MPI_Topo_test( comm, &comm_type );
    assert( comm_type == MPI_CART );
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

//TODO: just grab neighborlist already created by Force
//template<class t_neighbor>
//void ForceEwald<t_neighbor>::create_neigh_list(System* system) {
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
//}

template<class t_neighbor>
void ForceEwald<t_neighbor>::compute(System* system) {

  double Ur = 0.0, Uk = 0.0, Uself = 0.0, Udip = 0.0;
  double Udip_vec[3];

  N_local = system->N_local;//TODO: rename this n_max or other way around?
  x = Cabana::slice<Positions>(system->xvf);
  f = Cabana::slice<Forces>(system->xvf);
  f_a = Cabana::slice<Forces>(system->xvf);//TODO: What is this?
  id = Cabana::slice<IDs>(system->xvf);
  //type = Cabana::slice<Types>(system->xvf);
  q = Cabana::slice<Charges>(system->xvf);
  p = Cabana::slice<Potentials>//TODO: Should we have potentials as part of the system AoSoA?

  //Kokkos::View<double *, MemorySpace> domain_size( "domain size", 3);
  //TODO: How do the "domain" and "subdomain" of system compare to Rene's use of "domain" in ewald?
  //domain_size( 0 ) = system->domain_hi_x - system_domain_x;
  //domain_size( 1 ) = system->domain_hi_y - system_domain_y;
  //domain_size( 2 ) = system->domain_hi_z - system_domain_z;
  
  //get the solver parameters
  double alpha = _alpha;
  double r_max = _r_max;
  double eps_r = _eps_r;
  double k_max = _k_max;
  
  auto init_p = KOKKOS_LAMBDA( const int idx )
  {
    p( idx ) = 0.0;
  };//TODO: Is n_max here the N_local? I think so.
  Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ), init_parameters );
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

  //Compute partial sums
  Kokkkos::parallel_for( n_max, KOKKOS_LAMBDA( const int idx ) {
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
    } );
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

    Kokkos::parallel_reduce(
        n_max,
        KOKKOS_LAMBDA( const int idx, double &Uk_part ) {
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
                            double kr = k[0] * r( idx, 0 ) + k[1] * r( idx, 1 ) +
                                        k[2] * r( idx, 2 );

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
                            Uk_part += contrib;

                            for ( int dim = 0; dim < 3; ++dim )
                                f( idx, dim ) +=
                                    k_coeff * 2.0 * q( idx ) * k[dim] *
                                    ( U_trigonometric( 2 * kidx + 1 ) * cos( kr ) -
                                      U_trigonometric( 2 * kidx ) * sin( kr ) );
                    }
                }
            }
        },
        Uk );
    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &Uk, 1, MPI_DOUBLE, MPI_SUM, comm );

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
// TODO: change comment to reflect usage of CabanaMD neighborlist


//TODO: now in Ewald implementation is the creation of a neighbor 
//      list with the construction of a halo
//      How to do all the work there but with the neighbor list 
//      already created in short-range force calcs?


//TODO: progress is up to line 338 in ewald.cpp where real-space
//      contributions are calculated




template<class t_neighbor>
T_V_FLOAT ForceLJ<t_neighbor>::compute_energy(System* system) {
  N_local = system->N_local;
  x = Cabana::slice<Positions>(system->xvf);
  f = Cabana::slice<Forces>(system->xvf);
  f_a = Cabana::slice<Forces>(system->xvf);
  id = Cabana::slice<IDs>(system->xvf);
  type = Cabana::slice<Types>(system->xvf);

  T_V_FLOAT energy;

  if(half_neigh)
    Kokkos::parallel_reduce("ForceLJCabanaNeigh::compute_energy", t_policy_half_neigh_pe_stackparams(0, system->N_local), *this, energy);
  else
    Kokkos::parallel_reduce("ForceLJCabanaNeigh::compute_energy", t_policy_full_neigh_pe_stackparams(0, system->N_local), *this, energy);

  Kokkos::fence();

  step++;
  return energy;
}

template<class t_neighbor>
const char* ForceLJ<t_neighbor>::name() {return half_neigh?"Force:LJCabanaVerletHalf":"Force:LJCabanaVerletFull";}

template<class t_neighbor>
KOKKOS_INLINE_FUNCTION
void ForceLJ<t_neighbor>::operator() (TagFullNeigh, const T_INT& i) const {
  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);

  int num_neighs = Cabana::NeighborList<t_neighbor>::numNeighbor(neigh_list, i);

  T_F_FLOAT fxi = 0.0;
  T_F_FLOAT fyi = 0.0;
  T_F_FLOAT fzi = 0.0;

  for(int jj = 0; jj < num_neighs; jj++) {
    int j = Cabana::NeighborList<t_neighbor>::getNeighbor(neigh_list, i, jj);

    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    const T_F_FLOAT cutsq_ij = stack_cutsq[type_i][type_j];

    if( rsq < cutsq_ij ) {
      const T_F_FLOAT lj1_ij = stack_lj1[type_i][type_j];
      const T_F_FLOAT lj2_ij = stack_lj2[type_i][type_j];

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      T_F_FLOAT fpair = (r6inv * (lj1_ij*r6inv - lj2_ij)) * r2inv;
      fxi += dx*fpair;
      fyi += dy*fpair;
      fzi += dz*fpair;
    }
  }

  f(i,0) += fxi;
  f(i,1) += fyi;
  f(i,2) += fzi;

}

template<class t_neighbor>
KOKKOS_INLINE_FUNCTION
void ForceLJ<t_neighbor>::operator() (TagHalfNeigh, const T_INT& i) const {
  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);

  int num_neighs = Cabana::NeighborList<t_neighbor>::numNeighbor(neigh_list, i);

  T_F_FLOAT fxi = 0.0;
  T_F_FLOAT fyi = 0.0;
  T_F_FLOAT fzi = 0.0;
  for(int jj = 0; jj < num_neighs; jj++) {
    int j = Cabana::NeighborList<t_neighbor>::getNeighbor(neigh_list, i, jj);

    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    const T_F_FLOAT cutsq_ij = stack_cutsq[type_i][type_j];

    if( rsq < cutsq_ij ) {
      const T_F_FLOAT lj1_ij = stack_lj1[type_i][type_j];
      const T_F_FLOAT lj2_ij = stack_lj2[type_i][type_j];

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      T_F_FLOAT fpair = (r6inv * (lj1_ij*r6inv - lj2_ij)) * r2inv;
      fxi += dx*fpair;
      fyi += dy*fpair;
      fzi += dz*fpair;
      f_a(j,0) -= dx*fpair;
      f_a(j,1) -= dy*fpair;
      f_a(j,2) -= dz*fpair;
    }
  }
  f_a(i,0) += fxi;
  f_a(i,1) += fyi;
  f_a(i,2) += fzi;

}

template<class t_neighbor>
KOKKOS_INLINE_FUNCTION
void ForceLJ<t_neighbor>::operator() (TagFullNeighPE, const T_INT& i, T_V_FLOAT& PE) const {
  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);
  const bool shift_flag = true;

  int num_neighs = Cabana::NeighborList<t_neighbor>::numNeighbor(neigh_list, i);

  for(int jj = 0; jj < num_neighs; jj++) {
    int j = Cabana::NeighborList<t_neighbor>::getNeighbor(neigh_list, i, jj);

    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    const T_F_FLOAT cutsq_ij = stack_cutsq[type_i][type_j];

    if( rsq < cutsq_ij ) {
      const T_F_FLOAT lj1_ij = stack_lj1[type_i][type_j];
      const T_F_FLOAT lj2_ij = stack_lj2[type_i][type_j];

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      PE += 0.5*r6inv * (0.5*lj1_ij*r6inv - lj2_ij) / 6.0; // optimize later

      if (shift_flag) {
        T_F_FLOAT r2invc = 1.0/cutsq_ij;
        T_F_FLOAT r6invc = r2invc*r2invc*r2invc;
        PE -= 0.5*r6invc * (0.5*lj1_ij*r6invc - lj2_ij) / 6.0; // optimize later
      }
    }
  }
}

template<class t_neighbor>
KOKKOS_INLINE_FUNCTION
void ForceLJ<t_neighbor>::operator() (TagHalfNeighPE, const T_INT& i, T_V_FLOAT& PE) const {
  const T_F_FLOAT x_i = x(i,0);
  const T_F_FLOAT y_i = x(i,1);
  const T_F_FLOAT z_i = x(i,2);
  const int type_i = type(i);
  const bool shift_flag = true;

  int num_neighs = Cabana::NeighborList<t_neighbor>::numNeighbor(neigh_list, i);

  for(int jj = 0; jj < num_neighs; jj++) {
    int j = Cabana::NeighborList<t_neighbor>::getNeighbor(neigh_list, i, jj);

    const T_F_FLOAT dx = x_i - x(j,0);
    const T_F_FLOAT dy = y_i - x(j,1);
    const T_F_FLOAT dz = z_i - x(j,2);

    const int type_j = type(j);
    const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

    const T_F_FLOAT cutsq_ij = stack_cutsq[type_i][type_j];

    if( rsq < cutsq_ij ) {
      const T_F_FLOAT lj1_ij = stack_lj1[type_i][type_j];
      const T_F_FLOAT lj2_ij = stack_lj2[type_i][type_j];

      T_F_FLOAT r2inv = 1.0/rsq;
      T_F_FLOAT r6inv = r2inv*r2inv*r2inv;
      T_F_FLOAT fac;
      if(j<N_local) fac = 1.0;
      else fac = 0.5;

      PE += fac * r6inv * (0.5*lj1_ij*r6inv - lj2_ij) / 6.0;  // optimize later

      if (shift_flag) {
        T_F_FLOAT r2invc = 1.0/cutsq_ij;
        T_F_FLOAT r6invc = r2invc*r2invc*r2invc;
        PE -= fac * r6invc * (0.5*lj1_ij*r6invc - lj2_ij) / 6.0;  // optimize later
      }
    }
  }

}
