#include <types.h>
#include <system.h>
#include "CutoffFunction.h"
#include "SymmetryFunctionHelper.h"


__host__ __device__ double scale(int attype, double value, int k, t_SFscaling SFscaling)
{
    double scalingType = SFscaling(attype,k,7);
    double scalingFactor = SFscaling(attype,k,6);
    double Gmin = SFscaling(attype,k,0);
    double Gmax = SFscaling(attype,k,1);
    double Gmean = SFscaling(attype,k,2);
    double Gsigma = SFscaling(attype,k,3);
    double Smin = SFscaling(attype,k,4);
    double Smax = SFscaling(attype,k,5);
    
    if (scalingType == 0.0)
    {
        return value;
    }
    else if (scalingType == 1.0)
    {
        return Smin + scalingFactor * (value - Gmin);
    }
    else if (scalingType == 2.0)
    {
        return value - Gmean;
    }
    else if (scalingType == 3.0)
    {
        return Smin + scalingFactor * (value - Gmean);
    }
    else if (scalingType == 4.0)
    {
        return Smin + scalingFactor * (value - Gmean);
    }
    else
    {
        return 0.0;
    }

}

__host__ __device__ void calculateSFGR(System* s, AoSoA_NNP nnp_data, t_SF SF, t_SFscaling SFscaling, t_SFGmemberlist SFGmemberlist, int attype, int groupIndex, t_verletlist_full_2D neigh_list, T_INT i)
{
    auto x = Cabana::slice<Positions>(s->xvf);
    auto id = Cabana::slice<IDs>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto G = Cabana::slice<NNPNames::G>(nnp_data);
    
    //pick e1, rc from first member (since they are all the same)
    //SFGmemberlist(attype,groupIndex,0): second index = groupIndex, third index  = for first SF in group
    int e1 = SF(attype, SFGmemberlist(attype,groupIndex,0), 2);
    double rc = SF(attype, SFGmemberlist(attype,groupIndex,0), 7);
    //CutoffFunction fc(rc);
    
    int size;
    int num_neighs = Cabana::NeighborList<t_verletlist_full_2D>::numNeighbor(neigh_list, i);
    size_t numNeighbors = num_neighs;
   
    double pfc;
    for (size_t jj = 0; jj < numNeighbors; ++jj)
    {
        //Atom::Neighbor& n = atom.neighbors[j];
        int j = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, jj);
        //std::cout << "i: " << id(i) << " j: " << id(j) << std::endl; 
        //std::cout << "xi: " << x(i,0) << " " << x(i,1) << " " << x(i,2) << std::endl; 
        //std::cout << "xj: " << x(j,0) << " " << x(j,1) << " " << x(j,2) << std::endl; 
        T_F_FLOAT dxij = x(i,0) - x(j,0);
        T_F_FLOAT dyij = x(i,1) - x(j,1);
        T_F_FLOAT dzij = x(i,2) - x(j,2);
        dxij *= s->cflength;
        dyij *= s->cflength;
        dzij *= s->cflength;
        
        if (s->normalize) {
          dxij *= s->convLength;
          dyij *= s->convLength;
          dzij *= s->convLength;
        }
        
        double const r2ij = dxij*dxij + dyij*dyij + dzij*dzij;
        double const rij = sqrt(r2ij); 
        
        if (e1 == type(j) && rij < rc)
        {
            // Energy calculation.
            // Calculate cutoff function and derivative.
//#ifdef NOCFCACHE
            double temp = tanh(1.0 - rij / rc);
            double temp2 = temp * temp;
            pfc = temp * temp2;
            
            //fc.fdf(rij, pfc, pdfc);
//#else
            /*
            // If cutoff radius matches with the one in the neighbor storage
            // we can use the previously calculated value.
            double& pfc = n.fc;
            double& pdfc = n.dfc;
            if (n.cutoffType != cutoffType ||
                n.rc != rc ||
                n.cutoffAlpha != cutoffAlpha)
            {
                fc.fdf(rij, pfc, pdfc);
                n.rc = rc;
                n.cutoffType = cutoffType;
                n.cutoffAlpha = cutoffAlpha;
            }*/
//#endif
            //double const* const d1 = n.dr.r;
            //TODO: use subview for size calculation
            //auto SFGmemberlist_subview = Kokkos::subview(SFGmemberlist, make_pair(0,1), make_pair(0,Kokkos::ALL));
            for (int l=0; l < MAX_SF; ++l)
            {
                if (SFGmemberlist(attype,groupIndex,l) == 0 && SFGmemberlist(attype,groupIndex,l+1) == 0)
                {
                  size = l;
                  break;
                }
            }
            for (size_t k = 0; k < size; ++k) 
            {
                double eta = SF(attype, SFGmemberlist(attype,groupIndex,k), 4);
                double rs = SF(attype, SFGmemberlist(attype,groupIndex,k), 8);
                double pexp = exp(-eta * (rij - rs) * (rij - rs));
                //if (k == 0)
                //std::cout << "member: " << SFGmemberlist(attype,groupIndex,k) << " " ;
                //std::cout << eta << " " << rs << " " << rij << " " << pexp << " " << pfc << " " << pexp*pfc << std::endl;
                G(i,SFGmemberlist(attype,groupIndex,k)) += pexp*pfc;
            }
        }
    }

    double raw_value;
    for (size_t k = 0; k < size; ++k)
    {
        raw_value = G(i,SFGmemberlist(attype,groupIndex,k)); 
        G(i,SFGmemberlist(attype,groupIndex,k)) = scale(attype, raw_value, SFGmemberlist(attype,groupIndex,k), SFscaling);
    }
    return;
}


__host__ __device__ void calculateSFGAN(System* s, AoSoA_NNP nnp_data, t_SF SF, t_SFscaling SFscaling, t_SFGmemberlist SFGmemberlist, int attype, int groupIndex, t_verletlist_full_2D neigh_list, T_INT i) 
{
    auto x = Cabana::slice<Positions>(s->xvf);
    auto id = Cabana::slice<IDs>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto G = Cabana::slice<NNPNames::G>(nnp_data);
    
    //pick e1, rc from first member (since they are all the same)
    //SFGmemberlist(1,0): first index for angular narrow, second index for first SF in angular narrow group
    int e1 = SF(attype, SFGmemberlist(attype,groupIndex,0), 2);
    int e2 = SF(attype, SFGmemberlist(attype,groupIndex,0), 3);
    double rc = SF(attype, SFGmemberlist(attype,groupIndex,0), 7);
    //CutoffFunction fc(rc);
    
    int size;
    int num_neighs = Cabana::NeighborList<t_verletlist_full_2D>::numNeighbor(neigh_list, i);
    double const rc2 = rc * rc;
    size_t numNeighbors = num_neighs; 
    // Prevent problematic condition in loop test below (j < numNeighbors - 1).
    if (numNeighbors == 0) numNeighbors = 1;

    for (size_t jj = 0; jj < numNeighbors - 1; jj++)
    {
        //Atom::Neighbor& nj = atom.neighbors[j];
        int j = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, jj);
        size_t const nej = type(j);
        
        //std::cout << "i: " << i << " j: " << j << std::endl; 
        //std::cout << "xi: " << x(i,0) << " " << x(i,1) << " " << x(i,2) << std::endl; //<< " " << x(i,1) << " " << x(i,2) << std::endl; 
        //std::cout << "xj: " << x(j,0) << " " << x(j,1) << " " << x(j,2) << std::endl; 
        
        T_F_FLOAT dxij = x(i,0) - x(j,0);
        T_F_FLOAT dyij = x(i,1) - x(j,1);
        T_F_FLOAT dzij = x(i,2) - x(j,2);
        dxij *= s->cflength;
        dyij *= s->cflength;
        dzij *= s->cflength;
        if (s->normalize) {
          dxij *= s->convLength;
          dyij *= s->convLength;
          dzij *= s->convLength;
        }
        double const r2ij = dxij*dxij + dyij*dyij + dzij*dzij;
        double const rij = sqrt(r2ij);
        if ((e1 == nej || e2 == nej) && rij < rc)
        {
            // Calculate cutoff function and derivative.
//#ifdef NOCFCACHE
            double pfcij;
            double temp = tanh(1.0 - rij/rc);
            double temp2 = temp * temp;
            pfcij = temp * temp2;
            //fc.fdf(rij, pfcij, pdfcij);
//#else
            /*// If cutoff radius matches with the one in the neighbor storage
            // we can use the previously calculated value.
            double& pfcij = nj.fc;
            double& pdfcij = nj.dfc;
            if (nj.cutoffType != cutoffType ||
                nj.rc != rc ||
                nj.cutoffAlpha != cutoffAlpha)
            {
                fc.fdf(rij, pfcij, pdfcij);
                nj.rc = rc;
                nj.cutoffType = cutoffType;
                nj.cutoffAlpha = cutoffAlpha;
            }*/
//#endif
            // SIMPLE EXPRESSIONS:
            //Vec3D const drij(atom.neighbors[j].dr);
            //double const* const dr1 = drij.r;

            for (size_t kk = jj + 1; kk < numNeighbors; kk++)
            {
                int k = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, kk);
                //Atom::Neighbor& nk = atom.neighbors[k];
                size_t const nek = type(k);
                
                if ((e1 == nej && e2 == nek) ||
                    (e2 == nej && e1 == nek))
                {
                    T_F_FLOAT dxik = x(i,0) - x(k,0);
                    T_F_FLOAT dyik = x(i,1) - x(k,1);
                    T_F_FLOAT dzik = x(i,2) - x(k,2);
                    dxik *= s->cflength;
                    dyik *= s->cflength;
                    dzik *= s->cflength;
                    
                    if (s->normalize) {
                      dxik *= s->convLength;
                      dyik *= s->convLength;
                      dzik *= s->convLength;
                    }
                    double const r2ik = dxik*dxik + dyik*dyik + dzik*dzik;
                    double const rik = sqrt(r2ik);
                
                    if (rik < rc)
                    {
                        // SIMPLE EXPRESSIONS:
                        //Vec3D const drjk(atom.neighbors[k].dr
                        //               - atom.neighbors[j].dr);
                        //double rjk = drjk.norm2();
                        double dxjk = dxik - dxij;
                        double dyjk = dyik - dyij;
                        double dzjk = dzik - dzij;
                        double r2jk = dxjk*dxjk + dyjk*dyjk + dzjk*dzjk; 
                        if (r2jk < rc2)
                        {
                            // Energy calculation.
//#ifdef NOCFCACHE
                            double pfcik;
                            double temp = tanh(1.0 - rik/rc);
                            double temp2 = temp * temp;
                            pfcik = temp * temp2;
                            //fc.fdf(rik, pfcik, pdfcik);
//#else
                            /*double& pfcik = nk.fc;
                            double& pdfcik = nk.dfc;
                            if (nk.cutoffType != cutoffType ||
                                nk.rc != rc ||
                                nk.cutoffAlpha != cutoffAlpha)
                            {
                                fc.fdf(rik, pfcik, pdfcik);
                                nk.rc = rc;
                                nk.cutoffType = cutoffType;
                                nk.cutoffAlpha = cutoffAlpha;
                            }*/
//#endif
                            double rjk = sqrt(r2jk);
                            double pfcjk;
                            temp = tanh(1.0 - rjk/rc);
                            temp2 = temp * temp;
                            pfcjk = temp * temp2;
                            //fc.fdf(rjk, pfcjk, pdfcjk);

                            // SIMPLE EXPRESSIONS:
                            //Vec3D const drik(atom.neighbors[k].dr);
                            //double const* const dr2 = drik.r;
                            //double const* const dr3 = drjk.r;
                            double const rinvijik = 1.0 / rij / rik;
                            // SIMPLE EXPRESSIONS:
                            //double const costijk = (drij * drik) * rinvijik;
                            double const costijk = (dxij*dxik + dyij*dyik + dzij*dzik)* rinvijik;
                            double const pfc = pfcij * pfcik * pfcjk;
                            double const r2sum = r2ij + r2ik + r2jk;
                            //double const pr1 = pfcik * pfcjk * pdfcij / rij;
                            //double const pr2 = pfcij * pfcjk * pdfcik / rik;
                            //double const pr3 = pfcij * pfcik * pdfcjk / rjk;
                            double vexp = 0.0;
                            double rijs = 0.0;
                            double riks = 0.0;
                            double rjks = 0.0;
                            for (int ll=0; ll < MAX_SF; ++ll)
                            {
                                if (SFGmemberlist(attype,groupIndex,ll) == 0 && SFGmemberlist(attype,groupIndex,ll+1) == 0)
                                {
                                  size = ll;
                                  break;
                                }
                            }
                            for (size_t l = 0; l < size; ++l)
                            {
                              if (SF(attype,l,8) > 0.0)
                              {  
                                rijs = rij - SF(attype,l,8);
                                riks = rik - SF(attype,l,8);
                                rjks = rjk - SF(attype,l,8);
                                vexp = exp(-SF(attype,l,4) * (rijs * rijs
                                                    + riks * riks
                                                    + rjks * rjks));
                              }
                              else
                                  vexp = exp(-SF(attype,l,4) * r2sum);
                                /*if (calculateExp[l])
                                {
                                    if (rs[l] > 0.0)
                                    {
                                        rijs = rij - rs[l];
                                        riks = rik - rs[l];
                                        rjks = rjk - rs[l];
                                        vexp = exp(-eta[l] * (rijs * rijs
                                                            + riks * riks
                                                            + rjks * rjks));
                                    }
                                    else
                                    {
                                        vexp = exp(-eta[l] * r2sum);
                                    }
                                }*/
                                double const plambda = 1.0
                                                     + SF(attype,l,5) * costijk;
                                double fg = vexp;
                                if (plambda <= 0.0) fg = 0.0;
                                else
                                {
                                    fg *= pow(plambda, (SF(attype,l,6) - 1.0));
                                    /*if (useIntegerPow[l])
                                    {
                                        fg *= pow_int(plambda, zetaInt[l] - 1);
                                    }
                                    else
                                    {
                                        fg *= pow(plambda, zeta[l] - 1.0);
                                    }*/
                                }
                                G(i,SFGmemberlist(attype,groupIndex,l)) += fg * plambda * pfc;

                            } // l
                        } // rjk <= rc
                    } // rik <= rc
                } // elem
            } // k
        } // rij <= rc
    } // j

    double raw_value = 0.0;
    for (size_t k = 0; k < size; ++k)
    {
        raw_value = G(i,SFGmemberlist(attype,groupIndex,k)) * pow(2,(1-SF(attype,k,6))); 
        G(i,SFGmemberlist(attype,groupIndex,k)) = scale(attype, raw_value, SFGmemberlist(attype,groupIndex,k), SFscaling);
    }

    return;
}

