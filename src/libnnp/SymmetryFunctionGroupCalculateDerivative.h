//COMMENT

#include <types.h>
#include <system.h>
#include "CutoffFunction.h"
#include "SymmetryFunctionHelper.h"

using namespace std;
using namespace nnp;


__host__ __device__ void calculateSFGRD(System* s, AoSoA_NNP nnp_data, t_SF SF, t_SFscaling SFscaling, t_SFGmemberlist SFGmemberlist, t_dGdr dGdr, int attype, int groupIndex, t_verletlist_full_2D neigh_list, T_INT i)
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
   
    double pfc, pdfc;
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
            pdfc = 3.0 * temp2 * (temp2 - 1.0) / rc;
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
                // Force calculation.
                double const p1 = SFscaling(attype,k,6) * (pdfc - 2.0 * eta * (rij - rs) * pfc) * pexp / rij;
                // SIMPLE EXPRESSIONS:
                //Vec3D const dij = p1 * atom.neighbors[j].dr;
                //double const p1drijx = p1 * d1[0];
                //double const p1drijy = p1 * d1[1];
                //double const p1drijz = p1 * d1[2];

                // Save force contributions in Atom storage.
                size_t const ki = SFGmemberlist(attype,groupIndex,k);
                // SIMPLE EXPRESSIONS:
                //atom.dGdr[ki]              += dij;
                //atom.neighbors[j].dGdr[ki] -= dij;

                dGdr(i,ki,0) += (p1*dxij);
                dGdr(i,ki,1) += (p1*dyij);
                dGdr(i,ki,2) += (p1*dzij);

                dGdr(j,ki,0) -= (p1*dxij);
                dGdr(j,ki,1) -= (p1*dyij);
                dGdr(j,ki,2) -= (p1*dzij);
            }
        }
    }

    return;
}


__host__ __device__ void calculateSFGAND(System* s, AoSoA_NNP nnp_data, t_SF SF, t_SFscaling SFscaling, t_SFGmemberlist SFGmemberlist, t_dGdr dGdr, int attype, int groupIndex, t_verletlist_full_2D neigh_list, T_INT i) 
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
        if ((e1 == nej || e2 == nej) && rij < rc)
        {
            // Calculate cutoff function and derivative.
//#ifdef NOCFCACHE
            double pfcij;
            double pdfcij;
            double temp = tanh(1.0 - rij/rc);
            double temp2 = temp * temp;
            pfcij = temp * temp2;
            pdfcij = 3.0 * temp2 * (temp2 - 1.0) / rc;
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
                            double pdfcik;
                            double temp = tanh(1.0 - rik/rc);
                            double temp2 = temp * temp;
                            pfcik = temp * temp2;
                            pdfcik = 3.0 * temp2 * (temp2 - 1.0) / rc;
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
                            double pdfcjk;
                            temp = tanh(1.0 - rjk/rc);
                            temp2 = temp * temp;
                            pfcjk = temp * temp2;
                            pdfcjk = 3.0 * temp2 * (temp2 - 1.0) / rc;
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
                            double const pr1 = pfcik * pfcjk * pdfcij / rij;
                            double const pr2 = pfcij * pfcjk * pdfcik / rik;
                            double const pr3 = pfcij * pfcik * pdfcjk / rjk;
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
                                // Force calculation.
                                fg *= pow(2,(1-SF(attype,k,6))) * SFscaling(attype,k,6);
                                double const pfczl = pfc * SF(attype,k,6) * SF(attype,k,5);
                                double factorDeriv = 2.0 * SF(attype,k,4) / SF(attype,k,6) / SF(attype,k,5);
                                double const p2etapl = plambda * factorDeriv;
                                double p1;
                                double p2;
                                double p3;
                                if (SF(attype,1,8) > 0.0)
                                {
                                    p1 = fg * (pfczl * (rinvijik
                                       - costijk / r2ij - p2etapl
                                       * rijs / rij) + pr1 * plambda);
                                    p2 = fg * (pfczl * (rinvijik
                                       - costijk / r2ik - p2etapl
                                       * riks / rik) + pr2 * plambda);
                                    p3 = fg * (pfczl * (rinvijik
                                       + p2etapl * rjks / rjk)
                                       - pr3 * plambda);
                                }
                                else
                                {
                                    p1 = fg * (pfczl * (rinvijik - costijk
                                       / r2ij - p2etapl) + pr1 * plambda);
                                    p2 = fg * (pfczl * (rinvijik - costijk
                                       / r2ik - p2etapl) + pr2 * plambda);
                                    p3 = fg * (pfczl * (rinvijik + p2etapl)
                                       - pr3 * plambda);
                                }

                                // Save force contributions in Atom storage.
                                //
                                // SIMPLE EXPRESSIONS:
                                //size_t const li = members[l]->getIndex();
                                //atom.dGdr[li] += p1 * drij + p2 * drik;
                                //atom.neighbors[j].dGdr[li] -= p1 * drij
                                //                            + p3 * drjk;
                                //atom.neighbors[k].dGdr[li] -= p2 * drik
                                //                            - p3 * drjk;
                                /*
                                double const p1drijx = p1 * dxij;
                                double const p1drijy = p1 * dyij;
                                double const p1drijz = p1 * dzij;

                                double const p2drikx = p2 * dxik;
                                double const p2driky = p2 * dyik;
                                double const p2drikz = p2 * dzik;

                                double const p3drjkx = p3 * dxjk;
                                double const p3drjky = p3 * dyjk;
                                double const p3drjkz = p3 * dzjk;*/
                                
                                size_t const li = SFGmemberlist(attype,groupIndex,l);
                                dGdr(i,li,0) += (p1*dxij + p2*dxik);
                                dGdr(i,li,1) += (p1*dyij + p2*dyik);
                                dGdr(i,li,2) += (p1*dzij + p2*dzik);

                                dGdr(j,li,0) -= (p1*dxij + p3*dxjk);
                                dGdr(j,li,1) -= (p1*dyij + p3*dyjk);
                                dGdr(j,li,2) -= (p1*dzij + p3*dzjk);

                                dGdr(k,li,0) -= (p2*dxik - p3*dxjk);
                                dGdr(k,li,1) -= (p2*dyik - p3*dyjk);
                                dGdr(k,li,2) -= (p2*dzik - p3*dzjk);
                            } // l
                        } // rjk <= rc
                    } // rik <= rc
                } // elem
            } // k
        } // rij <= rc
    } // j

    return;
}
