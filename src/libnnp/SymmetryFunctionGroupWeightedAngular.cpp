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

#include "SymmetryFunctionGroupWeightedAngular.h"
#include "Atom.h"
#include "SymmetryFunction.h"
#include "SymmetryFunctionWeightedAngular.h"
#include "Vec3D.h"
#include "utility.h"
#include <algorithm> // std::sort
#include <cmath>     // exp
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

SymmetryFunctionGroupWeightedAngular::
SymmetryFunctionGroupWeightedAngular(ElementMap const& elementMap) :
    SymmetryFunctionGroup(13, elementMap)
{
    parametersMember.insert("eta");
    parametersMember.insert("rs");
    parametersMember.insert("lambda");
    parametersMember.insert("zeta");
    parametersMember.insert("mindex");
    parametersMember.insert("sfindex");
    parametersMember.insert("calcexp");
}

bool SymmetryFunctionGroupWeightedAngular::
operator==(SymmetryFunctionGroup const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymmetryFunctionGroupWeightedAngular const& c =
        dynamic_cast<SymmetryFunctionGroupWeightedAngular const&>(rhs);
    if (cutoffType  != c.cutoffType ) return false;
    if (cutoffAlpha != c.cutoffAlpha) return false;
    if (rc          != c.rc         ) return false;
    return true;
}

bool SymmetryFunctionGroupWeightedAngular::
operator<(SymmetryFunctionGroup const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymmetryFunctionGroupWeightedAngular const& c =
        dynamic_cast<SymmetryFunctionGroupWeightedAngular const&>(rhs);
    if      (cutoffType  < c.cutoffType ) return true;
    else if (cutoffType  > c.cutoffType ) return false;
    if      (cutoffAlpha < c.cutoffAlpha) return true;
    else if (cutoffAlpha > c.cutoffAlpha) return false;
    if      (rc          < c.rc         ) return true;
    else if (rc          > c.rc         ) return false;
    return false;
}

bool SymmetryFunctionGroupWeightedAngular::
addMember(SymmetryFunction const* const symmetryFunction)
{
    if (symmetryFunction->getType() != type) return false;

    SymmetryFunctionWeightedAngular const* sf =
        dynamic_cast<SymmetryFunctionWeightedAngular const*>(
        symmetryFunction);

    if (members.empty())
    {
        cutoffType  = sf->getCutoffType();
        cutoffAlpha = sf->getCutoffAlpha();
        ec          = sf->getEc();
        rc          = sf->getRc();
        convLength  = sf->getConvLength();

        fc.setCutoffType(cutoffType);
        fc.setCutoffRadius(rc);
        fc.setCutoffParameter(cutoffAlpha);
    }

    if (sf->getCutoffType()  != cutoffType ) return false;
    if (sf->getCutoffAlpha() != cutoffAlpha) return false;
    if (sf->getEc()          != ec         ) return false;
    if (sf->getRc()          != rc         ) return false;
    if (sf->getConvLength()  != convLength )
    {
        throw runtime_error("ERROR: Unable to add symmetry function members "
                            "with different conversion factors.\n");
    }

    members.push_back(sf);

    return true;
}

void SymmetryFunctionGroupWeightedAngular::sortMembers()
{
    sort(members.begin(),
         members.end(),
         comparePointerTargets<SymmetryFunctionWeightedAngular const>);

    // Members are now sorted with eta changing the slowest.
    for (size_t i = 0; i < members.size(); i++)
    {
        factorNorm.push_back(pow(2.0, 1.0 - members[i]->getZeta()));
        factorDeriv.push_back(2.0 * members[i]->getEta() /
                              members[i]->getZeta() / members[i]->getLambda());
        if (i == 0)
        {
            calculateExp.push_back(true);
        }
        else
        {
            if (members[i - 1]->getEta() != members[i]->getEta() ||
                members[i - 1]->getRs()  != members[i]->getRs() )
            {
                calculateExp.push_back(true);
            }
            else
            {
                calculateExp.push_back(false);
            }
        }
        useIntegerPow.push_back(members[i]->getUseIntegerPow());
        memberIndex.push_back(members[i]->getIndex());
        zetaInt.push_back(members[i]->getZetaInt());
        eta.push_back(members[i]->getEta());
        rs.push_back(members[i]->getRs());
        lambda.push_back(members[i]->getLambda());
        zeta.push_back(members[i]->getZeta());
        zetaLambda.push_back(members[i]->getZeta() * members[i]->getLambda());
    }

    return;
}

void SymmetryFunctionGroupWeightedAngular::setScalingFactors()
{
    scalingFactors.resize(members.size(), 0.0);
    for (size_t i = 0; i < members.size(); i++)
    {
        scalingFactors.at(i) = members[i]->getScalingFactor();
        factorNorm.at(i) *= scalingFactors.at(i);
    }

    return;
}

// Depending on chosen symmetry functions this function may be very
// time-critical when predicting new structures (e.g. in MD simulations). Thus,
// lots of optimizations were used sacrificing some readablity. Vec3D
// operations have been rewritten in simple C array style and the use of
// temporary objects has been minimized. Some of the originally coded
// expressions are kept in comments marked with "SIMPLE EXPRESSIONS:".
KOKKOS_INLINE_FUNCTION void SymmetryFunctionGroupWeightedAngular::calculate(System* s, AoSoA_NNP nnp_data, t_verletlist_full_2D neigh_list,
                                                  T_INT i, bool const derivatives) const
{
    auto x = Cabana::slice<Positions>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto G = Cabana::slice<NNPNames::G>(nnp_data);

    double* result = new double[members.size()];
    for (size_t l = 0; l < members.size(); ++l)
    {
        result[l] = 0.0;
    }

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

        if (rij < rc)
        {
            // Calculate cutoff function and derivative.
//#ifdef NOCFCACHE
            double pfcij;
            double pdfcij;
            fc.fdf(rij, pfcij, pdfcij);
//#else
            /*
            // If cutoff radius matches with the one in the neighbor storage
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
            //double const* const dr1 = nj.dr.r;

            for (size_t kk = jj + 1; kk < numNeighbors; kk++)
            {
                int k = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, kk);
                //Atom::Neighbor& nk = atom.neighbors[k];
                size_t const nek = type(k);
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
                        fc.fdf(rik, pfcik, pdfcik);
//#else
                        /*
                        double& pfcik = nk.fc;
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
                        fc.fdf(rjk, pfcjk, pdfcjk);

                        // SIMPLE EXPRESSIONS:
                        //Vec3D const drik(atom.neighbors[k].dr);
                        //double const* const dr2 = drik.r;
                        //double const* const dr3 = drjk.r;
                        double const rinvijik = 1.0 / rij / rik;
                        // SIMPLE EXPRESSIONS:
                        //double const costijk = (drij * drik) * rinvijik;
                        double const costijk = (dxij*dxik + dyij*dyik + dzij*dzik)* rinvijik;
                        double const z = elementMap.atomicNumber(nej)
                                       * elementMap.atomicNumber(nek);
                        double const pfc = z * pfcij * pfcik * pfcjk;
                        double const pr1 = z * pfcik * pfcjk * pdfcij / rij;
                        double const pr2 = z * pfcij * pfcjk * pdfcik / rik;
                        double const pr3 = z * pfcij * pfcik * pdfcjk / rjk;
                        double vexp = 0.0;
                        double rijs = 0.0;
                        double riks = 0.0;
                        double rjks = 0.0;

                        for (size_t l = 0; l < members.size(); ++l)
                        {
                            if (calculateExp[l])
                            {
                                rijs = rij - rs[l];
                                riks = rik - rs[l];
                                rjks = rjk - rs[l];
                                double const r2sum = rijs * rijs
                                                   + riks * riks
                                                   + rjks * rjks;
                                vexp = exp(-eta[l] * r2sum);
                            }
                            double const plambda = 1.0
                                                 + lambda[l] * costijk;
                            double fg = vexp;
                            if (plambda <= 0.0) fg = 0.0;
                            else
                            {
                                if (useIntegerPow[l])
                                {
                                    fg *= pow_int(plambda, zetaInt[l] - 1);
                                }
                                else
                                {
                                    fg *= pow(plambda, zeta[l] - 1.0);
                                }
                            }
                            result[l] += fg * plambda * pfc;

                            // Force calculation.
                            if (!derivatives) continue;
                        } // l
                    } // rjk <= rc
                } // rik <= rc
            } // k
        } // rij <= rc
    } // j

    for (size_t l = 0; l < members.size(); ++l)
    {
        result[l] *= factorNorm[l] / scalingFactors[l];
        G(i, memberIndex[l]) = members[l]->scale(result[l]);
    }

    delete[] result;

    return;
}

KOKKOS_INLINE_FUNCTION void SymmetryFunctionGroupWeightedAngular::calculate_derivatives(System* s, AoSoA_NNP nnp_data, t_dGdr dGdr, t_verletlist_full_2D neigh_list, T_INT i) const
{
    auto x = Cabana::slice<Positions>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto G = Cabana::slice<NNPNames::G>(nnp_data);

    int num_neighs = Cabana::NeighborList<t_verletlist_full_2D>::numNeighbor(neigh_list, i);
    double const rc2 = rc * rc;
    size_t numNeighbors = num_neighs;
    // Prevent problematic condition in loop test below (j < numNeighbors - 1).
    if (numNeighbors == 0) numNeighbors = 1;

    for (size_t jj = 0; jj < numNeighbors - 1; jj++)
    {
        int j = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, jj);
        size_t const nej = type(j);
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

        if (rij < rc)
        {
            // Calculate cutoff function and derivative.
            double pfcij;
            double pdfcij;
            fc.fdf(rij, pfcij, pdfcij);
            for (size_t kk = jj + 1; kk < numNeighbors; kk++)
            {
                int k = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, kk);
                //Atom::Neighbor& nk = atom.neighbors[k];
                size_t const nek = type(k);
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
                        double pfcik;
                        double pdfcik;
                        fc.fdf(rik, pfcik, pdfcik);
                        double rjk = sqrt(r2jk);

                        double pfcjk;
                        double pdfcjk;
                        fc.fdf(rjk, pfcjk, pdfcjk);

                        // SIMPLE EXPRESSIONS:
                        //Vec3D const drik(atom.neighbors[k].dr);
                        //double const* const dr2 = drik.r;
                        //double const* const dr3 = drjk.r;
                        double const rinvijik = 1.0 / rij / rik;
                        // SIMPLE EXPRESSIONS:
                        //double const costijk = (drij * drik) * rinvijik;
                        double const costijk = (dxij*dxik + dyij*dyik + dzij*dzik)* rinvijik;
                        double const z = elementMap.atomicNumber(nej)
                                       * elementMap.atomicNumber(nek);
                        double const pfc = z * pfcij * pfcik * pfcjk;
                        double const pr1 = z * pfcik * pfcjk * pdfcij / rij;
                        double const pr2 = z * pfcij * pfcjk * pdfcik / rik;
                        double const pr3 = z * pfcij * pfcik * pdfcjk / rjk;
                        double vexp = 0.0;
                        double rijs = 0.0;
                        double riks = 0.0;
                        double rjks = 0.0;

                        for (size_t l = 0; l < members.size(); ++l)
                        {
                            if (calculateExp[l])
                            {
                                rijs = rij - rs[l];
                                riks = rik - rs[l];
                                rjks = rjk - rs[l];
                                double const r2sum = rijs * rijs
                                                   + riks * riks
                                                   + rjks * rjks;
                                vexp = exp(-eta[l] * r2sum);
                            }
                            double const plambda = 1.0
                                                 + lambda[l] * costijk;
                            double fg = vexp;
                            if (plambda <= 0.0) fg = 0.0;
                            else
                            {
                                if (useIntegerPow[l])
                                {
                                    fg *= pow_int(plambda, zetaInt[l] - 1);
                                }
                                else
                                {
                                    fg *= pow(plambda, zeta[l] - 1.0);
                                }
                            }

                            // Force calculation.
                            fg *= factorNorm[l];
                            double const pfczl = pfc * zetaLambda[l];
                            double const p2etapl = plambda * factorDeriv[l];
                            double const p1 = fg * (pfczl * (rinvijik - costijk
                                            / r2ij - p2etapl * rijs / rij)
                                            + pr1 * plambda);
                            double const p2 = fg * (pfczl * (rinvijik - costijk
                                            / r2ik - p2etapl * riks / rik)
                                            + pr2 * plambda);
                            double const p3 = fg * (pfczl * (rinvijik + p2etapl
                                            * rjks / rjk) - pr3 * plambda);

                            // Save force contributions in Atom storage.
                            //
                            // SIMPLE EXPRESSIONS:
                            //size_t const li = members[l]->getIndex();
                            //atom.dGdr[li] += p1 * drij + p2 * drik;
                            //atom.neighbors[j].dGdr[li] -= p1 * drij
                            //                            + p3 * drjk;
                            //atom.neighbors[k].dGdr[li] -= p2 * drik
                            //                            - p3 * drjk;

                            /*double const p1drijx = p1 * dr1[0];
                            double const p1drijy = p1 * dr1[1];
                            double const p1drijz = p1 * dr1[2];

                            double const p2drikx = p2 * dr2[0];
                            double const p2driky = p2 * dr2[1];
                            double const p2drikz = p2 * dr2[2];

                            double const p3drjkx = p3 * dr3[0];
                            double const p3drjky = p3 * dr3[1];
                            double const p3drjkz = p3 * dr3[2];*/

                            size_t const li = memberIndex[l];
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
            } // k
        } // rij <= rc
    } // j

    return;
}

vector<string> SymmetryFunctionGroupWeightedAngular::parameterLines() const
{
    vector<string> v;

    v.push_back(strpr(getPrintFormatCommon().c_str(),
                      index + 1,
                      elementMap[ec].c_str(),
                      type,
                      rc / convLength,
                      (int)cutoffType,
                      cutoffAlpha));

    for (size_t i = 0; i < members.size(); ++i)
    {
        v.push_back(strpr(getPrintFormatMember().c_str(),
                          members[i]->getEta() * convLength * convLength,
                          members[i]->getRs() / convLength,
                          members[i]->getLambda(),
                          members[i]->getZeta(),
                          members[i]->getLineNumber(),
                          i + 1,
                          members[i]->getIndex() + 1,
                          (int)calculateExp[i]));
    }

    return v;
}
