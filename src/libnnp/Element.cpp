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

#include "Atom.h"
#include "Element.h"
#include "NeuralNetwork.h"
#include "SymmetryFunction.h"
#include "SymmetryFunctionRadial.h"
#include "SymmetryFunctionAngularNarrow.h"
#include "SymmetryFunctionAngularWide.h"
#include "SymmetryFunctionWeightedRadial.h"
#include "SymmetryFunctionWeightedAngular.h"
#include "SymmetryFunctionGroup.h"
#include "utility.h"
#include <iostream>  // std::cerr
#include <cstdlib>   // atoi
#include <algorithm> // std::sort, std::min, std::max
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

Element::Element(size_t const index, ElementMap const& elementMap) :
    neuralNetwork     (NULL                          ),
    elementMap        (elementMap                    ),
    index             (index                         ),
    atomicNumber      (elementMap.atomicNumber(index)),
    atomicEnergyOffset(0.0                           ),
    symbol            (elementMap.symbol(index)      )
{
}

Element::~Element()
{
    for (vector<SymmetryFunction*>::const_iterator
         it = symmetryFunctions.begin(); it != symmetryFunctions.end(); ++it)
    {
        delete *it;
    }

    for (vector<SymmetryFunctionGroup*>::const_iterator
         it = symmetryFunctionGroups.begin();
         it != symmetryFunctionGroups.end(); ++it)
    {
        delete *it;
    }

    if (neuralNetwork != NULL)
    {
        delete neuralNetwork;
    }
}

void Element::addSymmetryFunction(string const& parameters,
                                  size_t const& lineNumber, int attype, t_SF SF, double convLength, int (&countertotal)[2])
{
    vector<string> args = split(reduce(parameters));
    size_t         type = (size_t)atoi(args.at(1).c_str());
    const char* estring;
    const char* hstring = "H";
    const char* ostring = "O";
    int el;
    
    vector<string> splitLine = split(reduce(parameters));
    if (type == 2)
    {
      if (type != (size_t)atoi(splitLine.at(1).c_str()))
          throw runtime_error("ERROR: Incorrect symmetry function type.\n");
      
      estring = splitLine.at(0).c_str();
      if (strcmp(estring, hstring) == 0) el = 0;
      else if (strcmp(estring, ostring) == 0) el = 1;
      SF(attype,countertotal[attype],0) = el; //ec 
      SF(attype,countertotal[attype],1) = type; //type
      
      estring = splitLine.at(2).c_str();
      if (strcmp(estring, hstring) == 0) el = 0;
      else if (strcmp(estring, ostring) == 0) el = 1;
      SF(attype,countertotal[attype],2) = el; //e1
      
      SF(attype,countertotal[attype],4) = atof(splitLine.at(3).c_str())/(convLength*convLength); //eta
      SF(attype,countertotal[attype],8) = atof(splitLine.at(4).c_str())*convLength; //rs
      SF(attype,countertotal[attype],7) = atof(splitLine.at(5).c_str())*convLength; //rc
      
      countertotal[attype]++;
    }
    
    else if (type == 3)
    {
      if (type != (size_t)atoi(splitLine.at(1).c_str()))
          throw runtime_error("ERROR: Incorrect symmetry function type.\n");
      estring = splitLine.at(0).c_str();
      if (strcmp(estring, hstring) == 0) el = 0;
      else if (strcmp(estring, ostring) == 0) el = 1;
      SF(attype,countertotal[attype],0) = el; //ec
      SF(attype,countertotal[attype],1) = type; //type
      
      estring = splitLine.at(2).c_str();
      if (strcmp(estring, hstring) == 0) el = 0;
      else if (strcmp(estring, ostring) == 0) el = 1;
      SF(attype,countertotal[attype],2) = el; //e1
      
      estring = splitLine.at(3).c_str();
      if (strcmp(estring, hstring) == 0) el = 0;
      else if (strcmp(estring, ostring) == 0) el = 1;
      
      SF(attype,countertotal[attype],3) = el; //e2
      SF(attype,countertotal[attype],4) = atof(splitLine.at(4).c_str())/(convLength*convLength); //eta
      SF(attype,countertotal[attype],5) = atof(splitLine.at(5).c_str()); //lambda
      SF(attype,countertotal[attype],6) = atof(splitLine.at(6).c_str()); //zeta
      SF(attype,countertotal[attype],7) = atof(splitLine.at(7).c_str())*convLength; //rc
      // Shift parameter is optional.
      if (splitLine.size() > 8)
          SF(attype,countertotal[attype],8) = atof(splitLine.at(8).c_str())*convLength; //rs


      T_INT e1 = SF(attype,countertotal[attype],2);
      T_INT e2 = SF(attype,countertotal[attype],3);
      if (e1 > e2)
      {
          size_t tmp = e1;
          e1 = e2; 
          e2 = tmp;
      }
      T_FLOAT zeta = SF(attype,countertotal[attype],6);
      T_INT zetaInt = round(zeta);
      if (fabs(zeta - zetaInt) <= numeric_limits<double>::min())
          SF(attype,countertotal[attype],9) = 1;
      else
          SF(attype,countertotal[attype],9) = 0;
      
      countertotal[attype]++;
    }
    //TODO: Add this later
    else if (type == 9)
    {
    }
    else if (type == 12)
    {
    }
    else if (type == 13)
    {
    }
    else
    {
        throw runtime_error("ERROR: Unknown symmetry function type.\n");
    }

    return;
}


void Element::sortSymmetryFunctions()
{
    sort(symmetryFunctions.begin(),
         symmetryFunctions.end(),
         comparePointerTargets<SymmetryFunction>);

    for (size_t i = 0; i < symmetryFunctions.size(); ++i)
    {
        symmetryFunctions.at(i)->setIndex(i);
    }

    return;
}

vector<string> Element::infoSymmetryFunctionParameters(t_SF SF, int attype, int (&countertotal)[2]) const
{
    vector<string> v;
    string pushstring = "";

    for (int i = 0; i < countertotal[attype]; ++i)
    {
        //TODO: improve function
        for (int j = 0; j < 12 ; ++j)
          pushstring += to_string(SF(attype,i,j)) + " ";
        pushstring += "\n"; 
    }
    v.push_back(pushstring);

    return v;
}

vector<string> Element::infoSymmetryFunctionScaling(ScalingType scalingType, t_SFscaling SFscaling, int attype, int (&countertotal)[2]) const
{
    vector<string> v;
    for (int k = 0; k < countertotal[attype]; ++k)
        v.push_back(scalingLine(scalingType, SFscaling, attype, k));
    return v;
}

void Element::setupSymmetryFunctionGroups(t_SF SF, t_SFG SFG, t_SFGmemberlist SFGmemberlist, int attype, int (&countertotal)[2], int (&countergtotal)[2])
{
    int countergR = 0, countergAN = 0;
    for (int k = 0; k < countertotal[attype]; ++k)
    {
        bool createNewGroup = true;
        for (int l = 0; l < countergtotal[attype]; ++l)
        {
            if ((SFG(attype,l,1) == SF(attype,k,1)) && addMemberToGroup(SFG, SF, attype, l, k, countergR, countergAN))
            {
                createNewGroup = false;
                if (SFG(attype,l,1)==2)
                {
                  SFGmemberlist(attype,l,countergR) = k;
                  std::cout << "Added SF " << k+1 << " to group " << l+1 << " of atom type " << attype+1 << std::endl;
                  countergR++;
                }
                else if (SFG(attype,l,1)==3)
                {
                  SFGmemberlist(attype,l,countergAN) = k;
                  std::cout << "Added SF " << k+1 << " to group " << l+1 << " of atom type " << attype+1 << std::endl;
                  countergAN++;
                }
                break;
            }
        }
        
        if (createNewGroup)
        {
            int l = countergtotal[attype];
            countergtotal[attype]++;
            if (SF(attype,k,1)==2)
              countergR = 0;
            else if (SF(attype,k,1)==3)
              countergAN = 0;
            addMemberToGroup(SFG, SF, attype, l, k, countergR, countergAN);
            if (SFG(attype,l,1)==2)
            {
              SFGmemberlist(attype,l,countergR) = k;
              std::cout << "Added SF " << k+1 << " to group " << l+1 << " of atom type " << attype+1 << std::endl;
              countergR++;
            }
            else if (SFG(attype,l,1)==3)
            {
              SFGmemberlist(attype,l,countergAN) = k;
              std::cout << "Added SF " << k+1 << " to group " << l+1 << " of atom type " << attype+1 << std::endl;
              countergAN++;
            }
        }
    }

    /*sort(symmetryFunctionGroups.begin(),
         symmetryFunctionGroups.end(),
         comparePointerTargets<SymmetryFunctionGroup>);

    for (size_t i = 0; i < symmetryFunctionGroups.size(); ++i)
    {
        symmetryFunctionGroups.at(i)->sortMembers();
        symmetryFunctionGroups.at(i)->setIndex(i);
    }*/
    return;
}

vector<string> Element::infoSymmetryFunctionGroups(t_SFG SFG, int attype, int (&countergtotal)[2]) const
{
    vector<string> v;
    string pushstring = ""; 
    for (int i = 0; i < countergtotal[attype]; ++i)
    {
        //TODO: improve function
        for (int j = 0; j < 8 ; ++j)
          pushstring += to_string(SFG(attype,i,j)) + " ";
        pushstring += "\n";
    }
    v.push_back(pushstring);

    return v;
}

void Element::setCutoffFunction(CutoffFunction::CutoffType const cutoffType,
                                double const cutoffAlpha, t_SF SF, int attype, int (&countertotal)[2])
{
    for (int k = 0; k < countertotal[attype]; ++k)
    {
        SF(attype,k,10) = cutoffType;
        SF(attype,k,11) = cutoffAlpha;
    }
    return;
}

void Element::setScaling(ScalingType scalingType, vector<string> const& statisticsLine,
                         double Smin, double Smax, t_SF SF, t_SFscaling SFscaling, int attype, int (&countertotal)[2]) const
{
    for (int k = 0; k < countertotal[attype]; ++k)
        setScalingType(scalingType,statisticsLine.at(k),Smin,Smax,SF,SFscaling,attype,k);
   
    //TODO: groups 
    //for (int k = 0; k < countertotal[attype]; ++k)
    //    setScalingFactors(SF,attype,k);

    return;
}

size_t Element::getMinNeighbors() const
{
    size_t minNeighbors = 0;

    for (vector<SymmetryFunction*>::const_iterator
         it = symmetryFunctions.begin(); it != symmetryFunctions.end(); ++it)
    {
        minNeighbors = max((*it)->getMinNeighbors(), minNeighbors);
    }

    return minNeighbors;
}

double Element::getMinCutoffRadius(t_SF SF, int attype, int (&countertotal)[2]) const
{
    double minCutoffRadius = numeric_limits<double>::max();

    for (int k = 0; k < countertotal[attype]; ++k)
        minCutoffRadius = min(SF(attype,k,7), minCutoffRadius);

    return minCutoffRadius;
}

double Element::getMaxCutoffRadius(t_SF SF, int attype, int (&countertotal)[2]) const
{
    double maxCutoffRadius = 0.0;

    for (int k = 0; k < countertotal[attype]; ++k)
        maxCutoffRadius = max(SF(attype,k,7), maxCutoffRadius);

    return maxCutoffRadius;
}

void Element::calculateSymmetryFunctions(Atom&      atom,
                                         bool const derivatives) const
{
    for (vector<SymmetryFunction*>::const_iterator
         it = symmetryFunctions.begin();
         it != symmetryFunctions.end(); ++it)
    {
        (*it)->calculate(atom, derivatives);
    }

    return;
}


void Element::updateSymmetryFunctionStatistics(System* s, AoSoA_NNP nnp_data, T_INT atomindex)
{
    auto type = Cabana::slice<TypeNames::Types>(s->xvf);
    /*if (type(atomindex) != index)
    {
        throw runtime_error("ERROR: Atom has a different element index.\n");
    }*/

    /*if (atom.numSymmetryFunctions != symmetryFunctions.size())
    {
        throw runtime_error("ERROR: Number of symmetry functions"
                            " does not match.\n");
    }*/

    auto G = Cabana::slice<NNPNames::G>(nnp_data);
    //TODO:add functionality
    /*for (size_t i = 0; i < symmetryFunctions.size(); ++i)
    {
        double const Gmin = symmetryFunctions.at(i)->getGmin();
        double const Gmax = symmetryFunctions.at(i)->getGmax();
        double const value = symmetryFunctions.at(i)->unscale(G(atomindex, i));
        size_t const index = symmetryFunctions.at(i)->getIndex();
        if (statistics.collectStatistics)
        {
            statistics.addValue(index, value);
        }

        if (value < Gmin || value > Gmax)
        {
            if (statistics.collectExtrapolationWarnings)
            {
                statistics.addExtrapolationWarning(index,
                                                   value,
                                                   Gmin,
                                                   Gmax,
                                                   0, //TODO: indexStructure?
                                                   atomindex);
            }
            if (statistics.writeExtrapolationWarnings)
            {
                cerr << strpr("### NNP EXTRAPOLATION WARNING ### "
                              "STRUCTURE: %6zu ATOM: %6zu SYMFUNC: %4zu "
                              "VALUE: %10.3E MIN: %10.3E MAX: %10.3E\n",
                              0,
                              atomindex,
                              i,
                              value,
                              Gmin,
                              Gmax);
            }
            if (statistics.stopOnExtrapolationWarnings)
            {
                throw out_of_range(
                        strpr("### NNP EXTRAPOLATION WARNING ### "
                              "STRUCTURE: %6zu ATOM: %6zu SYMFUNC: %4zu "
                              "VALUE: %10.3E MIN: %10.3E MAX: %10.3E\n"
                              "ERROR: Symmetry function value out of range.\n",
                              0,
                              atomindex,
                              i,
                              value,
                              Gmin,
                              Gmax));
            }
        }
    }
    */

    return;
}
