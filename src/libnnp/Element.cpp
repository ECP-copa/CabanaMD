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

#include "Element.h"
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
    elementMap        (elementMap                    ),
    index             (index                         ),
    atomicNumber      (elementMap.atomicNumber(index)),
    atomicEnergyOffset(0.0                           ),
    symbol            (elementMap.symbol(index)      )
{
}

Element::~Element()
{

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
      estring = splitLine.at(0).c_str();
      //TODO: symbol to type map to replace hardcoded values 
      if (strcmp(estring, hstring) == 0) el = 0;
      else if (strcmp(estring, ostring) == 0) el = 1;
      SF(attype,countertotal[attype],0) = el; //ec 
      SF(attype,countertotal[attype],1) = type; //type
      
      estring = splitLine.at(2).c_str();
      if (strcmp(estring, hstring) == 0) el = 0;
      else if (strcmp(estring, ostring) == 0) el = 1;
      SF(attype,countertotal[attype],2) = el; //e1
      //set e2 to arbit high number for ease in creating groups
      SF(attype,countertotal[attype],3) = 100000; //e2
      
      SF(attype,countertotal[attype],4) = atof(splitLine.at(3).c_str())/(convLength*convLength); //eta
      SF(attype,countertotal[attype],8) = atof(splitLine.at(4).c_str())*convLength; //rs
      SF(attype,countertotal[attype],7) = atof(splitLine.at(5).c_str())*convLength; //rc
      
      SF(attype,countertotal[attype],13) = countertotal[attype]; 
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
      SF(attype,countertotal[attype],2) = e1;
      SF(attype,countertotal[attype],3) = e2;
      
      T_FLOAT zeta = SF(attype,countertotal[attype],6);
      T_INT zetaInt = round(zeta);
      if (fabs(zeta - zetaInt) <= numeric_limits<double>::min())
          SF(attype,countertotal[attype],9) = 1;
      else
          SF(attype,countertotal[attype],9) = 0;
      
      SF(attype,countertotal[attype],13) = countertotal[attype]; 
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


void Element::sortSymmetryFunctions(t_SF SF, h_t_mass h_numSymmetryFunctionsPerElement, int attype) 
{
    //print initial index
    int size = h_numSymmetryFunctionsPerElement(attype);
    int *SFvector = new int[size]; 
    //std::cout << "initial index: ";
    for (int i = 0; i < size; ++i)
    {
      SFvector[i] = i;
      //std::cout << SFvector[i]+1 << " "; 
    }
    //std::cout << std::endl; 

    //naive insertion sort
    
    int i,j,tmp;
    for (i = 1; i < size; ++i)
    {
      j = i;
      //explicit condition for sort
      while (j > 0 && compareSF(SF,attype,SFvector[j-1],SFvector[j]))
      {
        tmp = SFvector[j];
        SFvector[j] = SFvector[j-1];
        SFvector[j-1] = tmp;
        --j;
      }
    }

    std::cout << "final index: ";
    int tmpindex;
    for (int i = 0; i < size; ++i)
    {
      SF(attype,i,13) = SFvector[i];
      tmpindex = SF(attype,i,13);
      SF(attype,tmpindex,14) = i;
      std::cout << SF(attype,i,13)+1 << " "; 
    } 
    std::cout << std::endl;
    
    std::cout << "final index: ";
    for (int i = 0; i < size; ++i)
    {
      std::cout << SF(attype,i,14)+1 << " "; 
    } 
    std::cout << std::endl;
    delete [] SFvector;
    return;
}

bool Element::compareSF(t_SF SF, int attype, int index1, int index2)
{
    if (SF(attype,index2,0) < SF(attype,index1,0)) return true; //ec
    else if (SF(attype,index2,0) > SF(attype,index1,0)) return false;
    
    if (SF(attype,index2,1) < SF(attype,index1,1)) return true; //type
    else if (SF(attype,index2,1) > SF(attype,index1,1)) return false;
    
    if (SF(attype,index2,11) < SF(attype,index1,11)) return true; //cutofftype
    else if (SF(attype,index2,11) > SF(attype,index1,11)) return false;
    
    if (SF(attype,index2,12) < SF(attype,index1,12)) return true; //cutoffalpha
    else if (SF(attype,index2,12) > SF(attype,index1,12)) return false;
    
    if (SF(attype,index2,7) < SF(attype,index1,7)) return true; //rc
    else if (SF(attype,index2,7) > SF(attype,index1,7)) return false;
    
    if (SF(attype,index2,4) < SF(attype,index1,4)) return true; //eta
    else if (SF(attype,index2,4) > SF(attype,index1,4)) return false;
    
    if (SF(attype,index2,8) < SF(attype,index1,8)) return true; //rs
    else if (SF(attype,index2,8) > SF(attype,index1,8)) return false;
    
    if (SF(attype,index2,6) < SF(attype,index1,6)) return true; //zeta
    else if (SF(attype,index2,6) > SF(attype,index1,6)) return false;
    
    if (SF(attype,index2,5) < SF(attype,index1,5)) return true; //lambda
    else if (SF(attype,index2,5) > SF(attype,index1,5)) return false;

    if (SF(attype,index2,2) < SF(attype,index1,2)) return true; //e1
    else if (SF(attype,index2,2) > SF(attype,index1,2)) return false;
    
    if (SF(attype,index2,3) < SF(attype,index1,3)) return true; //e2
    else if (SF(attype,index2,3) > SF(attype,index1,3)) return false;
    
    else return false;
}

vector<string> Element::infoSymmetryFunctionParameters(t_SF SF, int attype, int (&countertotal)[2]) const
{
    vector<string> v;
    string pushstring = "";
    int index;
    float writestring;
    for (int i = 0; i < countertotal[attype]; ++i)
    {
        index = SF(attype,i,13);
        //TODO: improve function
        for (int j = 1; j < 12 ; ++j)
        {
          writestring = SF(attype,index,j);
          pushstring += to_string(writestring) + " ";
        }
        pushstring += "\n"; 
    }
    v.push_back(pushstring);

    return v;
}

vector<string> Element::infoSymmetryFunctionScaling(ScalingType scalingType, t_SF SF, t_SFscaling SFscaling, int attype, int (&countertotal)[2]) const
{
    vector<string> v;
    int index;
    for (int k = 0; k < countertotal[attype]; ++k)
    {
        index = SF(attype,k,13);
        v.push_back(scalingLine(scalingType, SFscaling, attype, index));
    }
    return v;
}

void Element::setupSymmetryFunctionGroups(t_SF SF, t_SFGmemberlist SFGmemberlist, int attype, int (&countertotal)[2], int (&countergtotal)[2])
{
    int *countergR = new int[countergtotal[attype]];
    int *countergAN = new int[countergtotal[attype]];
    int SFindex;
    for (int k = 0; k < countertotal[attype]; ++k)
    {
        bool createNewGroup = true;
        SFindex = SF(attype,k,13);
        for (int l = 0; l < countergtotal[attype]; ++l)
        {
            if (( SF(attype,SFindex,0) == SF(attype,SFGmemberlist(attype,l,0),0) ) && //same ec
                ( SF(attype,SFindex,2) == SF(attype,SFGmemberlist(attype,l,0),2) ) && //same e1
                ( SF(attype,SFindex,3) == SF(attype,SFGmemberlist(attype,l,0),3) ) && //same e2
                ( SF(attype,SFindex,7) == SF(attype,SFGmemberlist(attype,l,0),7) ) && //same rc 
                ( SF(attype,SFindex,11) == SF(attype,SFGmemberlist(attype,l,0),11) ) && //same cutoffType 
                ( SF(attype,SFindex,12) == SF(attype,SFGmemberlist(attype,l,0),12) ))    //same cutoffAlpha
            {
                createNewGroup = false;
                if (SF(attype,SFindex,1)==2)
                {
                    SFGmemberlist(attype,l,countergR[l]) = SFindex; 
                    std::cout << "Added SF " << SFindex+1 << " with global index " << SF(attype,SFindex,14) << " to group " << l+1 << " of atom type " << attype+1 << std::endl;
                    countergR[l]++;
                    break;
                }
            
                else if (SF(attype,SFindex,1)==3)
                {
                    SFGmemberlist(attype,l,countergAN[l]) = SFindex; 
                    std::cout << "Added SF " << SFindex+1 << " with global index " << SF(attype,SFindex,14) << " to group " << l+1 << " of atom type " << attype+1 << std::endl;
                    countergAN[l]++;
                    break;
                }
            }
        }
        
        if (createNewGroup)
        {
            //printf("Creating new group\n");
            int l = countergtotal[attype];
            countergtotal[attype]++;
            if (SF(attype,SFindex,1)==2)
            {
              countergR[l] = 0;
              SFGmemberlist(attype,l,countergR[l]) = SFindex; 
              std::cout << "Added SF " << SFindex+1 << " with global index " << SF(attype,SFindex,14) << " to group " << l+1 << " of atom type " << attype+1 << std::endl;
              countergR[l]++;
            }
            else if (SF(attype,SFindex,1)==3)
            {
              countergAN[l] = 0;
              SFGmemberlist(attype,l,countergAN[l]) = SFindex; 
              std::cout << "Added SF " << SFindex+1 << " with global index " << SF(attype,SFindex,14) << " to group " << l+1 << " of atom type " << attype+1 << std::endl;
              countergAN[l]++;
            }
        }
    }
            std::cout << "Members: ";
            for (size_t k = 0; k < 30; ++k) 
            {
                std::cout << SFGmemberlist(attype,0,k) << " ";
            }
            std::cout << std::endl;

    //TODO: naive insertion sort
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

vector<string> Element::infoSymmetryFunctionGroups(t_SF SF, t_SFGmemberlist SFGmemberlist, int attype, int (&countergtotal)[2]) const
{
    vector<string> v;
    string pushstring = ""; 
    for (int groupIndex = 0; groupIndex < countergtotal[attype]; ++groupIndex)
    {
        //TODO: improve function
        for (int j = 0; j < 8 ; ++j)
          pushstring += to_string(SF(attype,SFGmemberlist(attype,groupIndex,0),j)) + " ";
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
    int index;
    for (int k = 0; k < countertotal[attype]; ++k)
    {
        index = SF(attype,k,13);
        setScalingType(scalingType,statisticsLine.at(k),Smin,Smax,SF,SFscaling,attype,index);
    } 
    //TODO: groups 
    //for (int k = 0; k < countertotal[attype]; ++k)
    //    setScalingFactors(SF,attype,k);

    return;
}

size_t Element::getMinNeighbors(int attype, t_SF SF, int nSF) const
{
    //get max number of minNeighbors
    size_t global_minNeighbors = 0;
    size_t minNeighbors = 0;
    int SFtype;
    for (int k = 0; k < nSF; ++k)
    {
        SFtype = SF(attype,k,1);
        if (SFtype == 2)
            minNeighbors = 1;
        else if (SFtype == 3)
            minNeighbors = 2; 
        global_minNeighbors = max(minNeighbors, global_minNeighbors);
    }

    return global_minNeighbors;
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
