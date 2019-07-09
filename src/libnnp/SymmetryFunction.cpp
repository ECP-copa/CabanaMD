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
#include "SymmetryFunction.h"
#include "SymmetryFunctionStatistics.h"
#include "utility.h"
#include <cstdlib>   // atof, atoi
#include <stdexcept> // std::runtime_error, std::out_of_range
#include <iostream>  // std::cerr

using namespace std;
using namespace nnp;

size_t const SymmetryFunction::sfinfoWidth = 12;

SymmetryFunction::PrintFormat const
SymmetryFunction::printFormat = initializePrintFormat();

SymmetryFunction::PrintOrder const
SymmetryFunction::printOrder = initializePrintOrder();

vector<string> SymmetryFunction::parameterInfo() const
{
    vector<string> v;
    string s;
    size_t w = sfinfoWidth;

    s = "lineNumber";
    v.push_back(strpr((pad(s, w) + "%zu"   ).c_str(), lineNumber + 1));
    s = "index";
    v.push_back(strpr((pad(s, w) + "%zu"   ).c_str(), index + 1));
    s = "type";
    v.push_back(strpr((pad(s, w) + "%zu"   ).c_str(), type));
    s = "ec";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), elementMap[ec].c_str()));
    s = "rc";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), rc / convLength));
    s = "cutoffType";
    v.push_back(strpr((pad(s, w) + "%d"    ).c_str(), (int)cutoffType));
    s = "cutoffAlpha";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), cutoffAlpha));

    return v;
}

void SymmetryFunction::setCutoffFunction(CutoffFunction::
                                         CutoffType cutoffType,
                                         double     cutoffAlpha)
{
    this->cutoffType = cutoffType;
    this->cutoffAlpha = cutoffAlpha;
    fc.setCutoffType(cutoffType);
    fc.setCutoffParameter(cutoffAlpha);

    return;
}

SymmetryFunction::SymmetryFunction(size_t type, ElementMap const& elementMap) :
    type         (type                   ),
    elementMap   (elementMap             ),
    index        (0                      ),
    ec           (0                      ),
    minNeighbors (0                      ),
    Smin         (0.0                    ),
    Smax         (0.0                    ),
    Gmin         (0.0                    ),
    Gmax         (0.0                    ),
    Gmean        (0.0                    ),
    Gsigma       (0.0                    ),
    rc           (0.0                    ),
    scalingFactor(1.0                    ),
    cutoffAlpha  (0.0                    ),
    convLength   (1.0                    ),
    cutoffType   (CutoffFunction::CT_HARD),
    scalingType  (ST_NONE                )
{
    // Add standard parameter IDs to set.
    parameters.insert("index");
    parameters.insert("ec");
    parameters.insert("type");
    parameters.insert("rc");
    parameters.insert("cutoffType");
    parameters.insert("cutoffAlpha");
    parameters.insert("lineNumber");
}

double SymmetryFunction::scale(double value) const
{
    if (scalingType == ST_NONE)
    {
        return value;
    }
    else if (scalingType == ST_SCALE)
    {
        return Smin + scalingFactor * (value - Gmin);
    }
    else if (scalingType == ST_CENTER)
    {
        return value - Gmean;
    }
    else if (scalingType == ST_SCALECENTER)
    {
        return Smin + scalingFactor * (value - Gmean);
    }
    else if (scalingType == ST_SCALESIGMA)
    {
        return Smin + scalingFactor * (value - Gmean);
    }
    else
    {
        return 0.0;
    }
}


SymmetryFunction::PrintFormat const SymmetryFunction::initializePrintFormat()
{
    PrintFormat pf;

    pf["index"]       = make_pair("%4zu" , string(4, ' '));
    pf["ec"]          = make_pair("%2s"  , string(2, ' '));
    pf["type"]        = make_pair("%2zu" , string(2, ' '));
    pf["e1"]          = make_pair("%2s"  , string(2, ' '));
    pf["e2"]          = make_pair("%2s"  , string(2, ' '));
    pf["eta"]         = make_pair("%9.3E", string(9, ' '));
    pf["rs"]          = make_pair("%9.3E", string(9, ' '));
    pf["lambda"]      = make_pair("%2.0f", string(2, ' '));
    pf["zeta"]        = make_pair("%4.1f", string(4, ' '));
    pf["rc"]          = make_pair("%9.3E", string(9, ' '));
    pf["cutoffType"]  = make_pair("%2d"  , string(2, ' '));
    pf["cutoffAlpha"] = make_pair("%4.2f", string(4, ' '));
    pf["lineNumber"]  = make_pair("%5zu" , string(5, ' '));

    return pf;
}

SymmetryFunction::PrintOrder const SymmetryFunction::initializePrintOrder()
{
    vector<string> po;

    po.push_back("index"      );
    po.push_back("ec"         );
    po.push_back("type"       );
    po.push_back("e1"         );
    po.push_back("e2"         );
    po.push_back("eta"        );
    po.push_back("rs"         );
    po.push_back("lambda"     );
    po.push_back("zeta"       );
    po.push_back("rc"         );
    po.push_back("cutoffType" );
    po.push_back("cutoffAlpha");
    po.push_back("lineNumber" );

    return po;
}

string SymmetryFunction::getPrintFormat() const
{
    string s;

    for (PrintOrder::const_iterator it = printOrder.begin();
         it != printOrder.end(); ++it)
    {
        // If parameter is present add format string.
        if (parameters.find(*it) != parameters.end())
        {
            s += safeFind(printFormat, (*it)).first + ' ';
        }
        // Else add just enough empty spaces.
        else 
        {
            s += safeFind(printFormat, (*it)).second + ' ';
        }
    }
    // Remove extra space at the end.
    if (s.size () > 0)  s.resize (s.size () - 1);
    s += '\n';

    return s;
}

