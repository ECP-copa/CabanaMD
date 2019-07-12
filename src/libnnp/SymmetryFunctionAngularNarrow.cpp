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

#include "SymmetryFunctionAngularNarrow.h"
#include "ElementMap.h"
#include "utility.h"
#include "Vec3D.h"
#include <cstdlib>   // atof, atoi
#include <cmath>     // exp, pow, cos
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

SymmetryFunctionAngularNarrow::
SymmetryFunctionAngularNarrow(ElementMap const& elementMap) :
    SymmetryFunction(3, elementMap),
    useIntegerPow(false),
    e1           (0    ),
    e2           (0    ),
    zetaInt      (0    ),
    lambda       (0.0  ),
    eta          (0.0  ),
    zeta         (0.0  ),
    rs           (0.0  )
{
    minNeighbors = 2;
    parameters.insert("e1");
    parameters.insert("e2");
    parameters.insert("eta");
    parameters.insert("zeta");
    parameters.insert("lambda");
    parameters.insert("rs");
}

bool SymmetryFunctionAngularNarrow::
operator==(SymmetryFunction const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymmetryFunctionAngularNarrow const& c =
        dynamic_cast<SymmetryFunctionAngularNarrow const&>(rhs);
    if (cutoffType  != c.cutoffType ) return false;
    if (cutoffAlpha != c.cutoffAlpha) return false;
    if (rc          != c.rc         ) return false;
    if (eta         != c.eta        ) return false;
    if (zeta        != c.zeta       ) return false;
    if (lambda      != c.lambda     ) return false;
    if (e1          != c.e1         ) return false;
    if (e2          != c.e2         ) return false;
    return true;
}

bool SymmetryFunctionAngularNarrow::
operator<(SymmetryFunction const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymmetryFunctionAngularNarrow const& c =
        dynamic_cast<SymmetryFunctionAngularNarrow const&>(rhs);
    if      (cutoffType  < c.cutoffType ) return true;
    else if (cutoffType  > c.cutoffType ) return false;
    if      (cutoffAlpha < c.cutoffAlpha) return true;
    else if (cutoffAlpha > c.cutoffAlpha) return false;
    if      (rc          < c.rc         ) return true;
    else if (rc          > c.rc         ) return false;
    if      (eta         < c.eta        ) return true;
    else if (eta         > c.eta        ) return false;
    if      (rs          < c.rs         ) return true;
    else if (rs          > c.rs         ) return false;
    if      (zeta        < c.zeta       ) return true;
    else if (zeta        > c.zeta       ) return false;
    if      (lambda      < c.lambda     ) return true;
    else if (lambda      > c.lambda     ) return false;
    if      (e1          < c.e1         ) return true;
    else if (e1          > c.e1         ) return false;
    if      (e2          < c.e2         ) return true;
    else if (e2          > c.e2         ) return false;
    return false;
}

void SymmetryFunctionAngularNarrow::
     setParameters(string const& parameterString)
{
    vector<string> splitLine = split(reduce(parameterString));

    if (type != (size_t)atoi(splitLine.at(1).c_str()))
    {
        throw runtime_error("ERROR: Incorrect symmetry function type.\n");
    }

    ec     = elementMap[splitLine.at(0)];
    e1     = elementMap[splitLine.at(2)];
    e2     = elementMap[splitLine.at(3)];
    eta    = atof(splitLine.at(4).c_str());
    lambda = atof(splitLine.at(5).c_str());
    zeta   = atof(splitLine.at(6).c_str());
    rc     = atof(splitLine.at(7).c_str());
    // Shift parameter is optional.
    if (splitLine.size() > 8)
    {
        rs = atof(splitLine.at(8).c_str());
    }

    fc.setCutoffRadius(rc);
    fc.setCutoffParameter(cutoffAlpha);

    if (e1 > e2)
    {
        size_t tmp = e1;
        e1         = e2;
        e2         = tmp;
    }

    zetaInt = round(zeta);
    if (fabs(zeta - zetaInt) <= numeric_limits<double>::min())
    {
        useIntegerPow = true;
    }
    else
    {
        useIntegerPow = false;
    }

    return;
}

void SymmetryFunctionAngularNarrow::changeLengthUnit(double convLength)
{
    this->convLength = convLength;
    eta /= convLength * convLength;
    rc *= convLength;
    rs *= convLength;

    fc.setCutoffRadius(rc);
    fc.setCutoffParameter(cutoffAlpha);

    return;
}

string SymmetryFunctionAngularNarrow::getSettingsLine() const
{
    string s = strpr("symfunction_short %2s %2zu %2s %2s %16.8E %16.8E "
                     "%16.8E %16.8E %16.8E\n",
                     elementMap[ec].c_str(),
                     type,
                     elementMap[e1].c_str(),
                     elementMap[e2].c_str(),
                     eta * convLength * convLength,
                     lambda,
                     zeta,
                     rc / convLength,
                     rs / convLength);

    return s;
}


string SymmetryFunctionAngularNarrow::parameterLine() const
{
    return strpr(getPrintFormat().c_str(),
                 index + 1,
                 elementMap[ec].c_str(),
                 type,
                 elementMap[e1].c_str(),
                 elementMap[e2].c_str(),
                 eta * convLength * convLength,
                 rs / convLength,
                 lambda,
                 zeta,
                 rc / convLength,
                 (int)cutoffType,
                 cutoffAlpha,
                 lineNumber + 1);
}

vector<string> SymmetryFunctionAngularNarrow::parameterInfo() const
{
    vector<string> v = SymmetryFunction::parameterInfo();
    string s;
    size_t w = sfinfoWidth;

    s = "e1";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), elementMap[e1].c_str()));
    s = "e2";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), elementMap[e2].c_str()));
    s = "eta";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(),
                      eta * convLength * convLength));
    s = "lambda";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), lambda));
    s = "zeta";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), zeta));
    s = "rs";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), rs / convLength));

    return v;
}

double SymmetryFunctionAngularNarrow::calculateRadialPart(
                                                         double distance) const
{
    double const& r = distance * convLength;
    double const p = exp(-eta * (r - rs) * (r - rs)) * fc.f(r);

    return p * p * p;
}

double SymmetryFunctionAngularNarrow::calculateAngularPart(double angle) const
{
    return 2.0 * pow((1.0 + lambda * cos(angle)) / 2.0, zeta);
}
