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

#ifndef SYMMETRYFUNCTIONHELPER_H
#define SYMMETRYFUNCTIONHELPER_H

#include "CutoffFunction.h"
#include "ElementMap.h"
#include "utility.h"
#include <cstddef> // std::size_t
#include <map>     // std::map
#include <set>     // std::set
#include <string>  // std::string
#include <utility> // std::pair
#include <vector>  // std::vector
#include <system.h>

using namespace std;
namespace nnp
{

class SymmetryFunctionStatistics;

/** Symmetry function base class.
 *
 * Actual symmetry functions derive from this class. Provides common
 * functionality, e.g. scaling behavior.
 */
enum ScalingType
{
    /** @f$G_\text{scaled} = G@f$
     */
    ST_NONE,
    /** @f$G_\text{scaled} = S_\text{min} + \left(S_\text{max} -
     * S_\text{min}\right) \cdot \frac{G - G_\text{min}}
     * {G_\text{max} - G_\text{min}} @f$
     */
    ST_SCALE,
    /** @f$G_\text{scaled} = G - \left<G\right>@f$
     */
    ST_CENTER,
    /** @f$G_\text{scaled} = S_\text{min} + \left(S_\text{max} -
     * S_\text{min}\right) \cdot \frac{G - \left<G\right>}
     * {G_\text{max} - G_\text{min}} @f$
     */
    ST_SCALECENTER,
    /** @f$G_\text{scaled} = S_\text{min} + \left(S_\text{max} -
     * S_\text{min}\right) \cdot \frac{G - \left<G\right>}{\sigma_G} @f$
     */
    ST_SCALESIGMA
};

//CutoffFunction fc;

inline void setCF(CutoffFunction::CutoffType cutoffType, double cutoffAlpha, t_SF SF, int attype, int k)
{
    SF(attype,k,10) = cutoffType;
    SF(attype,k,11) = cutoffAlpha;
    //fc.setCutoffType(cutoffType);
    //fc.setCutoffParameter(cutoffAlpha);

    return;
}

inline void setScalingType(ScalingType scalingType, string statisticsLine, double Smin, 
                                            double Smax, t_SF SF, t_SFscaling SFscaling, int attype, int k)
{
    double Gmin, Gmax, Gmean, Gsigma, scalingFactor = 0;
    vector<string> s = split(reduce(statisticsLine));
    if (((size_t)atoi(s.at(0).c_str()) != SF(attype,k,0)+ 1) &&
        ((size_t)atoi(s.at(1).c_str()) != k + 1))
        throw runtime_error("ERROR: Inconsistent scaling statistics.\n");
    
    Gmin       = atof(s.at(2).c_str());
    Gmax       = atof(s.at(3).c_str());
    Gmean      = atof(s.at(4).c_str());
    SFscaling(attype,k,0) = Gmin;
    SFscaling(attype,k,1) = Gmax;
    SFscaling(attype,k,2) = Gmean;

    // Older versions may not supply sigma.
    if (s.size() > 5)
        Gsigma = atof(s.at(5).c_str());
        SFscaling(attype,k,3) = Gsigma;
    
    SFscaling(attype,k,4) = Smin;
    SFscaling(attype,k,5) = Smax;
    SFscaling(attype,k,7) = scalingType;

    if(scalingType == ST_NONE)
        scalingFactor = 1.0;
    else if (scalingType == ST_SCALE)
        scalingFactor = (Smax - Smin) / (Gmax - Gmin);
    else if (scalingType == ST_CENTER)
        scalingFactor = 1.0;
    else if (scalingType == ST_SCALECENTER)
        scalingFactor = (Smax - Smin) / (Gmax - Gmin);
    else if (scalingType == ST_SCALESIGMA)
        scalingFactor = (Smax - Smin) / Gsigma;
    SFscaling(attype,k,6) = scalingFactor;

    return;
}

inline string scalingLine(ScalingType scalingType, t_SFscaling SFscaling, int attype, int k)
{
    return strpr("%4zu %9.2E %9.2E %9.2E %9.2E %9.2E %5.2f %5.2f %d\n",
                 k + 1,
                 SFscaling(attype,k,0),
                 SFscaling(attype,k,1),
                 SFscaling(attype,k,2),
                 SFscaling(attype,k,3),
                 SFscaling(attype,k,6),
                 SFscaling(attype,k,4),
                 SFscaling(attype,k,5),
                 scalingType); 
}


inline bool addMemberToGroup(t_SFG SFG, t_SF SF, int attype, int groupIndex, 
    int k, int countergR, int countergAN)
{
    int type = SF(attype,k,1);
    if (type == 2)
    {
      if (countergR == 0)
      {
        SFG(attype,groupIndex,0) = SF(attype,k,0); //ec
        SFG(attype,groupIndex,1) = type; //type
        SFG(attype,groupIndex,2) = SF(attype,k,2); //e1
        SFG(attype,groupIndex,4) = k+1; //memberindex
        SFG(attype,groupIndex,5) = SF(attype,k,7); //rc
        SFG(attype,groupIndex,6) = SF(attype,k,11); //cutoffType
        SFG(attype,groupIndex,7) = SF(attype,k,12); //cutoffAlpha
        //fc.setCutoffType(SF(attype,k,11));
        //fc.setCutoffRadius(SF(attype,k,7));
        //fc.setCutoffParameter(SF(attype,k,12));
      }
      if (SF(attype,k,11) != SFG(attype,groupIndex,6)) return false;
      if (SF(attype,k,12) != SFG(attype,groupIndex,7)) return false;
      if (SF(attype,k,0) != SFG(attype,groupIndex,0)) return false;
      if (SF(attype,k,7) != SFG(attype,groupIndex,5)) return false;
      if (SF(attype,k,2) != SFG(attype,groupIndex,2)) return false;
    }
    else if (type == 3)
    {
      if (countergAN == 0)
      {
        SFG(attype,groupIndex,0) = SF(attype,k,0); //ec
        SFG(attype,groupIndex,1) = type; //type
        SFG(attype,groupIndex,2) = SF(attype,k,2); //e1
        SFG(attype,groupIndex,3) = SF(attype,k,3); //e2
        SFG(attype,groupIndex,4) = k+1; //memberindex
        SFG(attype,groupIndex,5) = SF(attype,k,7); //rc
        SFG(attype,groupIndex,6) = SF(attype,k,11); //cutoffType
        SFG(attype,groupIndex,7) = SF(attype,k,12); //cutoffAlpha
        //fc.setCutoffType(SF(attype,k,11));
        //fc.setCutoffRadius(SF(attype,k,7));
        //fc.setCutoffParameter(SF(attype,k,12));
      }
      if (SF(attype,k,11) != SFG(attype,groupIndex,6)) return false;
      if (SF(attype,k,12) != SFG(attype,groupIndex,7)) return false;
      if (SF(attype,k,0) != SFG(attype,groupIndex,0)) return false;
      if (SF(attype,k,7) != SFG(attype,groupIndex,5)) return false;
      if (SF(attype,k,2) != SFG(attype,groupIndex,2)) return false;
      if (SF(attype,k,3) != SFG(attype,groupIndex,3)) return false;
    }
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
        throw runtime_error("ERROR: Unknown symmetry function group"
                            " type.\n");
    }
    return true;
}

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

inline double unscale(int attype, double value, int k, t_SFscaling SFscaling)
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
        return (value - Smin) / scalingFactor + Gmin;
    }
    else if (scalingType == 2.0)
    {
        return value + Gmean;
    }
    else if (scalingType == 3.0)
    {
        return (value - Smin) / scalingFactor + Gmean;
    }
    else if (scalingType == 4.0)
    {
        return (value - Smin) / scalingFactor + Gmean;
    }
    else
    {
        return 0.0;
    }
}





//----------------- CLASS ---------------//

class SymmetryFunctionHelper
{
public:
    /// List of available scaling types.
    /** Set symmetry function scaling type.
     *
     * @param[in] scalingType Desired symmetry function scaling type.
     * @param[in] statisticsLine String containing symmetry function statistics
     *                           ("min max mean sigma").
     * @param[in] Smin Minimum for scaling range @f$S_\text{min}@f$.
     * @param[in] Smax Maximum for scaling range @f$S_\text{max}@f$.
     */
    /** Apply symmetry function scaling and/or centering.
     *
     * @param[in] value Raw symmetry function value.
     * @return Scaled symmetry function value.
     */
    double              scale(double value) const;
    /** Undo symmetry function scaling and/or centering.
     *
     * @param[in] value Scaled symmetry function value.
     * @return Raw symmetry function value.
     */
    __host__ __device__ double              unscale(double value) const;
    /** Get private #type member variable.
     */
    std::size_t         getType() const;
    /** Get private #index member variable.
     */
    __host__ __device__ std::size_t         getIndex() const;
    /** Get private #lineNumber member variable.
     */
    std::size_t         getLineNumber() const;
    /** Get private #ec member variable.
     */
    std::size_t         getEc() const;
    /** Get private #minNeighbors member variable.
     */
    std::size_t         getMinNeighbors() const;
    /** Get private #rc member variable.
     */
    double              getRc() const;
    /** Get private #Gmin member variable.
     */
    __host__ __device__ double              getGmin() const;
    /** Get private #Gmax member variable.
     */
    __host__ __device__ double              getGmax() const;
    /** Get private #scalingFactor member variable.
     */
    double              getScalingFactor() const;
    /** Get private #cutoffAlpha member variable.
     */
    double              getCutoffAlpha() const;
    /** Get private #convLength member variable.
     */
    double              getConvLength() const;
    /** Get private #cutoffType member variable.
     */
    CutoffFunction::
    CutoffType          getCutoffType() const;
    /** Get private #parameters member variable.
     */
    std::set<
    std::string>        getParameters() const;
    /** Set private #index member variable.
     *
     * @param[in] index Symmetry function index.
     */
    void                setIndex(std::size_t index);
    /** Set line number.
     *
     * @param[in] lineNumber Line number in settings file.
     */
    void                setLineNumber(std::size_t lineNumber);
    /** Get string with scaling information.
     *
     * @return Scaling information string.
     */

protected:
    typedef std::map<std::string,
                     std::pair<std::string, std::string> > PrintFormat;
    typedef std::vector<std::string>                       PrintOrder;
    /// Symmetry function type.
    std::size_t                type;
    /// Copy of element map.
    ElementMap                 elementMap;
    /// Symmetry function index (per element).
    std::size_t                index;
    /// Line number.
    std::size_t                lineNumber;
    /// Element index of center atom.
    std::size_t                ec;
    /// Minimum number of neighbors required.
    std::size_t                minNeighbors;
    /// Minimum for scaling range.
    double                     Smin;
    /// Maximum for scaling range.
    double                     Smax;
    /// Minimum unscaled symmetry function value.
    double                     Gmin;
    /// Maximum unscaled symmetry function value.
    double                     Gmax;
    /// Mean unscaled symmetry function value.
    double                     Gmean;
    /// Sigma of unscaled symmetry function values.
    double                     Gsigma;
    /// Cutoff radius @f$r_c@f$.
    double                     rc;
    /// Scaling factor.
    double                     scalingFactor;
    /// Cutoff parameter @f$\alpha@f$.
    double                     cutoffAlpha;
    /// Data set normalization length conversion factor.
    double                     convLength;
    /// Cutoff function used by this symmetry function.
    /// Cutoff type used by this symmetry function.
    CutoffFunction::CutoffType cutoffType;
    /// Symmetry function scaling type used by this symmetry function.
    ScalingType                scalingType;
    /// Set with symmetry function parameter IDs (lookup for printing).
    std::set<std::string>      parameters;
    /// Width of the SFINFO parameter description field (see #parameterInfo()).
    static std::size_t const   sfinfoWidth;
    /// Map of parameter format strings and empty strings.
    static PrintFormat const   printFormat;
    /// Vector of parameters in order of printing.
    static PrintOrder const    printOrder;
    std::vector<std::string> parameterInfo() const;

    /** Constructor, initializes #type.
     */
    SymmetryFunctionHelper(std::size_t type, ElementMap const&);
    /** Initialize static print format map for all possible parameters.
     */
    static PrintFormat const initializePrintFormat();
    /** Initialize static print order vector for all possible parameters.
     */
    static PrintOrder const  initializePrintOrder();
    /** Generate format string for symmetry function parameter printing.
     *
     * @return C-Style format string.
     */
    std::string              getPrintFormat() const;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline std::size_t SymmetryFunctionHelper::getType() const
{
    return type;
}

inline std::size_t SymmetryFunctionHelper::getEc() const
{
    return ec;
}

__host__ __device__ std::size_t SymmetryFunctionHelper::getIndex() const
{
    return index;
}

inline std::size_t SymmetryFunctionHelper::getLineNumber() const
{
    return lineNumber;
}

inline std::size_t SymmetryFunctionHelper::getMinNeighbors() const
{
    return minNeighbors;
}

inline double SymmetryFunctionHelper::getRc() const
{
    return rc;
}

__host__ __device__ double SymmetryFunctionHelper::getGmin() const
{
    return Gmin;
}

__host__ __device__ double SymmetryFunctionHelper::getGmax() const
{
    return Gmax;
}

inline double SymmetryFunctionHelper::getScalingFactor() const
{
    return scalingFactor;
}

inline double SymmetryFunctionHelper::getCutoffAlpha() const
{
    return cutoffAlpha;
}

inline double SymmetryFunctionHelper::getConvLength() const
{
    return convLength;
}

inline void SymmetryFunctionHelper::setIndex(std::size_t index)
{
    this->index = index;
    return;
}

inline void SymmetryFunctionHelper::setLineNumber(std::size_t lineNumber)
{
    this->lineNumber = lineNumber;
    return;
}

inline CutoffFunction::CutoffType SymmetryFunctionHelper::getCutoffType() const
{
    return cutoffType;
}

inline std::set<std::string> SymmetryFunctionHelper::getParameters() const
{
    return parameters;
}

}


#endif
