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

#include "nnp_mode.h"
#include "utility.h"
#include <algorithm> // std::min, std::max
#include <cstdlib>   // atoi, atof
#include <fstream>   // std::ifstream
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <string>

#define NNP_VERSION "2.0.0"
#define NNP_GIT_REV "7b73a36a9acfdcc80e44265bac92b055f41a1d07"
#define NNP_GIT_REV_SHORT "7b73a36"
#define NNP_GIT_BRANCH "master"

using namespace std;
using namespace nnpCbn;

Mode::Mode(bool do_print_)
             : normalize                 (false),
               checkExtrapolationWarnings(false),
               numElements               (0    ),
               maxCutoffRadius           (0.0  ),
               cutoffAlpha               (0.0  ),
               meanEnergy                (0.0  ),
               convEnergy                (1.0  ),
               convLength                (1.0  )
{
    do_print = do_print_;
}

void Mode::initialize()
{
    if(do_print)
    {
        log << "\n";
        log << "*****************************************"
               "**************************************\n";
        log << "\n";
        log << "   NNP LIBRARY v" NNP_VERSION "\n";
        log << "   ------------------\n";
        log << "\n";
        log << "Git branch  : " NNP_GIT_BRANCH "\n";
        log << "Git revision: " NNP_GIT_REV_SHORT " (" NNP_GIT_REV ")\n";
        log << "\n";
        log << "*****************************************"
               "**************************************\n";
    }
    return;
}

void Mode::loadSettingsFile(string const& fileName)
{
    if(do_print)
    {
        log << "\n";
        log << "*** SETUP: SETTINGS FILE ****************"
               "**************************************\n";
        log << "\n";
    }

    settings.loadFile(fileName);

    if(do_print)
    {
        log << settings.info();

        log << "*****************************************"
               "**************************************\n";
    }
    return;
}

void Mode::setupNormalization()
{
    if(do_print)
    {
        log << "\n";
        log << "*** SETUP: NORMALIZATION ****************"
               "**************************************\n";
        log << "\n";
    }
    if (settings.keywordExists("mean_energy") &&
        settings.keywordExists("conv_energy") &&
        settings.keywordExists("conv_length"))
    {
        normalize = true;
        meanEnergy = atof(settings["mean_energy"].c_str());
        convEnergy = atof(settings["conv_energy"].c_str());
        convLength = atof(settings["conv_length"].c_str());

        if(do_print)
        {
            log << "Data set normalization is used.\n";
            log << nnp::strpr("Mean energy per atom     : %24.16E\n", meanEnergy);
            log << nnp::strpr("Conversion factor energy : %24.16E\n", convEnergy);
            log << nnp::strpr("Conversion factor length : %24.16E\n", convLength);
        }
        if (settings.keywordExists("atom_energy"))
        {
            if(do_print)
            {
                log << "\n";
                log << "Atomic energy offsets are used in addition to"
                    " data set normalization.\n";
                log << "Offsets will be subtracted from reference energies BEFORE"
                    " normalization is applied.\n";
            }
        }
    }
    else if ((!settings.keywordExists("mean_energy")) &&
             (!settings.keywordExists("conv_energy")) &&
             (!settings.keywordExists("conv_length")))
    {
        normalize = false;
        if(do_print)
            log << "Data set normalization is not used.\n";
    }
    else
    {
        if(do_print)
            throw runtime_error("ERROR: Incorrect usage of normalization"
                                " keywords.\n"
                                "       Use all or none of \"mean_energy\", "
                                "\"conv_energy\" and \"conv_length\".\n");
    }

    if(do_print)
        log << "*****************************************"
               "**************************************\n";

    return;
}

void Mode::setupElementMap()
{
    if(do_print)
    {
        log << "\n";
        log << "*** SETUP: ELEMENT MAP ******************"
            "**************************************\n";
        log << "\n";
    }

    elementStrings = nnp::split(nnp::reduce(settings["elements"]));

    if(do_print)
    {
        log << nnp::strpr("Number of element strings found: %d\n", elementStrings.size());

        for (size_t i = 0; i < elementStrings.size(); ++i)
        {
            log << nnp::strpr("Element %2zu: %2s\n", i, elementStrings[i].c_str());
        }
    }

    //resize to match number of element types
    numElements = elementStrings.size();
    
    //setup device and host views
    //deep copy back when needed for calculations
    //setup SF info storage
    SF = t_SF("ForceNNP::SF", numElements, MAX_SF);
    SFscaling = t_SFscaling("ForceNNP::SFscaling", numElements, MAX_SF);
    SFGmemberlist = t_SFGmemberlist("ForceNNP::SFGmemberlist", numElements);

    d_SF = d_t_SF("ForceNNP::SF", numElements, MAX_SF);
    d_SFscaling = d_t_SFscaling("ForceNNP::SFscaling", numElements, MAX_SF);
    d_SFGmemberlist = d_t_SFGmemberlist("ForceNNP::SFGmemberlist", numElements);

    if(do_print)
        log << "*****************************************"
               "**************************************\n";

    return;
}

h_t_mass Mode::setupElements(h_t_mass atomicEnergyOffset)
{
    if(do_print)
    {
        log << "\n";
        log << "*** SETUP: ELEMENTS *********************"
            "**************************************\n";
        log << "\n";
    }

    numElements = (size_t)atoi(settings["number_of_elements"].c_str());
    atomicEnergyOffset = h_t_mass("ForceNNP::atomicEnergyOffset", numElements);
    if (numElements != elementStrings.size())
    {
        throw runtime_error("ERROR: Inconsistent number of elements.\n");
    }
    if(do_print)
        log << nnp::strpr("Number of elements is consistent: %zu\n", numElements);

    for (size_t i = 0; i < numElements; ++i)
    {
        elements.push_back(Element(i));
    }

    if (settings.keywordExists("atom_energy"))
    {
        nnp::Settings::KeyRange r = settings.getValues("atom_energy");
        for (nnp::Settings::KeyMap::const_iterator it = r.first; it != r.second; ++it)
        {
            vector<string> args    = nnp::split(nnp::reduce(it->second.first));
            const char* estring = args.at(0).c_str();
            //np.where element symbol == symbol encountered during parsing 
            for (size_t i = 0; i < elementStrings.size(); ++i)
            {
                if (strcmp(elementStrings[i].c_str(), estring) == 0)
                    atomicEnergyOffset(i) = atof(args.at(1).c_str());
            }
        }
    }

    if(do_print)
    {
        log << "Atomic energy offsets per element:\n";
        for (size_t i = 0; i < elementStrings.size(); ++i)
        {
            log << nnp::strpr("Element %2zu: %16.8E\n",
                              i, atomicEnergyOffset(i));
        }

        log << "Energy offsets are automatically subtracted from reference "
               "energies.\n";
        log << "*****************************************"
               "**************************************\n";
    }
    return atomicEnergyOffset;
}

void Mode::setupCutoff()
{
    if(do_print)
    {
        log << "\n";
        log << "*** SETUP: CUTOFF FUNCTIONS *************"
               "**************************************\n";
        log << "\n";
    }
    vector<string> args = nnp::split(settings["cutoff_type"]);

    cutoffType = (CutoffFunction::CutoffType) atoi(args.at(0).c_str());
    if (args.size() > 1)
    {
        cutoffAlpha = atof(args.at(1).c_str());
        if (0.0 < cutoffAlpha && cutoffAlpha >= 1.0)
        {
            throw invalid_argument("ERROR: 0 <= alpha < 1.0 is required.\n");
        }
    }

    if(do_print)
    {
        log << nnp::strpr("Parameter alpha for inner cutoff: %f\n", cutoffAlpha);
        log << "Inner cutoff = Symmetry function cutoff * alpha\n";

        log << "Equal cutoff function type for all symmetry functions:\n";

        if (cutoffType == CutoffFunction::CT_COS)
        {
            log << nnp::strpr("CutoffFunction::CT_COS (%d)\n", cutoffType);
            log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
            log << "f(x) = 1/2 * (cos(pi*x) + 1)\n";
        }
        else if (cutoffType == CutoffFunction::CT_TANHU)
        {
            log << nnp::strpr("CutoffFunction::CT_TANHU (%d)\n", cutoffType);
            log << "f(r) = tanh^3(1 - r/rc)\n";
            if (cutoffAlpha > 0.0)
            {
                log << "WARNING: Inner cutoff parameter not used in combination"
                    " with this cutoff function.\n";
            }
        }
        else if (cutoffType == CutoffFunction::CT_TANH)
        {
            log << nnp::strpr("CutoffFunction::CT_TANH (%d)\n", cutoffType);
            log << "f(r) = c * tanh^3(1 - r/rc), f(0) = 1\n";
            if (cutoffAlpha > 0.0)
            {
                log << "WARNING: Inner cutoff parameter not used in combination"
                    " with this cutoff function.\n";
            }
        }
        else if (cutoffType == CutoffFunction::CT_POLY1)
        {
            log << nnp::strpr("CutoffFunction::CT_POLY1 (%d)\n", cutoffType);
            log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
            log << "f(x) = (2x - 3)x^2 + 1\n";
        }
        else if (cutoffType == CutoffFunction::CT_POLY2)
        {
            log << nnp::strpr("CutoffFunction::CT_POLY2 (%d)\n", cutoffType);
            log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
            log << "f(x) = ((15 - 6x)x - 10)x^3 + 1\n";
        }
        else if (cutoffType == CutoffFunction::CT_POLY3)
        {
            log << nnp::strpr("CutoffFunction::CT_POLY3 (%d)\n", cutoffType);
            log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
            log << "f(x) = (x(x(20x - 70) + 84) - 35)x^4 + 1\n";
        }
        else if (cutoffType == CutoffFunction::CT_POLY4)
        {
            log << nnp::strpr("CutoffFunction::CT_POLY4 (%d)\n", cutoffType);
            log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
            log << "f(x) = (x(x((315 - 70x)x - 540) + 420) - 126)x^5 + 1\n";
        }
        else if (cutoffType == CutoffFunction::CT_EXP)
        {
            log << nnp::strpr("CutoffFunction::CT_EXP (%d)\n", cutoffType);
            log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
            log << "f(x) = exp(-1 / 1 - x^2)\n";
        }
        else if (cutoffType == CutoffFunction::CT_HARD)
        {
            log << nnp::strpr("CutoffFunction::CT_HARD (%d)\n", cutoffType);
            log << "f(r) = 1\n";
            log << "WARNING: Hard cutoff used!\n";
        }
        else
        {
            throw invalid_argument("ERROR: Unknown cutoff type.\n");
        }
        log << "*****************************************"
               "**************************************\n";
    }
    return;
}

h_t_mass Mode::setupSymmetryFunctions(h_t_mass h_numSFperElem)
{
    h_numSFperElem = h_t_mass("ForceNNP::numSymmetryFunctionsPerElement", numElements);

    if(do_print)
    {
        log << "\n";
        log << "*** SETUP: SYMMETRY FUNCTIONS ***********"
               "**************************************\n";
        log << "\n";
    }
    nnp::Settings::KeyRange r = settings.getValues("symfunction_short");
    for (nnp::Settings::KeyMap::const_iterator it = r.first; it != r.second; ++it)
    {
        vector<string> args    = nnp::split(nnp::reduce(it->second.first));
        int type = 0;
        const char* estring = args.at(0).c_str();
        //np.where element symbol == symbol encountered during parsing 
        for (size_t i = 0; i < elementStrings.size(); ++i)
        {
          if (strcmp(elementStrings[i].c_str(), estring) == 0)
             type = i; 
        }
        elements.at(type).addSymmetryFunction(it->second.first, elementStrings,
                                              type, SF, convLength, countertotal);
        //set numSFperElem to number of symmetry functions from reading
        h_numSFperElem(type) = countertotal[type];
    }

    if(do_print)
    {
        log << "Abbreviations:\n";
        log << "--------------\n";
        log << "ind .... Symmetry function index.\n";
        log << "ec ..... Central atom element.\n";
        log << "ty ..... Symmetry function type.\n";
        log << "e1 ..... Neighbor 1 element.\n";
        log << "e2 ..... Neighbor 2 element.\n";
        log << "eta .... Gaussian width eta.\n";
        log << "rs ..... Shift distance of Gaussian.\n";
        log << "la ..... Angle prefactor lambda.\n";
        log << "zeta ... Angle term exponent zeta.\n";
        log << "rc ..... Cutoff radius.\n";
        log << "ct ..... Cutoff type.\n";
        log << "ca ..... Cutoff alpha.\n";
        log << "ln ..... Line number in settings file.\n";
        log << "\n";
    }
    maxCutoffRadius = 0.0;
    
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        int attype = it->getIndex();
        it->sortSymmetryFunctions(SF, h_numSFperElem, attype);
        maxCutoffRadius = max(it->getMaxCutoffRadius(SF,attype,countertotal), maxCutoffRadius);
        it->setCutoffFunction(cutoffType, cutoffAlpha, SF, attype, countertotal);
        if(do_print)
        {
            log << nnp::strpr("Short range atomic symmetry functions element %2s :\n",
                              it->getSymbol().c_str());
            log << "-----------------------------------------"
                "--------------------------------------\n";
            log << " ind ec ty e1 e2       eta        rs la "
                "zeta        rc ct   ca    ln\n";
            log << "-----------------------------------------"
                "--------------------------------------\n";
            log << it->infoSymmetryFunctionParameters(SF, attype, countertotal);
            log << "-----------------------------------------"
                "--------------------------------------\n";
        }
    }
    minNeighbors.resize(numElements, 0);
    minCutoffRadius.resize(numElements, maxCutoffRadius);
    for (size_t i = 0; i < numElements; ++i)
    {
        int attype = elements.at(i).getIndex();
        int nSF = h_numSFperElem(attype);
        minNeighbors.at(i) = elements.at(i).getMinNeighbors(attype, SF, nSF);
        minCutoffRadius.at(i) = elements.at(i).getMinCutoffRadius(SF,attype,countertotal);

        if(do_print)
            log << nnp::strpr("Minimum cutoff radius for element %2s: %f\n",
                              elements.at(i).getSymbol().c_str(),
                              minCutoffRadius.at(i) / convLength);
    }

    if(do_print)
    {
        log << nnp::strpr("Maximum cutoff radius (global)      : %f\n",
                          maxCutoffRadius / convLength);

        log << "*****************************************"
            "**************************************\n";
    }
    return h_numSFperElem;
}

void Mode::setupSymmetryFunctionScaling(string const& fileName)
{
    if(do_print)
    {
        log << "\n";
        log << "*** SETUP: SYMMETRY FUNCTION SCALING ****"
            "**************************************\n";
        log << "\n";

        log << "Equal scaling type for all symmetry functions:\n";
    }
    if (   ( settings.keywordExists("scale_symmetry_functions" ))
           && (!settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = ST_SCALE;
        if(do_print)
        {
            log << nnp::strpr("Scaling type::ST_SCALE (%d)\n", scalingType);
            log << "Gs = Smin + (Smax - Smin) * (G - Gmin) / (Gmax - Gmin)\n";
        }
    }
    else if (   (!settings.keywordExists("scale_symmetry_functions" ))
             && ( settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = ST_CENTER;
        if(do_print)
        {
            log << nnp::strpr("Scaling type::ST_CENTER (%d)\n", scalingType);
            log << "Gs = G - Gmean\n";
        }
    }
    else if (   ( settings.keywordExists("scale_symmetry_functions" ))
             && ( settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = ST_SCALECENTER;
        if(do_print)
        {
            log << nnp::strpr("Scaling type::ST_SCALECENTER (%d)\n", scalingType);
            log << "Gs = Smin + (Smax - Smin) * (G - Gmean) / (Gmax - Gmin)\n";
        }
    }
    else if (settings.keywordExists("scale_symmetry_functions_sigma"))
    {
        scalingType = ST_SCALESIGMA;
        if(do_print)
        {
            log << nnp::strpr("Scaling type::ST_SCALESIGMA (%d)\n", scalingType);
            log << "Gs = Smin + (Smax - Smin) * (G - Gmean) / Gsigma\n";
        }
    }
    else
    {
        scalingType = ST_NONE;
        if(do_print)
        {
            log << nnp::strpr("Scaling type::ST_NONE (%d)\n", scalingType);
            log << "Gs = G\n";
            log << "WARNING: No symmetry function scaling!\n";
        }
    }

    double Smin = 0.0;
    double Smax = 0.0;
    if (scalingType == ST_SCALE ||
        scalingType == ST_SCALECENTER ||
        scalingType == ST_SCALESIGMA)
    {
        if (settings.keywordExists("scale_min_short"))
        {
            Smin = atof(settings["scale_min_short"].c_str());
        }
        else
        {
            if(do_print)
            {
                log << "WARNING: Keyword \"scale_min_short\" not found.\n";
                log << "         Default value for Smin = 0.0.\n";
            }
            Smin = 0.0;
        }

        if (settings.keywordExists("scale_max_short"))
        {
            Smax = atof(settings["scale_max_short"].c_str());
        }
        else
        {
            if(do_print)
            {
                log << "WARNING: Keyword \"scale_max_short\" not found.\n";
                log << "         Default value for Smax = 1.0.\n";
            }
            Smax = 1.0;
        }

        if(do_print)
        {
            log << nnp::strpr("Smin = %f\n", Smin);
            log << nnp::strpr("Smax = %f\n", Smax);
        }
    }

    if(do_print)
    {
        log << nnp::strpr("Symmetry function scaling statistics from file: %s\n",
                          fileName.c_str());
        log << "-----------------------------------------"
            "--------------------------------------\n";
    }

    ifstream file;
    file.open(fileName.c_str());
    if (!file.is_open())
    {
        throw runtime_error("ERROR: Could not open file: \"" + fileName
                            + "\".\n");
    }
    string line;
    vector<string> lines;
    while (getline(file, line))
    {
        if (line.at(0) != '#') lines.push_back(line);
    }
    file.close();

    if(do_print)
    {
        log << "\n";
        log << "Abbreviations:\n";
        log << "--------------\n";
        log << "ind ..... Symmetry function index.\n";
        log << "min ..... Minimum symmetry function value.\n";
        log << "max ..... Maximum symmetry function value.\n";
        log << "mean .... Mean symmetry function value.\n";
        log << "sigma ... Standard deviation of symmetry function values.\n";
        log << "sf ...... Scaling factor for derivatives.\n";
        log << "Smin .... Desired minimum scaled symmetry function value.\n";
        log << "Smax .... Desired maximum scaled symmetry function value.\n";
        log << "t ....... Scaling type.\n";
        log << "\n";
    }
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        int attype = it->getIndex();
        it->setScaling(scalingType, lines, Smin, Smax, SF, SFscaling, attype, countertotal);

        if(do_print)
        {
            log << nnp::strpr("Scaling data for symmetry functions element %2s :\n",
                              it->getSymbol().c_str());
            log << "-----------------------------------------"
                "--------------------------------------\n";
            log << " ind       min       max      mean     sigma        sf  Smin  Smax t\n";
            log << "-----------------------------------------"
                "--------------------------------------\n";
            log << it->infoSymmetryFunctionScaling(scalingType, SF, SFscaling, attype, countertotal);
            log << "-----------------------------------------"
                "--------------------------------------\n";
        }
        lines.erase(lines.begin(), lines.begin() + it->numSymmetryFunctions(attype, countertotal));
    }

    if(do_print)
        log << "*****************************************"
            "**************************************\n";

    Kokkos::deep_copy(d_SF, SF);
    Kokkos::deep_copy(d_SFscaling, SFscaling);
    Kokkos::deep_copy(d_SFGmemberlist, SFGmemberlist);

    return;
}

void Mode::setupSymmetryFunctionGroups()
{
    if(do_print)
    {
        log << "\n";
        log << "*** SETUP: SYMMETRY FUNCTION GROUPS *****"
            "**************************************\n";
        log << "\n";

        log << "Abbreviations:\n";
        log << "--------------\n";
        log << "ind .... Symmetry function group index.\n";
        log << "ec ..... Central atom element.\n";
        log << "ty ..... Symmetry function type.\n";
        log << "e1 ..... Neighbor 1 element.\n";
        log << "e2 ..... Neighbor 2 element.\n";
        log << "eta .... Gaussian width eta.\n";
        log << "rs ..... Shift distance of Gaussian.\n";
        log << "la ..... Angle prefactor lambda.\n";
        log << "zeta ... Angle term exponent zeta.\n";
        log << "rc ..... Cutoff radius.\n";
        log << "ct ..... Cutoff type.\n";
        log << "ca ..... Cutoff alpha.\n";
        log << "ln ..... Line number in settings file.\n";
        log << "mi ..... Member index.\n";
        log << "sfi .... Symmetry function index.\n";
        log << "e ...... Recalculate exponential term.\n";
        log << "\n";
    }

    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        int attype = it->getIndex();
        it->setupSymmetryFunctionGroups(SF, SFGmemberlist, attype, countertotal, countergtotal);

        if(do_print)
        {
            log << nnp::strpr("Short range atomic symmetry function groups "
                              "element %2s :\n", it->getSymbol().c_str());
            log << "-----------------------------------------"
                "--------------------------------------\n";
            log << " ind ec ty e1 e2       eta        rs la "
                "zeta        rc ct   ca    ln   mi  sfi e\n";
            log << "-----------------------------------------"
                "--------------------------------------\n";
            log << it->infoSymmetryFunctionGroups(SF, SFGmemberlist, attype, countergtotal);
            log << "-----------------------------------------"
                "--------------------------------------\n";
        }
    }

    if(do_print)
        log << "*****************************************"
            "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctionStatistics(bool collectStatistics,
                                           bool collectExtrapolationWarnings,
                                           bool writeExtrapolationWarnings,
                                           bool stopOnExtrapolationWarnings)
{
    if(do_print)
    {
        log << "\n";
        log << "*** SETUP: SYMMETRY FUNCTION STATISTICS *"
            "**************************************\n";
        log << "\n";

        log << "Equal symmetry function statistics for all elements.\n";
        log << nnp::strpr("Collect min/max/mean/sigma                        : %d\n",
                          (int)collectStatistics);
        log << nnp::strpr("Collect extrapolation warnings                    : %d\n",
                          (int)collectExtrapolationWarnings);
        log << nnp::strpr("Write extrapolation warnings immediately to stderr: %d\n",
                          (int)writeExtrapolationWarnings);
        log << nnp::strpr("Halt on any extrapolation warning                 : %d\n",
                          (int)stopOnExtrapolationWarnings);
    }

    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        //it->statistics.collectStatistics = collectStatistics;
        //it->statistics.collectExtrapolationWarnings =
        //                                          collectExtrapolationWarnings;
        //it->statistics.writeExtrapolationWarnings = writeExtrapolationWarnings;
        //it->statistics.stopOnExtrapolationWarnings =
        //                                           stopOnExtrapolationWarnings;
    }

    /*checkExtrapolationWarnings = collectStatistics
                              || collectExtrapolationWarnings
                              || writeExtrapolationWarnings
                              || stopOnExtrapolationWarnings;*/

    if(do_print)
        log << "*****************************************"
            "**************************************\n";
    return;
}

void Mode::setupNeuralNetwork()
{
    if(do_print)
    {
        log << "\n";
        log << "*** SETUP: NEURAL NETWORKS **************"
            "**************************************\n";
        log << "\n";
    }

    numLayers = 2 + atoi(settings["global_hidden_layers_short"].c_str());
    numHiddenLayers = numLayers - 2;
    
    vector<string> numNeuronsPerHiddenLayer =
        nnp::split(nnp::reduce(settings["global_nodes_short"]));
    vector<string> activationFunctions =
        nnp::split(nnp::reduce(settings["global_activation_short"]));

    for (int i = 0; i < numLayers; i++)
    {
        if (i == 0)
            AF[i] = 0;
        else if (i == numLayers - 1)
        {
            numNeuronsPerLayer[i] = 1;
            AF[i] = 0;
        } 
        else
        {
            numNeuronsPerLayer[i] = atoi(numNeuronsPerHiddenLayer.at(i-1).c_str());
            AF[i] = 1; //TODO: hardcoded atoi(activationFunctions.at(i-1));
        }
    }

    //TODO: add normalization of neurons
    bool normalizeNeurons = settings.keywordExists("normalize_nodes");

    if(do_print)
    {
        log << nnp::strpr("Normalize neurons (all elements): %d\n",
                          (int)normalizeNeurons);
        log << "-----------------------------------------"
            "--------------------------------------\n";
    }

    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        int attype = it->getIndex();
        numNeuronsPerLayer[0] = it->numSymmetryFunctions(attype, countertotal);

        if(do_print)
            log << nnp::strpr("Atomic short range NN for "
                              "element %2s :\n", it->getSymbol().c_str());

        int numWeights = 0, numBiases = 0, numConnections = 0;
        for (int j = 1; j < numLayers; ++j)
        { 
            numWeights += numNeuronsPerLayer[j-1]*numNeuronsPerLayer[j]; 
            numBiases += numNeuronsPerLayer[j];
        }
        numConnections = numWeights + numBiases;

        if(do_print)
        {
            log << nnp::strpr("Number of weights    : %6zu\n", numWeights);
            log << nnp::strpr("Number of biases     : %6zu\n", numBiases);
            log << nnp::strpr("Number of connections: %6zu\n", numConnections);
            log << nnp::strpr("Architecture    %6zu    %6zu    %6zu    %6zu\n", numNeuronsPerLayer[0], numNeuronsPerLayer[1],
                              numNeuronsPerLayer[2], numNeuronsPerLayer[3]);
            log << "-----------------------------------------"
                "--------------------------------------\n";
        }
    }

    //initialize Views
    maxNeurons = 0;
    for (int j = 0; j < numLayers; ++j)
        maxNeurons = max(maxNeurons, numNeuronsPerLayer[j]);

    h_bias = t_bias("ForceNNP::biases", numElements, numLayers, maxNeurons); 
    h_weights = t_weights("ForceNNP::weights", numElements, numLayers, maxNeurons, maxNeurons); 
    
    bias = d_t_bias("ForceNNP::biases", numElements, numLayers, maxNeurons); 
    weights = d_t_weights("ForceNNP::weights", numElements, numLayers, maxNeurons, maxNeurons); 

    if(do_print)
        log << "*****************************************"
            "**************************************\n";

    return;
}

void Mode::setupNeuralNetworkWeights(string const& fileNameFormat)
{
    if(do_print)
    {
        log << "\n";
        log << "*** SETUP: NEURAL NETWORK WEIGHTS *******"
            "**************************************\n";
        log << "\n";

        log << nnp::strpr("Weight file name format: %s\n", fileNameFormat.c_str());
    }

    int count = 0;
    int AN = 0;
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        const char* estring = elementStrings[count].c_str();
        //np.where element symbol == symbol encountered in knownElements 
        for (size_t i = 0; i < knownElements.size(); ++i)
        {
          if (strcmp(knownElements[i].c_str(), estring) == 0)
          {
            AN = i+1;
            break;
          } 
        }
        
        string fileName = nnp::strpr(fileNameFormat.c_str(), AN);

        if(do_print)
            log << nnp::strpr("Weight file for element %2s: %s\n",
                              elementStrings[count].c_str(),
                              fileName.c_str());

        ifstream file;
        file.open(fileName.c_str());
        if (!file.is_open())
        {
            throw runtime_error("ERROR: Could not open file: \"" + fileName
                                + "\".\n");
        }
        string line;
        int attype = it->getIndex();
        int layer, start, end;
        while (getline(file, line))
        {
            if (line.at(0) != '#')
            {
                vector<string> splitLine = nnp::split(nnp::reduce(line));
                if (strcmp(splitLine.at(1).c_str(), "a") == 0)
                {
                    layer = atoi(splitLine.at(3).c_str());
                    start = atoi(splitLine.at(4).c_str()) - 1;
                    end = atoi(splitLine.at(6).c_str()) - 1;
                    h_weights(attype,layer,end,start) = atof(splitLine.at(0).c_str());
                }
                else if (strcmp(splitLine.at(1).c_str(), "b") == 0)
                {
                    layer = atoi(splitLine.at(3).c_str()) - 1;
                    start = atoi(splitLine.at(4).c_str()) - 1;
                    h_bias(attype,layer,start) = atof(splitLine.at(0).c_str());
                }
            }
        }
        file.close();
        count += 1;
    }

    if(do_print)
        log << "*****************************************"
            "**************************************\n";

    Kokkos::deep_copy(bias, h_bias);
    Kokkos::deep_copy(weights, h_weights);

    return;
}

