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

#include "Mode.h"
#include "utility.h"
#include "version.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm> // std::min, std::max
#include <cstdlib>   // atoi, atof
#include <fstream>   // std::ifstream
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <iostream> //TODO: remove this 
#include <string>
#define MAX_SF 30

using namespace std;
using namespace nnp;

Mode::Mode() : normalize                 (false),
               checkExtrapolationWarnings(false),
               numElements               (0    ),
               maxCutoffRadius           (0.0  ),
               cutoffAlpha               (0.0  ),
               meanEnergy                (0.0  ),
               convEnergy                (1.0  ),
               convLength                (1.0  )
{
}

void Mode::initialize()
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
#ifdef _OPENMP
    log << strpr("Number of OpenMP threads: %d", omp_get_max_threads());
    log << "\n";
#endif
    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::loadSettingsFile(string const& fileName)
{
    log << "\n";
    log << "*** SETUP: SETTINGS FILE ****************"
           "**************************************\n";
    log << "\n";

    settings.loadFile(fileName);
    log << settings.info();

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupGeneric(t_mass numSymmetryFunctionsPerElement)
{
    setupNormalization();
    numSymmetryFunctionsPerElement = setupElementMap(numSymmetryFunctionsPerElement);
    setupElements();
    setupCutoff();
    numSymmetryFunctionsPerElement = setupSymmetryFunctions(numSymmetryFunctionsPerElement);
#ifndef NOSFGROUPS
    setupSymmetryFunctionGroups();
#endif
    setupNeuralNetwork();

    return;
}

void Mode::setupNormalization()
{
    log << "\n";
    log << "*** SETUP: NORMALIZATION ****************"
           "**************************************\n";
    log << "\n";

    if (settings.keywordExists("mean_energy") &&
        settings.keywordExists("conv_energy") &&
        settings.keywordExists("conv_length"))
    {
        normalize = true;
        meanEnergy = atof(settings["mean_energy"].c_str());
        convEnergy = atof(settings["conv_energy"].c_str());
        convLength = atof(settings["conv_length"].c_str());
        log << "Data set normalization is used.\n";
        log << strpr("Mean energy per atom     : %24.16E\n", meanEnergy);
        log << strpr("Conversion factor energy : %24.16E\n", convEnergy);
        log << strpr("Conversion factor length : %24.16E\n", convLength);
        if (settings.keywordExists("atom_energy"))
        {
            log << "\n";
            log << "Atomic energy offsets are used in addition to"
                   " data set normalization.\n";
            log << "Offsets will be subtracted from reference energies BEFORE"
                   " normalization is applied.\n";
        }
    }
    else if ((!settings.keywordExists("mean_energy")) &&
             (!settings.keywordExists("conv_energy")) &&
             (!settings.keywordExists("conv_length")))
    {
        normalize = false;
        log << "Data set normalization is not used.\n";
    }
    else
    {
        throw runtime_error("ERROR: Incorrect usage of normalization"
                            " keywords.\n"
                            "       Use all or none of \"mean_energy\", "
                            "\"conv_energy\" and \"conv_length\".\n");
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

t_mass Mode::setupElementMap(t_mass d_numSymmetryFunctionsPerElement)
{
    log << "\n";
    log << "*** SETUP: ELEMENT MAP ******************"
           "**************************************\n";
    log << "\n";

    elementMap.registerElements(settings["elements"]);
    log << strpr("Number of element strings found: %d\n", elementMap.size());
    for (size_t i = 0; i < elementMap.size(); ++i)
    {
        log << strpr("Element %2zu: %2s (%3zu)\n", i, elementMap[i].c_str(),
                     elementMap.atomicNumber(i));
    }
    //resize numSymmetryFunctionsPerElement to have size = num of atom types in system
    numElements = elementMap.size();
    
    //setup device and host views
    //deep copy back when needed for calculations

    d_numSymmetryFunctionsPerElement = t_mass("ForceNNP::numSymmetryFunctionsPerElement", numElements);
   
    //setup SF info storage
    SF = t_SF("ForceNNP::SF", numElements, MAX_SF);
    SFscaling = t_SFscaling("ForceNNP::SFscaling", numElements, MAX_SF);
    SFGmemberlist = t_SFGmemberlist("ForceNNP::SFGmemberlist", numElements);
    
    d_SF = d_t_SF("ForceNNP::SF", numElements, MAX_SF);
    d_SFscaling = d_t_SFscaling("ForceNNP::SFscaling", numElements, MAX_SF);
    d_SFGmemberlist = d_t_SFGmemberlist("ForceNNP::SFGmemberlist", numElements);
    
    log << "*****************************************"
           "**************************************\n";

    return d_numSymmetryFunctionsPerElement;
}

void Mode::setupElements()
{
    log << "\n";
    log << "*** SETUP: ELEMENTS *********************"
           "**************************************\n";
    log << "\n";

    numElements = (size_t)atoi(settings["number_of_elements"].c_str());
    if (numElements != elementMap.size())
    {
        throw runtime_error("ERROR: Inconsistent number of elements.\n");
    }
    log << strpr("Number of elements is consistent: %zu\n", numElements);

    for (size_t i = 0; i < numElements; ++i)
    {
        elements.push_back(Element(i, elementMap));
    }

    if (settings.keywordExists("atom_energy"))
    {
       log << "atom_energy not supported for now\n";
       /* Settings::KeyRange r = settings.getValues("atom_energy");
        for (Settings::KeyMap::const_iterator it = r.first;
             it != r.second; ++it)
        {
            vector<string> args    = split(reduce(it->second.first));
            size_t         element = elementMap[args.at(0)];
            elements.at(element).
                setAtomicEnergyOffset(atof(args.at(1).c_str()));
        }*/
    }
    /*
    log << "Atomic energy offsets per element:\n";
    for (size_t i = 0; i < elementMap.size(); ++i)
    {
        log << strpr("Element %2zu: %16.8E\n",
                     i, elements.at(i).getAtomicEnergyOffset());
    }

    log << "Energy offsets are automatically subtracted from reference "
           "energies.\n";
    log << "*****************************************"
           "**************************************\n";
    numAtomsPerElement.resize(numElements, 0);
    TODO: Add back support for offsets */ 
    return;
}

void Mode::setupCutoff()
{
    log << "\n";
    log << "*** SETUP: CUTOFF FUNCTIONS *************"
           "**************************************\n";
    log << "\n";

    vector<string> args = split(settings["cutoff_type"]);

    cutoffType = (CutoffFunction::CutoffType) atoi(args.at(0).c_str());
    if (args.size() > 1)
    {
        cutoffAlpha = atof(args.at(1).c_str());
        if (0.0 < cutoffAlpha && cutoffAlpha >= 1.0)
        {
            throw invalid_argument("ERROR: 0 <= alpha < 1.0 is required.\n");
        }
    }
    log << strpr("Parameter alpha for inner cutoff: %f\n", cutoffAlpha);
    log << "Inner cutoff = Symmetry function cutoff * alpha\n";

    log << "Equal cutoff function type for all symmetry functions:\n";
    if (cutoffType == CutoffFunction::CT_COS)
    {
        log << strpr("CutoffFunction::CT_COS (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = 1/2 * (cos(pi*x) + 1)\n";
    }
    else if (cutoffType == CutoffFunction::CT_TANHU)
    {
        log << strpr("CutoffFunction::CT_TANHU (%d)\n", cutoffType);
        log << "f(r) = tanh^3(1 - r/rc)\n";
        if (cutoffAlpha > 0.0)
        {
            log << "WARNING: Inner cutoff parameter not used in combination"
                   " with this cutoff function.\n";
        }
    }
    else if (cutoffType == CutoffFunction::CT_TANH)
    {
        log << strpr("CutoffFunction::CT_TANH (%d)\n", cutoffType);
        log << "f(r) = c * tanh^3(1 - r/rc), f(0) = 1\n";
        if (cutoffAlpha > 0.0)
        {
            log << "WARNING: Inner cutoff parameter not used in combination"
                   " with this cutoff function.\n";
        }
    }
    else if (cutoffType == CutoffFunction::CT_POLY1)
    {
        log << strpr("CutoffFunction::CT_POLY1 (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = (2x - 3)x^2 + 1\n";
    }
    else if (cutoffType == CutoffFunction::CT_POLY2)
    {
        log << strpr("CutoffFunction::CT_POLY2 (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = ((15 - 6x)x - 10)x^3 + 1\n";
    }
    else if (cutoffType == CutoffFunction::CT_POLY3)
    {
        log << strpr("CutoffFunction::CT_POLY3 (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = (x(x(20x - 70) + 84) - 35)x^4 + 1\n";
    }
    else if (cutoffType == CutoffFunction::CT_POLY4)
    {
        log << strpr("CutoffFunction::CT_POLY4 (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = (x(x((315 - 70x)x - 540) + 420) - 126)x^5 + 1\n";
    }
    else if (cutoffType == CutoffFunction::CT_EXP)
    {
        log << strpr("CutoffFunction::CT_EXP (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = exp(-1 / 1 - x^2)\n";
    }
    else if (cutoffType == CutoffFunction::CT_HARD)
    {
        log << strpr("CutoffFunction::CT_HARD (%d)\n", cutoffType);
        log << "f(r) = 1\n";
        log << "WARNING: Hard cutoff used!\n";
    }
    else
    {
        throw invalid_argument("ERROR: Unknown cutoff type.\n");
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

t_mass Mode::setupSymmetryFunctions(t_mass numSymmetryFunctionsPerElement)
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTIONS ***********"
           "**************************************\n";
    log << "\n";

    Settings::KeyRange r = settings.getValues("symfunction_short");
    for (Settings::KeyMap::const_iterator it = r.first; it != r.second; ++it)
    {
        vector<string> args    = split(reduce(it->second.first));
        size_t         element = elementMap[args.at(0)];
        int type;
        const char* estring = args.at(0).c_str();
        const char* hstring = "H";
        const char* ostring = "O";
        //TODO: hardcoded symbol to type conversions
        if (strcmp(estring, hstring) == 0)
          type = 0;
        else if (strcmp(estring, ostring) == 0)
          type = 1;
        //type = atoi(args.at(0).c_str());
        elements.at(element).addSymmetryFunction(it->second.first,
                                                 it->second.second, type, SF, convLength, countertotal);
    }

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
    maxCutoffRadius = 0.0;
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        int attype = it->getIndex();
        //it->sortSymmetryFunctions(t_SF SF);
        maxCutoffRadius = max(it->getMaxCutoffRadius(SF,attype,countertotal), maxCutoffRadius);
        it->setCutoffFunction(cutoffType, cutoffAlpha, SF, attype, countertotal);
        log << strpr("Short range atomic symmetry functions element %2s :\n",
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
        //set numSymmetryFunctionsPerElement to have number of symmetry functions detected after reading

        t_mass::HostMirror h_numSymmetryFunctionsPerElement = Kokkos::create_mirror_view(numSymmetryFunctionsPerElement);
        Kokkos::deep_copy(h_numSymmetryFunctionsPerElement, numSymmetryFunctionsPerElement);
        h_numSymmetryFunctionsPerElement(attype) = it->numSymmetryFunctions(attype, countertotal);
    }
    minNeighbors.resize(numElements, 0);
    minCutoffRadius.resize(numElements, maxCutoffRadius);
    for (size_t i = 0; i < numElements; ++i)
    {
        int attype = elements.at(i).getIndex();
        minNeighbors.at(i) = elements.at(i).getMinNeighbors();
        minCutoffRadius.at(i) = elements.at(i).getMinCutoffRadius(SF,attype,countertotal);
        log << strpr("Minimum cutoff radius for element %2s: %f\n",
                     elements.at(i).getSymbol().c_str(),
                     minCutoffRadius.at(i) / convLength);
    }
    log << strpr("Maximum cutoff radius (global)      : %f\n",
                 maxCutoffRadius / convLength);

    log << "*****************************************"
           "**************************************\n";

    return numSymmetryFunctionsPerElement;
}

void Mode::setupSymmetryFunctionScaling(string const& fileName)
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTION SCALING ****"
           "**************************************\n";
    log << "\n";

    log << "Equal scaling type for all symmetry functions:\n";
    if (   ( settings.keywordExists("scale_symmetry_functions" ))
        && (!settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = ST_SCALE;
        log << strpr("Scaling type::ST_SCALE (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmin) / (Gmax - Gmin)\n";
    }
    else if (   (!settings.keywordExists("scale_symmetry_functions" ))
             && ( settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = ST_CENTER;
        log << strpr("Scaling type::ST_CENTER (%d)\n", scalingType);
        log << "Gs = G - Gmean\n";
    }
    else if (   ( settings.keywordExists("scale_symmetry_functions" ))
             && ( settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = ST_SCALECENTER;
        log << strpr("Scaling type::ST_SCALECENTER (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmean) / (Gmax - Gmin)\n";
    }
    else if (settings.keywordExists("scale_symmetry_functions_sigma"))
    {
        scalingType = ST_SCALESIGMA;
        log << strpr("Scaling type::ST_SCALESIGMA (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmean) / Gsigma\n";
    }
    else
    {
        scalingType = ST_NONE;
        log << strpr("Scaling type::ST_NONE (%d)\n", scalingType);
        log << "Gs = G\n";
        log << "WARNING: No symmetry function scaling!\n";
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
            log << "WARNING: Keyword \"scale_min_short\" not found.\n";
            log << "         Default value for Smin = 0.0.\n";
            Smin = 0.0;
        }

        if (settings.keywordExists("scale_max_short"))
        {
            Smax = atof(settings["scale_max_short"].c_str());
        }
        else
        {
            log << "WARNING: Keyword \"scale_max_short\" not found.\n";
            log << "         Default value for Smax = 1.0.\n";
            Smax = 1.0;
        }

        log << strpr("Smin = %f\n", Smin);
        log << strpr("Smax = %f\n", Smax);
    }

    log << strpr("Symmetry function scaling statistics from file: %s\n",
                 fileName.c_str());
    log << "-----------------------------------------"
           "--------------------------------------\n";
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
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        int attype = it->getIndex();
        it->setScaling(scalingType, lines, Smin, Smax, SF, SFscaling, attype, countertotal);
        log << strpr("Scaling data for symmetry functions element %2s :\n",
                     it->getSymbol().c_str());
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << " ind       min       max      mean     sigma        sf  Smin  Smax t\n";
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << it->infoSymmetryFunctionScaling(scalingType, SFscaling, attype, countertotal);
        log << "-----------------------------------------"
               "--------------------------------------\n";
        lines.erase(lines.begin(), lines.begin() + it->numSymmetryFunctions(attype, countertotal));
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctionGroups()
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
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        int attype = it->getIndex();
        it->setupSymmetryFunctionGroups(SF, SFGmemberlist, attype, countertotal, countergtotal);
        log << strpr("Short range atomic symmetry function groups "
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

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctionStatistics(bool collectStatistics,
                                           bool collectExtrapolationWarnings,
                                           bool writeExtrapolationWarnings,
                                           bool stopOnExtrapolationWarnings)
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTION STATISTICS *"
           "**************************************\n";
    log << "\n";

    log << "Equal symmetry function statistics for all elements.\n";
    log << strpr("Collect min/max/mean/sigma                        : %d\n",
                 (int)collectStatistics);
    log << strpr("Collect extrapolation warnings                    : %d\n",
                 (int)collectExtrapolationWarnings);
    log << strpr("Write extrapolation warnings immediately to stderr: %d\n",
                 (int)writeExtrapolationWarnings);
    log << strpr("Halt on any extrapolation warning                 : %d\n",
                 (int)stopOnExtrapolationWarnings);
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->statistics.collectStatistics = collectStatistics;
        it->statistics.collectExtrapolationWarnings =
                                                  collectExtrapolationWarnings;
        it->statistics.writeExtrapolationWarnings = writeExtrapolationWarnings;
        it->statistics.stopOnExtrapolationWarnings =
                                                   stopOnExtrapolationWarnings;
    }

    checkExtrapolationWarnings = collectStatistics
                              || collectExtrapolationWarnings
                              || writeExtrapolationWarnings
                              || stopOnExtrapolationWarnings;

    log << "*****************************************"
           "**************************************\n";
    return;
}

void Mode::setupNeuralNetwork()
{
    log << "\n";
    log << "*** SETUP: NEURAL NETWORKS **************"
           "**************************************\n";
    log << "\n";

    numLayers = 2 + atoi(settings["global_hidden_layers_short"].c_str());
    numHiddenLayers = numLayers - 2;
    
    vector<string> numNeuronsPerHiddenLayer =
        split(reduce(settings["global_nodes_short"]));
    vector<string> activationFunctions =
        split(reduce(settings["global_activation_short"]));

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
            AF[i-1] = 1; //TODO: hardcoded atoi(activationFunctions.at(i-1));
        }
    }

    //TODO: add normalization of neurons
    bool normalizeNeurons = settings.keywordExists("normalize_nodes");
    log << strpr("Normalize neurons (all elements): %d\n",
                 (int)normalizeNeurons);
    log << "-----------------------------------------"
           "--------------------------------------\n";
    
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        int attype = it->getIndex();
        numNeuronsPerLayer[0] = it->numSymmetryFunctions(attype, countertotal);
        log << strpr("Atomic short range NN for "
                     "element %2s :\n", it->getSymbol().c_str());

        int numWeights = 0, numBiases = 0, numConnections = 0;
        for (int j = 1; j < numLayers; ++j)
        { 
            numWeights += numNeuronsPerLayer[j-1]*numNeuronsPerLayer[j]; 
            numBiases += numNeuronsPerLayer[j];
        }
        numConnections = numWeights + numBiases;
        log << strpr("Number of weights    : %6zu\n", numWeights);
        log << strpr("Number of biases     : %6zu\n", numBiases);
        log << strpr("Number of connections: %6zu\n", numConnections);
        log << strpr("Architecture    %6zu    %6zu    %6zu    %6zu\n", numNeuronsPerLayer[0], numNeuronsPerLayer[1],
            numNeuronsPerLayer[2], numNeuronsPerLayer[3]);
        log << "-----------------------------------------"
               "--------------------------------------\n";
    }

    //initialize Views
    int maxNeurons = 0;
    for (int j = 0; j < numLayers; ++j)
        maxNeurons = max(maxNeurons, numNeuronsPerLayer[j]);

    h_bias = t_bias("ForceNNP::biases", numElements, numLayers, maxNeurons); 
    h_weights = t_weights("ForceNNP::weights", numElements, numLayers, maxNeurons, maxNeurons); 
    
    bias = d_t_bias("ForceNNP::biases", numElements, numLayers, maxNeurons); 
    weights = d_t_weights("ForceNNP::weights", numElements, numLayers, maxNeurons, maxNeurons); 
    NN = d_t_NN("ForceNNP::NN", numLayers, maxNeurons); 
    dfdx = d_t_NN("ForceNNP::dfdx", numLayers, maxNeurons); 

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupNeuralNetworkWeights(string const& fileNameFormat)
{
    log << "\n";
    log << "*** SETUP: NEURAL NETWORK WEIGHTS *******"
           "**************************************\n";
    log << "\n";

    log << strpr("Weight file name format: %s\n", fileNameFormat.c_str());
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        string fileName = strpr(fileNameFormat.c_str(), it->getAtomicNumber());
        log << strpr("Weight file for element %2s: %s\n",
                     it->getSymbol().c_str(),
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
                vector<string> splitLine = split(reduce(line));
                if (strcmp(splitLine.at(1).c_str(), "a") == 0)
                {
                    layer = atoi(splitLine.at(3).c_str());
                    start = atoi(splitLine.at(4).c_str()) - 1;
                    end = atoi(splitLine.at(6).c_str()) - 1;
                    h_weights(attype,layer,start,end) = atof(splitLine.at(0).c_str());
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
    }
    log << "*****************************************"
           "**************************************\n";
    return;
}

void Mode::resetExtrapolationWarnings()
{
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->statistics.resetExtrapolationWarnings();
    }

    return;
}

size_t Mode::getNumExtrapolationWarnings() const
{
    size_t numExtrapolationWarnings = 0;

    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        numExtrapolationWarnings +=
            it->statistics.countExtrapolationWarnings();
    }

    return numExtrapolationWarnings;
}


bool Mode::settingsKeywordExists(std::string const& keyword) const
{
    return settings.keywordExists(keyword);
}

string Mode::settingsGetValue(std::string const& keyword) const
{
    return settings.getValue(keyword);
}


void Mode::writeSettingsFile(ofstream* const& file) const
{
    settings.writeSettingsFile(file);

    return;
}



















//------------------- MEGA FUNCTION ATTEMPT --------------//

  //mode->calculateSymmetryFunctionGroups(s, nnp_data, neigh_list, true);
  //mode->calculateAtomicNeuralNetworks(s, nnp_data, true);
  //mode->calculateForces(s, numSymmetryFunctionsPerElement, nnp_data, neigh_list);
  //
//

void Mode::calculateSymmetryFunctionGroups(System *s, AoSoA_NNP nnp_data, t_verletlist_full_2D neigh_list,
    t_mass numSymmetryFunctionsPerElement) 
{
    auto id = Cabana::slice<IDs>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto x = Cabana::slice<Positions>(s->xvf);
    auto G = Cabana::slice<NNPNames::G>(nnp_data);
    dGdr = t_dGdr("ForceNNP::dGdr", s->N_local+s->N_ghost);
    
    //deep copy into device Views
    Kokkos::deep_copy(d_SF, SF);
    Kokkos::deep_copy(d_SFscaling, SFscaling);
    Kokkos::deep_copy(d_SFGmemberlist, SFGmemberlist);

    std::cout << "Calculating SF groups" << std::endl;
    // Calculate symmetry functions (and derivatives).
    //calculateSFGroups(s, nnp_data, SF, SFscaling, SFGmemberlist, attype, neigh_list, i, countergtotal); 
    Kokkos::parallel_for ("Mode::calculateSymmetryFunctionGroups", s->N_local, KOKKOS_LAMBDA (const size_t i) 
    {
        int attype = type(i);
        T_INT numSymmetryFunctions = numSymmetryFunctionsPerElement(type(i));
        
        // Check if atom has low number of neighbors.
        int num_neighs = Cabana::NeighborList<t_verletlist_full_2D>::numNeighbor(neigh_list, i);
        for (int groupIndex = 0; groupIndex < countergtotal[attype]; ++groupIndex)
        {
          if (d_SF(attype,d_SFGmemberlist(attype,groupIndex,0),1) == 2)
          {
            int e1 = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 2);
            double rc = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 7);
            int size;
            int memberindex;
            for (int l=0; l < MAX_SF; ++l)
            {
                if (d_SFGmemberlist(attype,groupIndex,l) == 0 && d_SFGmemberlist(attype,groupIndex,l+1) == 0)
                {
                  size = l;
                  break;
                }
            }
            int num_neighs = Cabana::NeighborList<t_verletlist_full_2D>::numNeighbor(neigh_list, i);
            size_t numNeighbors = num_neighs;
            double pfc;
            for (size_t jj = 0; jj < numNeighbors; ++jj)
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
                if (e1 == type(j) && rij < rc)
                {
                    double temp = tanh(1.0 - rij / rc);
                    double temp2 = temp * temp;
                    pfc = temp * temp2;
                    for (size_t k = 0; k < size; ++k) 
                    {
                        double eta = d_SF(attype, d_SFGmemberlist(attype,groupIndex,k), 4);
                        double rs = d_SF(attype, d_SFGmemberlist(attype,groupIndex,k), 8);
                        double pexp = exp(-eta * (rij - rs) * (rij - rs));
                        G(i,d_SFGmemberlist(attype,groupIndex,k)) += pexp*pfc;
                    }
                }
            }
            double raw_value, scaled_value;
            for (size_t k = 0; k < size; ++k)
            {
                memberindex = d_SFGmemberlist(attype,groupIndex,k);
                raw_value = G(i,memberindex); 
                scaled_value = scale(attype, raw_value, memberindex, d_SFscaling);
                G(i,memberindex) = scaled_value; 
            }
          }
          else if (d_SF(attype,d_SFGmemberlist(attype,groupIndex,0),1) == 3)
          {
              int e1 = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 2);
              int e2 = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 3);
              double rc = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 7);
              int size, memberindex, curr_index, next_index;
              for (int l=0; l < MAX_SF; ++l)
              {
                  curr_index = d_SFGmemberlist(attype,groupIndex,l);
                  next_index = d_SFGmemberlist(attype,groupIndex,l+1);
                  if (curr_index == 0 && next_index == 0)
                  {
                    size = l;
                    break;
                  }
              }
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
                  if ((e1 == nej || e2 == nej) && rij < rc)
                  {
                      // Calculate cutoff function and derivative.
                      double pfcij;
                      double temp = tanh(1.0 - rij/rc);
                      double temp2 = temp * temp;
                      pfcij = temp * temp2;
                      
                      for (size_t kk = jj + 1; kk < numNeighbors; kk++)
                      {
                          int k = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, kk);
                          size_t const nek = type(k);
                          
                          if ((e1 == nej && e2 == nek) || (e2 == nej && e1 == nek))
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
                                  double dxjk = dxik - dxij;
                                  double dyjk = dyik - dyij;
                                  double dzjk = dzik - dzij;
                                  double r2jk = dxjk*dxjk + dyjk*dyjk + dzjk*dzjk; 
                                  if (r2jk < rc2)
                                  {
                                      // Energy calculation.
                                      double pfcik;
                                      double temp = tanh(1.0 - rik/rc);
                                      double temp2 = temp * temp;
                                      pfcik = temp * temp2;
                                      
                                      double rjk = sqrt(r2jk);
                                      double pfcjk;
                                      temp = tanh(1.0 - rjk/rc);
                                      temp2 = temp * temp;
                                      pfcjk = temp * temp2;
                                      
                                      double const rinvijik = 1.0 / rij / rik;
                                      double const costijk = (dxij*dxik + dyij*dyik + dzij*dzik)* rinvijik;
                                      double const pfc = pfcij * pfcik * pfcjk;
                                      double const r2sum = r2ij + r2ik + r2jk;
                                      double vexp = 0.0, rijs = 0.0, riks = 0.0, rjks = 0.0;
                                      double eta, lambda, zeta, rs;
                                      for (size_t l = 0; l < size; ++l)
                                      {
                                        memberindex = d_SFGmemberlist(attype,groupIndex,l);
                                        eta = d_SF(attype,memberindex,4);
                                        lambda = d_SF(attype,memberindex,5);
                                        zeta = d_SF(attype,memberindex,6);
                                        rs = d_SF(attype,memberindex,8);
                                        if (rs > 0.0)
                                        {  
                                          rijs = rij - rs;
                                          riks = rik - rs;
                                          rjks = rjk - rs;
                                          vexp = exp(-eta  * (rijs * rijs + riks * riks + rjks * rjks));
                                        }
                                        else
                                            vexp = exp(-eta * r2sum);
                                        double const plambda = 1.0 + lambda * costijk;
                                        double fg = vexp;
                                        if (plambda <= 0.0) fg = 0.0;
                                        else
                                            fg *= pow(plambda, (zeta - 1.0));
                                        G(i,memberindex) += fg * plambda * pfc;
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
                  memberindex = d_SFGmemberlist(attype,groupIndex,k);
                  raw_value = G(i,memberindex) * pow(2,(1-d_SF(attype,memberindex,6))); 
                  G(i,memberindex) = scale(attype, raw_value, memberindex, d_SFscaling);
              }
          }
        }
    });
} 

void Mode::calculateAtomicNeuralNetworks(System* s, AoSoA_NNP nnp_data)
{
    //Calculate Atomic Neural Networks 
    auto id = Cabana::slice<IDs>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto G = Cabana::slice<NNPNames::G>(nnp_data);
    auto dEdG = Cabana::slice<NNPNames::dEdG>(nnp_data);
    auto energy = Cabana::slice<NNPNames::energy>(nnp_data);
    
    //deep copy into device Views
    Kokkos::deep_copy(bias, h_bias);
    Kokkos::deep_copy(weights, h_weights);

    std::cout << "Calculating neural networks" << std::endl;
    Kokkos::parallel_for ("Mode::calculateAtomicNeuralNetworks", s->N_local, KOKKOS_LAMBDA (const size_t atomindex)
    {
        int attype = type(atomindex);
        //set input layer of NN
        for (int k = 0; k < numNeuronsPerLayer[0]; ++k)
            NN(0,k) = G(attype,k);
        //forward propagation
        for (int l = 1; l < numLayers; l++)
        {
            //propagateLayer(layers[i], layers[i-1]);
            //propagateLayer(Layer& layer, Layer& layerPrev) //l and l-1
            double dtmp;
            for (int i = 0; i < numNeuronsPerLayer[l]; i++)
            {
                dtmp = 0.0;
                for (int j = 0; j < numNeuronsPerLayer[l-1]; j++)
                {
                    dtmp += weights(attype,l-1,i,j) * NN(l-1,j);
                }
                dtmp += bias(attype,l,i);
                //if (normalizeNeurons) dtmp /= numNeuronsPerLayer[l-1];

                if (AF[l] == 0)
                {
                    NN(l,i) = dtmp;
                    dfdx(l,i) = 1.0; 
                }
                else if (AF[l] == 1)
                {
                    dtmp = tanh(dtmp);
                    NN(l,i)  = dtmp;
                    dfdx(l,i) = 1.0 - dtmp * dtmp;
                }
                dtmp = NN(l,i);
            }
        }

        //derivative of network w.r.t NN inputs
        double** inner = new double*[numHiddenLayers];
        double** outer = new double*[numHiddenLayers];
        for (int i = 0; i < numHiddenLayers; i++)
        {
            inner[i] = new double[numNeuronsPerLayer[i+1]];
            outer[i] = new double[numNeuronsPerLayer[i+2]];
        }

        for (int k = 0; k < numNeuronsPerLayer[0]; k++)
        {
            for (int i = 0; i < numNeuronsPerLayer[1]; i++)
                inner[0][i] = weights(attype,0,i,k) * dfdx(1,i); 
            
            for (int l = 1; l < numHiddenLayers+1; l++)
            {
                for (int i2 = 0; i2 < numNeuronsPerLayer[l+1]; i2++)
                {
                    outer[l-1][i2] = 0.0;
                    
                    for (int i1 = 0; i1 < numNeuronsPerLayer[l]; i1++)
                        outer[l-1][i2] += weights(attype,l,i2,i1) * inner[l-1][i1];
                    outer[l-1][i2] *= dfdx(l+1,i2);
                    
                    if (l < numHiddenLayers)
                      inner[l][i2] = outer[l-1][i2];
                }
            }
            dEdG(atomindex,k) = outer[numHiddenLayers-1][0];
        }

        for (int i = 0; i < numHiddenLayers; i++)
        {
            delete[] inner[i];
            delete[] outer[i];
        }
        delete[] inner;
        delete[] outer;

        energy(atomindex) = NN(numLayers-1,0); 
    });
}


void Mode::calculateForces(System *s, AoSoA_NNP nnp_data, t_verletlist_full_2D neigh_list, 
    t_mass numSymmetryFunctionsPerElement)
{
    //Calculate Forces 
    auto id = Cabana::slice<IDs>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto x = Cabana::slice<Positions>(s->xvf);
    auto f = Cabana::slice<Forces>(s->xvf);
    auto dEdG = Cabana::slice<NNPNames::dEdG>(nnp_data);
    
    double convForce = 1.0;
    if (s->normalize)
      convForce = convLength/convEnergy;
    
    std::cout << "Calculating forces" << std::endl;
    Kokkos::parallel_for ("Mode::calculateForces", s->N_local, KOKKOS_LAMBDA (const size_t i)
    {
        // Now loop over all neighbor atoms j of atom i. These may hold
        // non-zero derivatives of their symmetry functions with respect to
        // atom i's coordinates. Some atoms may appear multiple times in the
        // neighbor list because of periodic boundary conditions. To avoid
        // that the same contributions are added multiple times use the
        // "unique neighbor" list (but skip the first entry, this is always
        // atom i itself).
        int attype = type(i); 
        //Reset dGdr to zero TODO: deep_copy
        for (int groupIndex = 0; groupIndex < countergtotal[attype]; ++groupIndex)
        {
          if (d_SF(attype,d_SFGmemberlist(attype,groupIndex,0),1) == 2)
          {
              int e1 = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 2);
              double rc = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 7);
              int size, memberindex, curr_index, next_index;
              for (int l=0; l < MAX_SF; ++l)
              {
                  curr_index = d_SFGmemberlist(attype,groupIndex,l);
                  next_index = d_SFGmemberlist(attype,groupIndex,l+1);
                  if (curr_index == 0 && next_index == 0)
                  {
                    size = l;
                    break;
                  }
              }
              int num_neighs = Cabana::NeighborList<t_verletlist_full_2D>::numNeighbor(neigh_list, i);
              size_t numNeighbors = num_neighs;
              double pfc, pdfc;
              for (size_t jj = 0; jj < numNeighbors; ++jj)
              {
                  int j = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, jj);
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
                      double temp = tanh(1.0 - rij / rc);
                      double temp2 = temp * temp;
                      pfc = temp * temp2;
                      pdfc = 3.0 * temp2 * (temp2 - 1.0) / rc;
                      double eta, rs;
                      for (size_t k = 0; k < size; ++k) 
                      {
                          memberindex = d_SFGmemberlist(attype,groupIndex,k);
                          eta = d_SF(attype, memberindex, 4);
                          rs = d_SF(attype, memberindex, 8);
                          double pexp = exp(-eta * (rij - rs) * (rij - rs));
                          // Force calculation.
                          double const p1 = d_SFscaling(attype,memberindex,6) * 
                            (pdfc - 2.0 * eta * (rij - rs) * pfc) * pexp / rij;
                          
                          dGdr(i,memberindex,1) += (p1*dyij);
                          dGdr(i,memberindex,2) += (p1*dzij);

                          dGdr(j,memberindex,0) -= (p1*dxij);
                          dGdr(j,memberindex,1) -= (p1*dyij);
                          dGdr(j,memberindex,2) -= (p1*dzij);
                      }
                  }
              }
          }
          else if (d_SF(attype,d_SFGmemberlist(attype,groupIndex,0),1) == 3)
          {
              int e1 = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 2);
              int e2 = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 3);
              double rc = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 7);
              int size, memberindex, curr_index, next_index;
              for (int l=0; l < MAX_SF; ++l)
              {
                  curr_index = d_SFGmemberlist(attype,groupIndex,l);
                  next_index = d_SFGmemberlist(attype,groupIndex,l+1);
                  if (curr_index == 0 && next_index == 0)
                  {
                    size = l;
                    break;
                  }
              }
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
                  if ((e1 == nej || e2 == nej) && rij < rc)
                  {
                      // Calculate cutoff function and derivative.
                      double pfcij;
                      double pdfcij;
                      double temp = tanh(1.0 - rij/rc);
                      double temp2 = temp * temp;
                      pfcij = temp * temp2;
                      pdfcij = 3.0 * temp2 * (temp2 - 1.0) / rc;
                      
                      for (size_t kk = jj + 1; kk < numNeighbors; kk++)
                      {
                          int k = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, kk);
                          size_t const nek = type(k);
                          
                          if ((e1 == nej && e2 == nek) || (e2 == nej && e1 == nek))
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
                                  double dxjk = dxik - dxij;
                                  double dyjk = dyik - dyij;
                                  double dzjk = dzik - dzij;
                                  double r2jk = dxjk*dxjk + dyjk*dyjk + dzjk*dzjk; 
                                  if (r2jk < rc2)
                                  {
                                      // Energy calculation.
                                      double pfcik;
                                      double pdfcik;
                                      double temp = tanh(1.0 - rik/rc);
                                      double temp2 = temp * temp;
                                      pfcik = temp * temp2;
                                      pdfcik = 3.0 * temp2 * (temp2 - 1.0) / rc;
                                      double rjk = sqrt(r2jk);

                                      double pfcjk;
                                      double pdfcjk;
                                      temp = tanh(1.0 - rjk/rc);
                                      temp2 = temp * temp;
                                      pfcjk = temp * temp2;
                                      pdfcjk = 3.0 * temp2 * (temp2 - 1.0) / rc;
                                      
                                      double const rinvijik = 1.0 / rij / rik;
                                      double const costijk = (dxij*dxik + dyij*dyik + dzij*dzik)* rinvijik;
                                      double const pfc = pfcij * pfcik * pfcjk;
                                      double const r2sum = r2ij + r2ik + r2jk;
                                      double const pr1 = pfcik * pfcjk * pdfcij / rij;
                                      double const pr2 = pfcij * pfcjk * pdfcik / rik;
                                      double const pr3 = pfcij * pfcik * pdfcjk / rjk;
                                      double vexp = 0.0, rijs = 0.0, riks = 0.0, rjks = 0.0;
                                      double rs, eta, lambda, zeta;
                                      for (size_t l = 0; l < size; ++l)
                                      {
                                        memberindex = d_SFGmemberlist(attype,groupIndex,l);
                                        rs = d_SF(attype,memberindex,8);
                                        eta = d_SF(attype,memberindex,4);
                                        lambda = d_SF(attype,memberindex,5);
                                        zeta = d_SF(attype,memberindex,6);
                                        if (rs > 0.0)
                                        {  
                                          rijs = rij - rs;
                                          riks = rik - rs;
                                          rjks = rjk - rs;
                                          vexp = exp(-eta * (rijs * rijs + riks * riks + rjks * rjks));
                                        }
                                        else
                                            vexp = exp(-eta * r2sum);
                                        
                                        double const plambda = 1.0 + lambda * costijk;
                                        double fg = vexp;
                                        if (plambda <= 0.0)
                                          fg = 0.0;
                                        else
                                            fg *= pow(plambda, (zeta - 1.0));
                                        
                                        fg *= pow(2,(1-zeta)) * d_SFscaling(attype,memberindex,6);
                                        double const pfczl = pfc * zeta * lambda; 
                                        double factorDeriv = 2.0 * eta / zeta / lambda;
                                        double const p2etapl = plambda * factorDeriv;
                                        double p1, p2, p3;
                                        if (rs > 0.0)
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
                                        
                                        dGdr(i,memberindex,0) += (p1*dxij + p2*dxik);
                                        dGdr(i,memberindex,1) += (p1*dyij + p2*dyik);
                                        dGdr(i,memberindex,2) += (p1*dzij + p2*dzik);

                                        dGdr(j,memberindex,0) -= (p1*dxij + p3*dxjk);
                                        dGdr(j,memberindex,1) -= (p1*dyij + p3*dyjk);
                                        dGdr(j,memberindex,2) -= (p1*dzij + p3*dzjk);

                                        dGdr(k,memberindex,0) -= (p2*dxik - p3*dxjk);
                                        dGdr(k,memberindex,1) -= (p2*dyik - p3*dyjk);
                                        dGdr(k,memberindex,2) -= (p2*dzik - p3*dzjk);
                                      } // l
                                  } // rjk <= rc
                              } // rik <= rc
                          } // elem
                      } // k
                  } // rij <= rc
              } // j
                    
          }
        }    

        //Use computed dEdG and dGdr to calculate forces
        int num_neighs = Cabana::NeighborList<t_verletlist_full_2D>::numNeighbor(neigh_list, i);
    
        for (size_t jj = 0; jj < num_neighs; ++jj)
        {
            int j = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, jj);
            for (size_t k = 0; k < numSymmetryFunctionsPerElement(type(i)); ++k)
            {
                f(j,0) -= (dEdG(i,k) * dGdr(j,k,0) * s->cfforce * convForce);
                f(j,1) -= (dEdG(i,k) * dGdr(j,k,1) * s->cfforce * convForce);
                f(j,2) -= (dEdG(i,k) * dGdr(j,k,2) * s->cfforce * convForce);
            }
        }
        
        for (size_t k = 0; k < numSymmetryFunctionsPerElement(type(i)); ++k)
        {
            f(i,0) -= (dEdG(i,k) * dGdr(i,k,0) * s->cfforce * convForce);
            f(i,1) -= (dEdG(i,k) * dGdr(i,k,1) * s->cfforce * convForce);
            f(i,2) -= (dEdG(i,k) * dGdr(i,k,2) * s->cfforce * convForce);
        }

    });

    return;
}


