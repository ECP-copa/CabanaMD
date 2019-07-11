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
#include "NeuralNetwork.h"
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
    
    //for each view, create host mirror for I/O
    //deep copy back when needed for calculations

    d_numSymmetryFunctionsPerElement = t_mass("ForceNNP::numSymmetryFunctionsPerElement", numElements);
   
    //setup SF info storage
    SF = t_SF("ForceNNP::SF", numElements, MAX_SF);
    SFG = t_SFG("ForceNNP::SFG", numElements, MAX_SF);
    SFscaling = t_SFscaling("ForceNNP::SFscaling", numElements, MAX_SF);
    SFGmemberlist = t_SFGmemberlist("ForceNNP::SFGmemberlist", numElements);
    
    d_SF = d_t_SF("ForceNNP::SF", numElements, MAX_SF);
    d_SFG = d_t_SFG("ForceNNP::SFG", numElements, MAX_SF);
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
        //TODO: it->sortSymmetryFunctions();
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
        it->setupSymmetryFunctionGroups(SF, SFG, SFGmemberlist, attype, countertotal, countergtotal);
        printf("INIT\n");
        log << strpr("Short range atomic symmetry function groups "
                     "element %2s :\n", it->getSymbol().c_str());
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << " ind ec ty e1 e2       eta        rs la "
               "zeta        rc ct   ca    ln   mi  sfi e\n";
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << it->infoSymmetryFunctionGroups(SFG, attype, countergtotal);
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
    
    int* numNeuronsPerLayer = new int[numLayers];
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
    
    //initialize Views
    int maxNeurons = 0;
    for (int j = 0; j < numLayers; ++j)
        maxNeurons = max(maxNeurons, numNeuronsPerLayer[j]);
    
    NN = t_NN("ForceNNP::NN", numElements, numLayers, maxNeurons); 
    bias = t_NN("ForceNNP::biases", numElements, numLayers, maxNeurons); 
    weights = t_weights("ForceNNP::weights", numElements, numLayers, maxNeurons, maxNeurons); 
    
    d_NN = d_t_NN("ForceNNP::NN", numElements, numLayers, maxNeurons); 
    d_bias = d_t_NN("ForceNNP::biases", numElements, numLayers, maxNeurons); 
    d_weights = d_t_weights("ForceNNP::weights", numElements, numLayers, maxNeurons, maxNeurons); 

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

    delete[] numNeuronsPerLayer;

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
                    weights(attype,layer,start,end) = atof(splitLine.at(0).c_str());
                }
                else if (strcmp(splitLine.at(1).c_str(), "b") == 0)
                {
                    layer = atoi(splitLine.at(3).c_str()) - 1;
                    start = atoi(splitLine.at(4).c_str()) - 1;
                    bias(attype,layer,start) = atof(splitLine.at(0).c_str());
                }
            }
        }
        file.close();
    }
    log << "*****************************************"
           "**************************************\n";
    std::cout << weights(0,0,0,0) << " " << bias(0,0,0) << std::endl;
    return;
}

void Mode::calculateSymmetryFunctions(Structure& structure,
                                      bool const derivatives)
{
    // Skip calculation for whole structure if results are already saved.
    if (structure.hasSymmetryFunctionDerivatives) return;
    if (structure.hasSymmetryFunctions && !derivatives) return;

    Atom* a = NULL;
    Element* e = NULL;
#ifdef _OPENMP
    #pragma omp parallel for private (a, e)
#endif
    for (size_t i = 0; i < structure.atoms.size(); ++i)
    {
        // Pointer to atom.
        a = &(structure.atoms.at(i));

        // Skip calculation for individual atom if results are already saved.
        if (a->hasSymmetryFunctionDerivatives) continue;
        if (a->hasSymmetryFunctions && !derivatives) continue;

        // Get element of atom and set number of symmetry functions.
        e = &(elements.at(a->element));
        int attype = e->getIndex();
        a->numSymmetryFunctions = e->numSymmetryFunctions(attype, countertotal);

#ifndef NONEIGHCHECK
        // Check if atom has low number of neighbors.
        size_t numNeighbors = a->getNumNeighbors(
                                            minCutoffRadius.at(e->getIndex()));
        if (numNeighbors < minNeighbors.at(e->getIndex()))
        {
            log << strpr("WARNING: Structure %6zu Atom %6zu : %zu "
                         "neighbors.\n",
                         a->indexStructure,
                         a->index,
                         numNeighbors);
        }
#endif

        // Allocate symmetry function data vectors in atom.
        a->allocate(derivatives);

        // Calculate symmetry functions (and derivatives).
        //e->calculateSymmetryFunctions(*a, derivatives);

        // Remember that symmetry functions of this atom have been calculated.
        a->hasSymmetryFunctions = true;
        if (derivatives) a->hasSymmetryFunctionDerivatives = true;
    }

    // If requested, check extrapolation warnings or update statistics.
    // Needed to shift this out of the loop above to make it thread-safe.
    if (checkExtrapolationWarnings)
    {
        for (size_t i = 0; i < structure.atoms.size(); ++i)
        {
            a = &(structure.atoms.at(i));
            e = &(elements.at(a->element));
            //e->updateSymmetryFunctionStatistics(*a);
        }
    }

    // Remember that symmetry functions of this structure have been calculated.
    structure.hasSymmetryFunctions = true;
    if (derivatives) structure.hasSymmetryFunctionDerivatives = true;

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
    Kokkos::deep_copy(d_SFG, SFG);
    Kokkos::deep_copy(d_SFGmemberlist, SFGmemberlist);

    // Calculate symmetry functions (and derivatives).
    Kokkos::parallel_for ("Mode::calculateSymmetryFunctionGroups", s->N_local, KOKKOS_LAMBDA (const size_t i) 
    {
        int attype = type(i);
        T_INT numSymmetryFunctions = numSymmetryFunctionsPerElement(type(i));
        
        // Check if atom has low number of neighbors.
        int num_neighs = Cabana::NeighborList<t_verletlist_full_2D>::numNeighbor(neigh_list, i);
        //calculateSFGroups(s, nnp_data, SF, SFscaling, SFGmemberlist, attype, neigh_list, i, countergtotal); 
        printf("about to loop\n");
        for (int groupIndex = 0; groupIndex < countergtotal[attype]; ++groupIndex)
        {
          if (d_SF(attype,d_SFGmemberlist(attype,groupIndex,0),1) == 2)
          {
            printf("begin calculating\n");
            calculateSFGR(s, nnp_data, d_SF, d_SFscaling, d_SFGmemberlist, attype, groupIndex, neigh_list, i);
            printf("done calculating\n");
          }
          else if (d_SF(attype,d_SFGmemberlist(attype,groupIndex,0),1) == 3)
          {
            calculateSFGAN(s, nnp_data, d_SF, d_SFscaling, d_SFGmemberlist, attype, groupIndex, neigh_list, i);
            printf("done calculating\n");
          }
        }
    });

} 

void Mode::calculateAtomicNeuralNetworks(System* s, AoSoA_NNP nnp_data)
{
    //Calculate Atomic Neural Networks 
    auto id = Cabana::slice<IDs>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto energy = Cabana::slice<NNPNames::energy>(nnp_data);

    Kokkos::parallel_for ("Mode::calculateAtomicNeuralNetworks", s->N_local, KOKKOS_LAMBDA (const size_t i)
    {
        setinput(nnp_data,i);
        Propagate();
        calculatedEdG(nnp_data,i);
        //energy(i) = e.neuralNetwork->getOutput();
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
            calculateSFGRD(s, nnp_data, d_SF, d_SFscaling, d_SFGmemberlist, dGdr, attype, groupIndex, neigh_list, i);
          else if (d_SF(attype,d_SFGmemberlist(attype,groupIndex,0),1) == 3)
            calculateSFGAND(s, nnp_data, d_SF, d_SFscaling, d_SFGmemberlist, dGdr, attype, groupIndex, neigh_list, i);
        }    

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


