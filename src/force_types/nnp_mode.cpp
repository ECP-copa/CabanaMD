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
#define MAX_SF 30

#define NNP_VERSION "2.0.0"
#define NNP_GIT_REV "7b73a36a9acfdcc80e44265bac92b055f41a1d07"
#define NNP_GIT_REV_SHORT "7b73a36"
#define NNP_GIT_BRANCH "master"

using namespace std;
using namespace nnpCbn;

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
        log << nnp::strpr("Mean energy per atom     : %24.16E\n", meanEnergy);
        log << nnp::strpr("Conversion factor energy : %24.16E\n", convEnergy);
        log << nnp::strpr("Conversion factor length : %24.16E\n", convLength);
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

void Mode::setupElementMap()
{
    log << "\n";
    log << "*** SETUP: ELEMENT MAP ******************"
           "**************************************\n";
    log << "\n";

    elementStrings = nnp::split(nnp::reduce(settings["elements"]));
    
    log << nnp::strpr("Number of element strings found: %d\n", elementStrings.size());
    for (size_t i = 0; i < elementStrings.size(); ++i)
    {
        log << nnp::strpr("Element %2zu: %2s\n", i, elementStrings[i].c_str());
    }
    //resize numSymmetryFunctionsPerElement to have size = num of atom types in system
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
   
    log << "*****************************************"
           "**************************************\n";

    return;
}

h_t_mass Mode::setupElements(h_t_mass atomicEnergyOffset)
{
    log << "\n";
    log << "*** SETUP: ELEMENTS *********************"
           "**************************************\n";
    log << "\n";

    numElements = (size_t)atoi(settings["number_of_elements"].c_str());
    atomicEnergyOffset = h_t_mass("ForceNNP::atomicEnergyOffset", numElements);
    if (numElements != elementStrings.size())
    {
        throw runtime_error("ERROR: Inconsistent number of elements.\n");
    }
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
            for (int i = 0; i < elementStrings.size(); ++i)
            {
              if (strcmp(elementStrings[i].c_str(), estring) == 0)
                atomicEnergyOffset(i) = atof(args.at(1).c_str());
            }
        }
    }
    
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
    return atomicEnergyOffset;
}

void Mode::setupCutoff()
{
    log << "\n";
    log << "*** SETUP: CUTOFF FUNCTIONS *************"
           "**************************************\n";
    log << "\n";

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

    return;
}

h_t_mass Mode::setupSymmetryFunctions(h_t_mass h_numSymmetryFunctionsPerElement)
{
    h_numSymmetryFunctionsPerElement = h_t_mass("ForceNNP::numSymmetryFunctionsPerElement", numElements);
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTIONS ***********"
           "**************************************\n";
    log << "\n";

    nnp::Settings::KeyRange r = settings.getValues("symfunction_short");
    for (nnp::Settings::KeyMap::const_iterator it = r.first; it != r.second; ++it)
    {
        vector<string> args    = nnp::split(nnp::reduce(it->second.first));
        int type;
        const char* estring = args.at(0).c_str();
        //np.where element symbol == symbol encountered during parsing 
        for (int i = 0; i < elementStrings.size(); ++i)
        {
          if (strcmp(elementStrings[i].c_str(), estring) == 0)
             type = i; 
        }
        elements.at(type).addSymmetryFunction(it->second.first, elementStrings,
                                                 it->second.second, type, SF, convLength, countertotal);
        //set numSymmetryFunctionsPerElement to have number of symmetry functions detected after reading
        h_numSymmetryFunctionsPerElement(type) = countertotal[type];
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
        it->sortSymmetryFunctions(SF, h_numSymmetryFunctionsPerElement, attype);
        maxCutoffRadius = max(it->getMaxCutoffRadius(SF,attype,countertotal), maxCutoffRadius);
        it->setCutoffFunction(cutoffType, cutoffAlpha, SF, attype, countertotal);
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
    minNeighbors.resize(numElements, 0);
    minCutoffRadius.resize(numElements, maxCutoffRadius);
    for (size_t i = 0; i < numElements; ++i)
    {
        int attype = elements.at(i).getIndex();
        int nSF = h_numSymmetryFunctionsPerElement(attype);
        minNeighbors.at(i) = elements.at(i).getMinNeighbors(attype, SF, nSF);
        minCutoffRadius.at(i) = elements.at(i).getMinCutoffRadius(SF,attype,countertotal);
        log << nnp::strpr("Minimum cutoff radius for element %2s: %f\n",
                     elements.at(i).getSymbol().c_str(),
                     minCutoffRadius.at(i) / convLength);
    }
    log << nnp::strpr("Maximum cutoff radius (global)      : %f\n",
                 maxCutoffRadius / convLength);

    log << "*****************************************"
           "**************************************\n";

    return h_numSymmetryFunctionsPerElement;
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
        log << nnp::strpr("Scaling type::ST_SCALE (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmin) / (Gmax - Gmin)\n";
    }
    else if (   (!settings.keywordExists("scale_symmetry_functions" ))
             && ( settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = ST_CENTER;
        log << nnp::strpr("Scaling type::ST_CENTER (%d)\n", scalingType);
        log << "Gs = G - Gmean\n";
    }
    else if (   ( settings.keywordExists("scale_symmetry_functions" ))
             && ( settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = ST_SCALECENTER;
        log << nnp::strpr("Scaling type::ST_SCALECENTER (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmean) / (Gmax - Gmin)\n";
    }
    else if (settings.keywordExists("scale_symmetry_functions_sigma"))
    {
        scalingType = ST_SCALESIGMA;
        log << nnp::strpr("Scaling type::ST_SCALESIGMA (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmean) / Gsigma\n";
    }
    else
    {
        scalingType = ST_NONE;
        log << nnp::strpr("Scaling type::ST_NONE (%d)\n", scalingType);
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

        log << nnp::strpr("Smin = %f\n", Smin);
        log << nnp::strpr("Smax = %f\n", Smax);
    }

    log << nnp::strpr("Symmetry function scaling statistics from file: %s\n",
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
    log << nnp::strpr("Collect min/max/mean/sigma                        : %d\n",
                 (int)collectStatistics);
    log << nnp::strpr("Collect extrapolation warnings                    : %d\n",
                 (int)collectExtrapolationWarnings);
    log << nnp::strpr("Write extrapolation warnings immediately to stderr: %d\n",
                 (int)writeExtrapolationWarnings);
    log << nnp::strpr("Halt on any extrapolation warning                 : %d\n",
                 (int)stopOnExtrapolationWarnings);
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
    log << nnp::strpr("Normalize neurons (all elements): %d\n",
                 (int)normalizeNeurons);
    log << "-----------------------------------------"
           "--------------------------------------\n";
    
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        int attype = it->getIndex();
        numNeuronsPerLayer[0] = it->numSymmetryFunctions(attype, countertotal);
        log << nnp::strpr("Atomic short range NN for "
                     "element %2s :\n", it->getSymbol().c_str());

        int numWeights = 0, numBiases = 0, numConnections = 0;
        for (int j = 1; j < numLayers; ++j)
        { 
            numWeights += numNeuronsPerLayer[j-1]*numNeuronsPerLayer[j]; 
            numBiases += numNeuronsPerLayer[j];
        }
        numConnections = numWeights + numBiases;
        log << nnp::strpr("Number of weights    : %6zu\n", numWeights);
        log << nnp::strpr("Number of biases     : %6zu\n", numBiases);
        log << nnp::strpr("Number of connections: %6zu\n", numConnections);
        log << nnp::strpr("Architecture    %6zu    %6zu    %6zu    %6zu\n", numNeuronsPerLayer[0], numNeuronsPerLayer[1],
            numNeuronsPerLayer[2], numNeuronsPerLayer[3]);
        log << "-----------------------------------------"
               "--------------------------------------\n";
    }

    //initialize Views
    maxNeurons = 0;
    for (int j = 0; j < numLayers; ++j)
        maxNeurons = max(maxNeurons, numNeuronsPerLayer[j]);

    h_bias = t_bias("ForceNNP::biases", numElements, numLayers, maxNeurons); 
    h_weights = t_weights("ForceNNP::weights", numElements, numLayers, maxNeurons, maxNeurons); 
    
    bias = d_t_bias("ForceNNP::biases", numElements, numLayers, maxNeurons); 
    weights = d_t_weights("ForceNNP::weights", numElements, numLayers, maxNeurons, maxNeurons); 

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

    log << nnp::strpr("Weight file name format: %s\n", fileNameFormat.c_str());
    int count = 0;
    int AN = 0;
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        const char* estring = elementStrings[count].c_str();
        //np.where element symbol == symbol encountered in knownElements 
        for (int i = 0; i < knownElements.size(); ++i)
        {
          if (strcmp(knownElements[i].c_str(), estring) == 0)
          {
            AN = i+1;
            break;
          } 
        }
        
        string fileName = nnp::strpr(fileNameFormat.c_str(), AN);
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
    log << "*****************************************"
           "**************************************\n";
    return;
}






void Mode::calculateSymmetryFunctionGroups(System *s, AoSoA_NNP nnp_data, t_verletlist_full_2D neigh_list,
    t_mass numSymmetryFunctionsPerElement) 
{
    auto id = Cabana::slice<IDs>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto x = Cabana::slice<Positions>(s->xvf);
    auto G = Cabana::slice<NNPNames::G>(nnp_data);
    
    //deep copy into device Views
    Kokkos::deep_copy(d_SF, SF);
    Kokkos::deep_copy(d_SFscaling, SFscaling);
    Kokkos::deep_copy(d_SFGmemberlist, SFGmemberlist);
    
    //initialize G to 0.0
    Cabana::deep_copy(G,0.0); 

    // Calculate symmetry functions (and derivatives).
    //calculateSFGroups(s, nnp_data, SF, SFscaling, SFGmemberlist, attype, neigh_list, i, countergtotal);
    //Eval functor;
    //Kokkos::parallel_for (s->N_local, functor);
    //Kokkos::fence();
    timer.reset();
    /*printf("Positions : \n");
    printf("%d %f %f %f\n", id(0), x(0,0), x(0,1), x(0,2));    
    printf("%d %f %f %f\n", id(1), x(1,0), x(1,1), x(1,2));    
    printf("%d %f %f %f\n", id(2), x(2,0), x(2,1), x(2,2));    
    printf("%d %f %f %f\n", id(3), x(3,0), x(3,1), x(3,2));    
    printf("%d %f %f %f\n", id(4), x(4,0), x(4,1), x(4,2));    
    printf("%d %f %f %f\n", id(5), x(5,0), x(5,1), x(5,2));    
   
    printf("Sym func props: \n");
    int attype = 1;
    for (int groupIndex = 0; groupIndex < countergtotal[attype]; ++groupIndex)
    {
        int e1 = SF(attype, SFGmemberlist(attype,groupIndex,0), 2);
        int e2 = SF(attype, SFGmemberlist(attype,groupIndex,0), 3);
        double rc = SF(attype, SFGmemberlist(attype,groupIndex,0), 7);
        int size = SFGmemberlist(attype,groupIndex,MAX_SF);
        printf("%d %d %f %d\n", e1, e2, rc, size);    
        for (size_t k = 0; k < size; ++k) 
        {
            int globalIndex = SF(attype,SFGmemberlist(attype,groupIndex,k),14);
            double eta = SF(attype, SFGmemberlist(attype,groupIndex,k), 4);
            double rs = SF(attype, SFGmemberlist(attype,groupIndex,k), 8);
            double lambda = SF(attype,SFGmemberlist(attype,groupIndex,k),5);
            double zeta = SF(attype,SFGmemberlist(attype,groupIndex,k),6);
            printf("%d %f %f %f %f\n", globalIndex, eta, rs, lambda, zeta);    
        }
    }*/
    //Kokkos::TeamPolicy<> team_policy (s->N_local, Kokkos::AUTO());
    //typedef Kokkos::TeamPolicy<>::member_type member_type;
    //Kokkos::parallel_for ("Mode::calculateSymmetryFunctionGroups", team_policy, KOKKOS_LAMBDA (member_type team_member)
    Kokkos::parallel_for ("Mode::calculateSymmetryFunctionGroups", s->N_local, KOKKOS_LAMBDA (int i)
    {
        //const int i = team_member.league_rank();
        int attype = type(i);
        // Check if atom has low number of neighbors.
        int num_neighs = Cabana::NeighborList<t_verletlist_full_2D>::numNeighbor(neigh_list, i);
        for (int groupIndex = 0; groupIndex < countergtotal[attype]; ++groupIndex)
        {
          if (d_SF(attype,d_SFGmemberlist(attype,groupIndex,0),1) == 2)
          {
            int e1 = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 2);
            double rc = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 7);
            int size = d_SFGmemberlist(attype,groupIndex,MAX_SF);
            //printf("i: %f %f %f %f\n", x(i,0), x(i,1), x(i,2), rc);
            //Kokkos::parallel_for (Kokkos::TeamThreadRange(team_member,num_neighs), [=] (int jj) 
            for (int jj = 0; jj < num_neighs; ++jj)
            {
                double pfcij, pdfcij, eta, rs;
                size_t nej;
                double rij, r2ij; 
                T_F_FLOAT dxij, dyij, dzij;
                int j = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, jj);
                //printf("j: %f %f %f\n", x(j,0), x(j,1), x(j,2));
                nej = type(j);
                dxij = (x(i,0) - x(j,0)) * s->cflength * convLength;
                dyij = (x(i,1) - x(j,1)) * s->cflength * convLength;
                dzij = (x(i,2) - x(j,2)) * s->cflength * convLength;
                r2ij = dxij*dxij + dyij*dyij + dzij*dzij;
                rij = sqrt(r2ij); 
                if (e1 == nej && rij < rc)
                {
                    compute_cutoff(cutoffType, pfcij, pdfcij, rij, rc, false);
                    for (size_t k = 0; k < size; ++k) 
                    {
                        int globalIndex = d_SF(attype,d_SFGmemberlist(attype,groupIndex,k),14);
                        eta = d_SF(attype, d_SFGmemberlist(attype,groupIndex,k), 4);
                        rs = d_SF(attype, d_SFGmemberlist(attype,groupIndex,k), 8);
                        //lsum += exp(-eta * (rij - rs) * (rij - rs))*pfcij;
                        G(i,globalIndex) += exp(-eta * (rij - rs) * (rij - rs))*pfcij;
                    }
                }
            }//);
            
            //team_member.team_barrier();
             
            double raw_value, scaled_value;
            int memberindex, globalIndex;
            for (size_t k = 0; k < size; ++k)
            {
                globalIndex = d_SF(attype,d_SFGmemberlist(attype,groupIndex,k),14);
                memberindex = d_SFGmemberlist(attype,groupIndex,k);
                raw_value = G(i,globalIndex);
                scaled_value = scale(attype, raw_value, memberindex, d_SFscaling);
                G(i,globalIndex) = scaled_value;
            }
          }
          else if (d_SF(attype,d_SFGmemberlist(attype,groupIndex,0),1) == 3)
          {
              int e1 = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 2);
              int e2 = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 3);
              double rc = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 7);
              int size = d_SFGmemberlist(attype,groupIndex,MAX_SF);
              // Prevent problematic condition in loop test below (j < numNeighbors - 1).
              if (num_neighs == 0) num_neighs = 1;
              for (int jj = 0; jj < num_neighs-1; ++jj)
              //Kokkos::parallel_for (Kokkos::TeamThreadRange(team_member,num_neighs-1), [=] (int jj) 
              {
                  double pfcij, pdfcij, pfcik, pdfcik, pfcjk, pdfcjk;
                  size_t nej, nek;
                  int memberindex, globalIndex;
                  double rij, r2ij, rik, r2ik, rjk, r2jk;
                  T_F_FLOAT dxij, dyij, dzij, dxik, dyik, dzik, dxjk, dyjk, dzjk;
                  double eta, rs, lambda, zeta;
                  int j = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, jj);
                  nej = type(j);
                  dxij = (x(i,0) - x(j,0)) * s->cflength * convLength;
                  dyij = (x(i,1) - x(j,1)) * s->cflength * convLength;
                  dzij = (x(i,2) - x(j,2)) * s->cflength * convLength;
                  r2ij = dxij*dxij + dyij*dyij + dzij*dzij;
                  rij = sqrt(r2ij);
                  if ((e1 == nej || e2 == nej) && rij < rc)
                  {
                      // Calculate cutoff function and derivative.
                      compute_cutoff(cutoffType, pfcij, pdfcij, rij, rc, false);
                      
                      for (size_t kk = jj + 1; kk < num_neighs ; kk++)
                      {
                          int k = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, kk);
                          nek = type(k);
                          
                          if ((e1 == nej && e2 == nek) || (e2 == nej && e1 == nek))
                          {
                              dxik = (x(i,0) - x(k,0)) * s->cflength * convLength;
                              dyik = (x(i,1) - x(k,1)) * s->cflength * convLength;
                              dzik = (x(i,2) - x(k,2)) * s->cflength * convLength;
                              r2ik = dxik*dxik + dyik*dyik + dzik*dzik;
                              rik = sqrt(r2ik);
                              if (rik < rc) 
                              {
                                  dxjk = dxik - dxij;
                                  dyjk = dyik - dyij;
                                  dzjk = dzik - dzij;
                                  r2jk = dxjk*dxjk + dyjk*dyjk + dzjk*dzjk; 
                                  if (r2jk < rc*rc)
                                  {
                                      // Energy calculation.
                                      compute_cutoff(cutoffType, pfcik, pdfcik, rik, rc, false);
                                      
                                      rjk = sqrt(r2jk);
                                      compute_cutoff(cutoffType, pfcjk, pdfcjk, rjk, rc, false);
                                      
                                      double const rinvijik = 1.0 / rij / rik;
                                      double const costijk = (dxij*dxik + dyij*dyik + dzij*dzik)* rinvijik;
                                      double vexp = 0.0, rijs = 0.0, riks = 0.0, rjks = 0.0;
                                      for (size_t l = 0; l < size; ++l)
                                      {
                                        globalIndex = d_SF(attype,d_SFGmemberlist(attype,groupIndex,l),14);
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
                                            vexp = exp(-eta * (r2ij + r2ik + r2jk));
                                        double const plambda = 1.0 + lambda * costijk;
                                        double fg = vexp;
                                        if (plambda <= 0.0) fg = 0.0;
                                        else
                                            fg *= pow(plambda, (zeta - 1.0));
                                        //lsum += fg * plambda * pfcij * pfcik * pfcjk;
                                        G(i,globalIndex) += fg * plambda * pfcij * pfcik * pfcjk;
                                      } // l
                                  } // rjk <= rc
                              } // rik <= rc
                          } // elem
                      } // k
                  } // rij <= rc
              }//); // j
              
              //team_member.team_barrier();
              
              double raw_value;
              int memberindex, globalIndex;
              for (size_t k = 0; k < size; ++k)
              {
                  globalIndex = d_SF(attype,d_SFGmemberlist(attype,groupIndex,k),14);
                  memberindex = d_SFGmemberlist(attype,groupIndex,k);
                  //raw_value = sum * pow(2,(1-d_SF(attype,memberindex,6))); 
                  raw_value = G(i,globalIndex) * pow(2,(1-d_SF(attype,memberindex,6))); 
                  G(i,globalIndex) = scale(attype, raw_value, memberindex, d_SFscaling);
              }
          }
        }
    });
    Kokkos::fence();
    double time1 = timer.seconds();
    //printf("Time taken for calculateSF: %f\n", time1);
    /*printf("Sym funcs: \n");
    printf("%d %f %f %f\n", id(0), G(0,0), G(0,1), G(0,2));    
    printf("%d %f %f %f\n", id(1), G(1,0), G(1,1), G(1,2));    
    printf("%d %f %f %f\n", id(2), G(2,0), G(2,1), G(2,2));    
    printf("%d %f %f %f\n", id(3), G(3,0), G(3,1), G(3,2));    
    printf("%d %f %f %f\n", id(4), G(4,0), G(4,1), G(4,2));    
    printf("%d %f %f %f\n", id(5), G(5,0), G(5,1), G(5,2));*/    
} 

void Mode::calculateAtomicNeuralNetworks(System* s, AoSoA_NNP nnp_data, t_mass numSymmetryFunctionsPerElement)
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

    NN = d_t_NN("Mode::NN",s->N,numLayers,maxNeurons);
    dfdx = d_t_NN("Mode::dfdx",s->N,numLayers,maxNeurons);
    inner = d_t_NN("Mode::inner",s->N,numHiddenLayers,maxNeurons);
    outer = d_t_NN("Mode::inner",s->N,numHiddenLayers,maxNeurons);
    
    timer.reset();
    Kokkos::parallel_for ("Mode::calculateAtomicNeuralNetworks", s->N_local, KOKKOS_LAMBDA (const int atomindex)
    {
        for (int i = 0; i < numLayers; ++i)
        {
            for (int j = 0; j < maxNeurons; ++j)
            {
                NN(atomindex,i,j) = 0.0;
                dfdx(atomindex,i,j) = 0.0; 
            }
        }
        
        for (int i = 0; i < numHiddenLayers; i++)
        {
            for (int j = 0; j < maxNeurons; ++j)
            {
                inner(atomindex,i,j) = 0.0;
                outer(atomindex,i,j) = 0.0; 
            }
        }
    
        int attype = type(atomindex);
        //set input layer of NN
        int layer_0, layer_lminusone;
        layer_0 = (int)numSymmetryFunctionsPerElement(attype);
        
        for (int k = 0; k < layer_0; ++k)
            NN(atomindex,0,k) = G(atomindex,k);
        //forward propagation
        for (int l = 1; l < numLayers; l++)
        {
            //propagateLayer(layers[i], layers[i-1]);
            //propagateLayer(Layer& layer, Layer& layerPrev) //l and l-1
            if (l == 1) layer_lminusone = layer_0;
            else layer_lminusone = numNeuronsPerLayer[l-1];
            double dtmp;
            for (int i = 0; i < numNeuronsPerLayer[l]; i++)
            {
                dtmp = 0.0;
                for (int j = 0; j < layer_lminusone; j++)
                    dtmp += weights(attype,l-1,i,j) * NN(atomindex,l-1,j);
                dtmp += bias(attype,l-1,i);
                //if (normalizeNeurons) dtmp /= numNeuronsPerLayer[l-1];
                if (AF[l] == 0)
                {
                    NN(atomindex,l,i) = dtmp;
                    dfdx(atomindex,l,i) = 1.0; 
                }
                else if (AF[l] == 1)
                {
                    dtmp = tanh(dtmp);
                    NN(atomindex,l,i)  = dtmp;
                    dfdx(atomindex,l,i) = 1.0 - dtmp * dtmp;
                }
            }
        }
        
        energy(atomindex) = NN(atomindex,numLayers-1, 0); 
        
        //derivative of network w.r.t NN inputs
        for (int k = 0; k < numNeuronsPerLayer[0]; k++)
        {
            for (int i = 0; i < numNeuronsPerLayer[1]; i++)
                inner(atomindex,0,i) = weights(attype,0,i,k) * dfdx(atomindex,1,i); 
            
            for (int l = 1; l < numHiddenLayers+1; l++)
            {
                for (int i2 = 0; i2 < numNeuronsPerLayer[l+1]; i2++)
                {
                    outer(atomindex,l-1,i2) = 0.0;
                    
                    for (int i1 = 0; i1 < numNeuronsPerLayer[l]; i1++)
                        outer(atomindex,l-1,i2) += weights(attype,l,i2,i1) * inner(atomindex,l-1,i1);
                    outer(atomindex,l-1,i2) *= dfdx(atomindex,l+1,i2);
                    
                    if (l < numHiddenLayers)
                      inner(atomindex,l,i2) = outer(atomindex,l-1,i2);
                }
            }
            dEdG(atomindex,k) = outer(atomindex,numHiddenLayers-1,0);
        }
    });
    Kokkos::fence();
    double time2 = timer.seconds();
    /*printf("dEdG: \n");
    printf("%d %f %f %f\n", id(0), dEdG(0,0), dEdG(0,1), dEdG(0,2));    
    printf("%d %f %f %f\n", id(1), dEdG(1,0), dEdG(1,1), dEdG(1,2));    
    printf("%d %f %f %f\n", id(2), dEdG(2,0), dEdG(2,1), dEdG(2,2));*/ 
    //printf("Time taken for NN: %f\n", time2);
}


void Mode::calculateForces(System *s, AoSoA_NNP nnp_data, t_verletlist_full_2D neigh_list, 
    t_mass numSymmetryFunctionsPerElement)
{
    //Calculate Forces 
    auto id = Cabana::slice<IDs>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto x = Cabana::slice<Positions>(s->xvf);
    auto f = Cabana::slice<Forces>(s->xvf);
    typename AoSoA::member_slice_type<Forces>::atomic_access_slice f_a;
    f_a = Cabana::slice<Forces>(s->xvf);
    auto dEdG = Cabana::slice<NNPNames::dEdG>(nnp_data);
    
    double convForce = convLength/convEnergy;
   
    timer.reset();
    //Kokkos::TeamPolicy<> team_policy (s->N_local, Kokkos::AUTO());
    //typedef Kokkos::TeamPolicy<>::member_type member_type;
    //Kokkos::parallel_for ("Mode::calculateForces", team_policy, KOKKOS_LAMBDA (member_type team_member)
    Kokkos::parallel_for ("Mode::calculateForces", s->N_local, KOKKOS_LAMBDA (int i)
    {
        //const int i = team_member.league_rank();
        
        int attype = type(i); 
        
        int num_neighs = Cabana::NeighborList<t_verletlist_full_2D>::numNeighbor(neigh_list, i);
        for (int groupIndex = 0; groupIndex < countergtotal[attype]; ++groupIndex)
        {
          if (d_SF(attype,d_SFGmemberlist(attype,groupIndex,0),1) == 2)
          {
              int e1 = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 2);
              double rc = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 7);
              int size = d_SFGmemberlist(attype,groupIndex,MAX_SF);
              //Kokkos::parallel_for (Kokkos::TeamThreadRange(team_member,num_neighs), [=] (int jj) 
              for (int jj = 0; jj < num_neighs; ++jj)
              {
                  double pfcij, pdfcij;
                  double rij, r2ij; 
                  T_F_FLOAT dxij, dyij, dzij;
                  double eta, rs;
                  int memberindex, globalIndex;
                  int j = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, jj);
                  size_t nej = type(j);
                  dxij = (x(i,0) - x(j,0)) * s->cflength * convLength;
                  dyij = (x(i,1) - x(j,1)) * s->cflength * convLength;
                  dzij = (x(i,2) - x(j,2)) * s->cflength * convLength;
                  r2ij = dxij*dxij + dyij*dyij + dzij*dzij;
                  rij = sqrt(r2ij); 
                  if (e1 == nej && rij < rc)
                  {
                      // Energy calculation.
                      // Calculate cutoff function and derivative.
                      compute_cutoff(cutoffType, pfcij, pdfcij, rij, rc, true);
                      for (size_t k = 0; k < size; ++k) 
                      {
                          globalIndex = d_SF(attype,d_SFGmemberlist(attype,groupIndex,k),14);
                          memberindex = d_SFGmemberlist(attype,groupIndex,k);
                          eta = d_SF(attype, memberindex, 4);
                          rs = d_SF(attype, memberindex, 8);
                          double pexp = exp(-eta * (rij - rs) * (rij - rs));
                          // Force calculation.
                          double const p1 = d_SFscaling(attype,memberindex,6) * 
                            (pdfcij - 2.0 * eta * (rij - rs) * pfcij) * pexp / rij;
                          f_a(i,0) -= (dEdG(i,globalIndex) * (p1*dxij) * s->cfforce * convForce);
                          f_a(i,1) -= (dEdG(i,globalIndex) * (p1*dyij) * s->cfforce * convForce);
                          f_a(i,2) -= (dEdG(i,globalIndex) * (p1*dzij) * s->cfforce * convForce);

                          f_a(j,0) += (dEdG(i,globalIndex) * (p1*dxij) * s->cfforce * convForce);
                          f_a(j,1) += (dEdG(i,globalIndex) * (p1*dyij) * s->cfforce * convForce);
                          f_a(j,2) += (dEdG(i,globalIndex) * (p1*dzij) * s->cfforce * convForce);
                      }
                  }
              }//);
          }
          else if (d_SF(attype,d_SFGmemberlist(attype,groupIndex,0),1) == 3)
          {
              int e1 = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 2);
              int e2 = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 3);
              double rc = d_SF(attype, d_SFGmemberlist(attype,groupIndex,0), 7);
              int size = d_SFGmemberlist(attype,groupIndex,MAX_SF);
              // Prevent problematic condition in loop test below (j < numNeighbors - 1).
              if (num_neighs == 0) num_neighs = 1;

              //Kokkos::parallel_for (Kokkos::TeamThreadRange(team_member,num_neighs-1), [=] (int jj) 
              for (int jj = 0; jj < num_neighs-1; ++jj)
              {
                  double pfcij, pdfcij, pfcik, pdfcik, pfcjk, pdfcjk;
                  size_t nej, nek;
                  double rij, r2ij, rik, r2ik, rjk, r2jk;
                  T_F_FLOAT dxij, dyij, dzij, dxik, dyik, dzik, dxjk, dyjk, dzjk;
                  double eta, rs, lambda, zeta;
                  int memberindex, globalIndex;
                  int j = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, jj);
                  nej = type(j);
                  dxij = (x(i,0) - x(j,0)) * s->cflength * convLength;
                  dyij = (x(i,1) - x(j,1)) * s->cflength * convLength;
                  dzij = (x(i,2) - x(j,2)) * s->cflength * convLength;
                  r2ij = dxij*dxij + dyij*dyij + dzij*dzij;
                  rij = sqrt(r2ij);
                  if ((e1 == nej || e2 == nej) && rij < rc)
                  {
                      // Calculate cutoff function and derivative.
                      compute_cutoff(cutoffType, pfcij, pdfcij, rij, rc, true);
                      
                      for (size_t kk = jj + 1; kk < num_neighs ; kk++)
                      {
                          int k = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, kk);
                          nek = type(k);
                          
                          if ((e1 == nej && e2 == nek) || (e2 == nej && e1 == nek))
                          {
                              dxik = (x(i,0) - x(k,0)) * s->cflength * convLength;
                              dyik = (x(i,1) - x(k,1)) * s->cflength * convLength;
                              dzik = (x(i,2) - x(k,2)) * s->cflength * convLength;
                              r2ik = dxik*dxik + dyik*dyik + dzik*dzik;
                              rik = sqrt(r2ik);
                              if (rik < rc)
                              {
                                  dxjk = dxik - dxij;
                                  dyjk = dyik - dyij;
                                  dzjk = dzik - dzij;
                                  r2jk = dxjk*dxjk + dyjk*dyjk + dzjk*dzjk; 
                                  if (r2jk < rc*rc)
                                  {
                                      // Energy calculation.
                                      compute_cutoff(cutoffType, pfcik, pdfcik, rik, rc, true);
                                      rjk = sqrt(r2jk);
                                      
                                      compute_cutoff(cutoffType, pfcjk, pdfcjk, rjk, rc, true);
                                      
                                      double const rinvijik = 1.0 / rij / rik;
                                      double const costijk = (dxij*dxik + dyij*dyik + dzij*dzik)* rinvijik;
                                      double const pfc = pfcij * pfcik * pfcjk;
                                      double const r2sum = r2ij + r2ik + r2jk;
                                      double const pr1 = pfcik * pfcjk * pdfcij / rij;
                                      double const pr2 = pfcij * pfcjk * pdfcik / rik;
                                      double const pr3 = pfcij * pfcik * pdfcjk / rjk;
                                      double vexp = 0.0, rijs = 0.0, riks = 0.0, rjks = 0.0;
                                      for (size_t l = 0; l < size; ++l)
                                      {
                                        globalIndex = d_SF(attype,d_SFGmemberlist(attype,groupIndex,l),14);
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
                                        f_a(i,0) -= (dEdG(i,globalIndex) * (p1*dxij + p2*dxik) * s->cfforce * convForce);
                                        f_a(i,1) -= (dEdG(i,globalIndex) * (p1*dyij + p2*dyik) * s->cfforce * convForce);
                                        f_a(i,2) -= (dEdG(i,globalIndex) * (p1*dzij + p2*dzik) * s->cfforce * convForce);

                                        f_a(j,0) += (dEdG(i,globalIndex) * (p1*dxij + p3*dxjk) * s->cfforce * convForce);
                                        f_a(j,1) += (dEdG(i,globalIndex) * (p1*dyij + p3*dyjk) * s->cfforce * convForce);
                                        f_a(j,2) += (dEdG(i,globalIndex) * (p1*dzij + p3*dzjk) * s->cfforce * convForce);
                                        
                                        f_a(k,0) += (dEdG(i,globalIndex) * (p2*dxik - p3*dxjk) * s->cfforce * convForce);
                                        f_a(k,1) += (dEdG(i,globalIndex) * (p2*dyik - p3*dyjk) * s->cfforce * convForce);
                                        f_a(k,2) += (dEdG(i,globalIndex) * (p2*dzik - p3*dzjk) * s->cfforce * convForce);
                                      } // l
                                  } // rjk <= rc
                              } // rik <= rc
                          } // elem
                      } // k
                  } // rij <= rc
              }//); // j
          }
        }
        
    });
    
    Kokkos::fence();
    double time3 = timer.seconds();
    //printf("Time taken for calculateForces: %f\n", time3);
    return;
}


