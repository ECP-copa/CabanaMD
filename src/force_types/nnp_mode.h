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

#ifndef CBN_MODE_H
#define CBN_MODE_H

#include "nnp_cutoff.h"
#include "nnp_element.h"
#include "Log.h"
#include "Settings.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector
#include <Cabana_Core.hpp>
#include <system.h>
#include <types.h>
#include <types_nnp.h>

namespace nnpCbn
{

/** Base class for all NNP applications.
 *
 * This top-level class is the anchor point for existing and future
 * applications. It contains functions to set up an existing neural network
 * potential and calculate energies and forces for configurations given as
 * Structure. A minimal setup requires some consecutive functions calls as this
 * minimal example shows:
 *
 * ```
 * Mode mode;
 * mode.initialize();
 * mode.loadSettingsFile();
 * mode.setupElementMap();
 * mode.setupElements();
 * mode.setupCutoff();
 * mode.setupSymmetryFunctions();
 * mode.setupSymmetryFunctionGroups();
 * mode.setupSymmetryFunctionStatistics(false, false, true, false);
 * mode.setupNeuralNetwork();
 * ```
 * To load weights and scaling information from files add these lines:
 * ```
 * mode.setupSymmetryFunctionScaling();
 * mode.setupNeuralNetworkWeights();
 * ```
 * The NNP is now ready! If we load a structure from a data file:
 * ```
 * Structure structure;
 * ifstream file;
 * file.open("input.data");
 * structure.setupElementMap(mode.elementMap);
 * structure.readFromFile(file);
 * file.close();
 * ```
 * we can finally predict the energy and forces from the neural network
 * potential:
 * ```
 * structure.calculateNeighborList(mode.getMaxCutoffRadius());
 * mode.calculateSymmetryFunctionGroups(structure, true);
 * mode.calculateAtomicNeuralNetworks(structure, true);
 * mode.calculateEnergy(structure);
 * mode.calculateForces(structure);
 * cout << structure.energy << '\n';
 * ```
 * The resulting potential energy is stored in Structure::energy, the forces
 * on individual atoms are located within the Structure::atoms vector in
 * Atom::f.
 */
class Mode
{
public:
    Mode();
    /** Write welcome message with version information.
     */
    void                     initialize();
    /** Open settings file and load all keywords into memory.
     *
     * @param[in] fileName Settings file name.
     */
    void                     loadSettingsFile(std::string const& fileName
                                                                 = "input.nn");
    /** Set up normalization.
     *
     * If the keywords `mean_energy`, `conv_length` and
     * `conv_length` are present, the provided conversion factors are used to
     * internally use a different unit system.
     */
    void                     setupNormalization();
    /** Set up the element map.
     *
     * Uses keyword `elements`. This function should follow immediately after
     * settings are loaded via loadSettingsFile().
     */
    void setupElementMap();
    /** Set up all Element instances.
     *
     * Uses keywords `number_of_elements` and `atom_energy`. This function
     * should follow immediately after setupElementMap().
     */
    h_t_mass                     setupElements(h_t_mass atomicEnergyOffset);
    /** Set up cutoff function for all symmetry functions.
     *
     * Uses keyword `cutoff_type`. Cutoff parameters are read from settings
     * keywords and stored internally. As soon as setupSymmetryFunctions() is
     * called the settings are restored and used for all symmetry functions.
     * Thus, this function must be called before setupSymmetryFunctions().
     */
    void                     setupCutoff();
    /** Set up all symmetry functions.
     *
     * Uses keyword `symfunction_short`. Reads all symmetry functions from
     * settings and automatically assigns them to the correct element.
     */
    h_t_mass                     setupSymmetryFunctions(h_t_mass h_numSFperElem);
    /** Set up symmetry function scaling from file.
     *
     * @param[in] fileName Scaling file name.
     *
     * Uses keywords `scale_symmetry_functions`, `center_symmetry_functions`,
     * `scale_symmetry_functions_sigma`, `scale_min_short` and
     * `scale_max_short`. Reads in scaling information and sets correct scaling
     * behavior for all symmetry functions. Call after
     * setupSymmetryFunctions().
     */
    void                     setupSymmetryFunctionScaling(
                                 std::string const& fileName = "scaling.data");
    /** Set up symmetry function groups.
     *
     * Does not use any keywords. Call after setupSymmetryFunctions() and
     * ensure that correct scaling behavior has already been set.
     */
    void setupSymmetryFunctionGroups();
    /** Set up symmetry function statistics collection.
     *
     * @param[in] collectStatistics Whether statistics (min, max, mean, sigma)
     *                              is collected.
     * @param[in] collectExtrapolationWarnings Whether extrapolation warnings
     *                                         are logged.
     * @param[in] writeExtrapolationWarnings Write extrapolation warnings
     *                                       immediately when they occur.
     * @param[in] stopOnExtrapolationWarnings Throw error immediately when
     *                                        an extrapolation warning occurs.
     *
     * Does not use any keywords. Calling this setup function is not required,
     * by default no statistics collection is enabled (all arguments `false`).
     * Call after setupElements().
     */
    void                     setupSymmetryFunctionStatistics(
                                             bool collectStatistics,
                                             bool collectExtrapolationWarnings,
                                             bool writeExtrapolationWarnings,
                                             bool stopOnExtrapolationWarnings);
    /** Set up neural networks for all elements.
     *
     * Uses keywords `global_hidden_layers_short`, `global_nodes_short`,
     * `global_activation_short`, `normalize_nodes`. Call after
     * setupSymmetryFunctions(), only then the number of input layer neurons is
     * known.
     */
    void                     setupNeuralNetwork();
    /** Set up neural network weights from files.
     *
     * @param[in] fileNameFormat Format for weights file name. The string must
     *                           contain one placeholder for the atomic number.
     *
     * Does not use any keywords. The weight files should contain one weight
     * per line, see NeuralNetwork::setConnections() for the correct order.
     */
    void                     setupNeuralNetworkWeights(
                                 std::string const& fileNameFormat
                                                       = "weights.%03zu.data");
    /** Apply normalization to given energy.
     *
     * @param[in] energy Input energy in physical units.
     *
     * @return Energy in normalized units.
     */
    double                   normalizedEnergy(double energy) const;
    /** Apply normalization to given force.
     *
     * @param[in] force Input force in physical units.
     *
     * @return Force in normalized units.
     */
    double                   normalizedForce(double force) const;
    /** Undo normalization for a given energy.
     *
     * @param[in] energy Input energy in normalized units.
     *
     * @return Energy in physical units.
     */
    double                   physicalEnergy(double energy) const;
    /** Undo normalization for a given force.
     *
     * @param[in] force Input force in normalized units.
     *
     * @return Force in physical units.
     */
    double                   physicalForce(double force) const;
    /** Count total number of extrapolation warnings encountered for all
     * elements and symmetry functions.
     *
     * @return Number of extrapolation warnings.
     */
    std::size_t              getNumExtrapolationWarnings() const;
    /** Erase all extrapolation warnings and reset counters.
     */
    void                     resetExtrapolationWarnings();
    /** Getter for Mode::meanEnergy.
     *
     * @return Mean energy per atom.
     */
    double                   getMeanEnergy() const;
    /** Getter for Mode::convEnergy.
     *
     * @return Energy unit conversion factor.
     */
    double                   getConvEnergy() const;
    /** Getter for Mode::convLength.
     *
     * @return Length unit conversion factor.
     */
    double                   getConvLength() const;
    /** Getter for Mode::maxCutoffRadius.
     *
     * @return Maximum cutoff radius of all symmetry functions.
     *
     * The maximum cutoff radius is determined by setupSymmetryFunctions().
     */
    double                   getMaxCutoffRadius() const;
    /** Getter for Mode::numElements.
     *
     * @return Number of elements defined.
     *
     * The number of elements is determined by setupElements().
     */
    std::size_t              getNumElements() const;
    /** Get number of symmetry functions per element.
     *
     * @return Vector with number of symmetry functions for each element.
     */
    std::vector<std::size_t> getNumSymmetryFunctions(); 
    /** Check if normalization is enabled.
     *
     * @return Value of #normalize.
     */
    bool                     useNormalization() const;
    /** Check if keyword was found in settings file.
     *
     * @param[in] keyword Keyword for which value is requested.
     *
     * @return `true` if keyword exists, `false` otherwise.
     */
    bool                     settingsKeywordExists(
                                             std::string const& keyword) const;
    /** Get value for given keyword in Settings instance.
     *
     * @param[in] keyword Keyword for which value is requested.
     *
     * @return Value string corresponding to keyword.
     */
    std::string              settingsGetValue(
                                             std::string const& keyword) const;
    /** Prune symmetry functions according to their range and write settings
     * file.
     *
     * @param[in] threshold Symmetry functions with range (max - min) smaller
     *                      than this threshold will be pruned.
     *
     * @return List of line numbers with symmetry function to be removed.
     */
    std::vector<std::size_t> pruneSymmetryFunctionsRange(double threshold);
    /** Prune symmetry functions with sensitivity analysis data.
     *
     * @param[in] threshold Symmetry functions with sensitivity lower than this
     *                      threshold will be pruned.
     * @param[in] sensitivity Sensitivity data for each element and symmetry
     *                        function.
     *
     * @return List of line numbers with symmetry function to be removed.
     */
    std::vector<std::size_t> pruneSymmetryFunctionsSensitivity(
                                            double threshold,
                                            std::vector<
                                            std::vector<double> > sensitivity);
    /** Copy settings file but comment out lines provided.
     *
     * @param[in] prune List of line numbers to comment out.
     * @param[in] fileName Output file name.
     */
    void                     writePrunedSettingsFile(
                                              std::vector<std::size_t> prune,
                                              std::string              fileName
                                                          = "output.nn") const;
    /** Write complete settings file.
     *
     * @param[in,out] file Settings file.
     */
    void                     writeSettingsFile(
                                             std::ofstream* const& file) const;

    /// Global element map, populated by setupElementMap().
    //ElementMap elementMap;
    /// Global list of number of atoms per element
    std::vector<std::size_t> numAtomsPerElement;
    /** Allocate vectors related to symmetry functions (#G, #dEdG).
     *
     * @param[in] all If `true` allocate also vectors corresponding to
     *                derivatives of symmetry functions (#dEdG, #dGdr, #dGdxia
     *                and Neighbor::dGdr, neighbors must be present). If
     *                `false` allocate only #G.
     *
     * Warning: #numSymmetryFunctions needs to be set first!
     */
    void allocate(System* s, T_INT numSymmetryFunctions, bool all);
    
    KOKKOS_INLINE_FUNCTION
    void compute_cutoff(CutoffFunction::CutoffType cutoffType, double &fc,
                        double &dfc, double r, double rc, bool derivative);

    KOKKOS_INLINE_FUNCTION
    double scale(int attype, double value, int k, d_t_SFscaling SFscaling);

    template<class t_neighbor, class t_neigh_parallel, class t_angle_parallel>
    void calculateForces(System *s, AoSoA_NNP nnp_data, t_neighbor neigh_list);

    template<class t_neighbor, class t_neigh_parallel, class t_angle_parallel>
    void calculateAtomicNeuralNetworks(System* s, AoSoA_NNP nnp_data, t_mass numSFperElem);

    template<class t_neighbor, class t_neigh_parallel, class t_angle_parallel>
    void calculateSymmetryFunctionGroups(System *s, AoSoA_NNP nnp_data, t_neighbor neigh_list);

    /// Global log file.
    nnp::Log        log;
    
    /// list of element symbols in order of periodic table
    vector<string> knownElements = { 
    "H" , "He", "Li", "Be", "B" , "C" , "N" , "O" , "F" , "Ne", "Na", "Mg", "Al",
    "Si", "P" , "S" , "Cl", "Ar", "K" , "Ca", "Sc", "Ti", "V" , "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y" ,
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
    "I" , "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W" , "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
    "U" , "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No"
    };
    
    //SymmetryFunctionTypes
    d_t_SF d_SF;
    t_SF SF;
    d_t_SFGmemberlist d_SFGmemberlist;
    t_SFGmemberlist SFGmemberlist;
    d_t_SFscaling d_SFscaling;
    t_SFscaling SFscaling;
   
    //NN data structures
    d_t_NN NN,dfdx,inner,outer;
    d_t_bias bias; 
    d_t_weights weights; 
    t_bias h_bias;
    t_weights h_weights;
    int numLayers, numHiddenLayers, maxNeurons; 
    int numNeuronsPerLayer[4];
    int AF[4];
    //int* numNeuronsPerLayer = new int[numLayers];
    //int* AF = new int[numLayers];

    int countertotal[2] = {0,0};
    int countergtotal[2] = {0,0};

    bool                          normalize;
    bool                          checkExtrapolationWarnings;
    std::size_t                   numElements;
    std::vector<std::size_t>      minNeighbors;
    std::vector<double>           minCutoffRadius;
    double                        maxCutoffRadius;
    double                        cutoffAlpha;
    double                        meanEnergy;
    double                        convEnergy;
    double                        convLength;
    ScalingType                   scalingType;
    nnp::Settings                 settings;
    CutoffFunction::CutoffType    cutoffType;
    std::vector<Element>          elements;
    std::vector<string>           elementStrings;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline double Mode::getMeanEnergy() const
{
    return meanEnergy;
}

inline double Mode::getConvEnergy() const
{
    return convEnergy;
}

inline double Mode::getConvLength() const
{
    return convLength;
}

inline double Mode::getMaxCutoffRadius() const
{
    return maxCutoffRadius;
}

inline std::size_t Mode::getNumElements() const
{
    return numElements;
}

inline bool Mode::useNormalization() const
{
    return normalize;
}

//------------------- HELPERS TO MEGA FUNCTION  --------------//

KOKKOS_INLINE_FUNCTION void Mode::compute_cutoff(CutoffFunction::CutoffType cutoffType, double &fc, double &dfc, double r, double rc, bool derivative)
{
   double temp;
   if (cutoffType == CutoffFunction::CT_TANHU)
   {
      temp = tanh(1.0 - r/rc);
      fc = temp * temp * temp;
      if (derivative)
        dfc = 3.0 * temp*temp * (temp*temp - 1.0) / rc;
   }

   if (cutoffType == CutoffFunction::CT_COS)
   {
      
      double rci = rc * cutoffAlpha;
      double iw = 1.0 / (rc - rci);
      double PI = 4.0 * atan(1.0);
      if (r < rci)
      {
          fc = 1.0;
          dfc = 0.0;
      }
      else
      {
          temp = cos(PI * (r - rci) * iw);
          fc = 0.5 * (temp + 1.0);
          if (derivative)
            dfc = -0.5 * iw * PI * sqrt(1.0 - temp * temp);
      }
   }

}

KOKKOS_INLINE_FUNCTION double Mode::scale(int attype, double value, int k, d_t_SFscaling SFscaling)
{
    double scalingType = SFscaling(attype,k,7);
    double scalingFactor = SFscaling(attype,k,6);
    double Gmin = SFscaling(attype,k,0);
    //double Gmax = SFscaling(attype,k,1);
    double Gmean = SFscaling(attype,k,2);
    //double Gsigma = SFscaling(attype,k,3);
    double Smin = SFscaling(attype,k,4);
    //double Smax = SFscaling(attype,k,5);
    
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

}

#endif
