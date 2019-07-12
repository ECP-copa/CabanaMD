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

#ifndef MODE_H
#define MODE_H

#include "CutoffFunction.h"
#include "Element.h"
#include "ElementMap.h"
#include "Log.h"
#include "Settings.h"
#include "SymmetryFunction.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector
#include <system.h>
#include <Cabana_NeighborList.hpp>
#include <Cabana_Slice.hpp>
#include <types.h>

namespace nnp
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
    /** Combine multiple setup routines and provide a basic NNP setup.
     *
     * Sets up elements, symmetry functions, symmetry function groups, neural
     * networks. No symmetry function scaling data is read, no weights are set.
     */
    void                     setupGeneric(t_mass numSymmetryFunctionsPerElement);
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
    t_mass                     setupElementMap(t_mass numSymmetryFunctionsPerElement);
    /** Set up all Element instances.
     *
     * Uses keywords `number_of_elements` and `atom_energy`. This function
     * should follow immediately after setupElementMap().
     */
    void                     setupElements();
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
    t_mass                     setupSymmetryFunctions(t_mass numSymmetryFunctionsPerElement);
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
    ElementMap elementMap;
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
    
    void calculateSymmetryFunctionGroups(System *s, AoSoA_NNP nnp_data, t_verletlist_full_2D neigh_list, 
        t_mass numSymmetryFunctionsPerElement);
    
    void calculateAtomicNeuralNetworks(System* s, AoSoA_NNP nnp_data);
    
    KOKKOS_INLINE_FUNCTION double scale(int attype, double value, int k, d_t_SFscaling SFscaling);

    KOKKOS_INLINE_FUNCTION void calculateSFGR(System* s, AoSoA_NNP nnp_data, d_t_SF SF, d_t_SFscaling SFscaling, d_t_SFGmemberlist SFGmemberlist, int attype, int groupIndex, t_verletlist_full_2D neigh_list, T_INT i);

    KOKKOS_INLINE_FUNCTION void calculateSFGAN(System* s, AoSoA_NNP nnp_data, d_t_SF SF, d_t_SFscaling SFscaling, d_t_SFGmemberlist SFGmemberlist, int attype, int groupIndex, t_verletlist_full_2D neigh_list, T_INT i);

    KOKKOS_INLINE_FUNCTION void calculateSFGRD(System* s, AoSoA_NNP nnp_data, d_t_SF SF, d_t_SFscaling SFscaling, d_t_SFGmemberlist SFGmemberlist, t_dGdr dGdr, int attype, int groupIndex, t_verletlist_full_2D neigh_list, T_INT i);

    KOKKOS_INLINE_FUNCTION void calculateSFGAND(System* s, AoSoA_NNP nnp_data, d_t_SF SF, d_t_SFscaling SFscaling, d_t_SFGmemberlist SFGmemberlist, t_dGdr dGdr, int attype, int groupIndex, t_verletlist_full_2D neigh_list, T_INT i);
    
    void calculateForces(System *s, AoSoA_NNP nnp_data, t_verletlist_full_2D neigh_list, 
    t_mass numSymmetryFunctionsPerElement);
    
    /// Global log file.
    Log        log;
    
    //SymmetryFunctionTypes
    d_t_SF d_SF;
    t_SF SF;
    d_t_SFGmemberlist d_SFGmemberlist;
    t_SFGmemberlist SFGmemberlist;
    d_t_SFscaling d_SFscaling;
    t_SFscaling SFscaling;
    t_dGdr dGdr;
   
    //NN data structures
    d_t_NN NN, dfdx;
    d_t_bias bias; 
    d_t_weights weights; 
    t_bias h_bias;
    t_weights h_weights;
    int numLayers, numHiddenLayers; 
    int numNeuronsPerLayer[4];
    int AF[4];
    //int* AF = new int[numLayers]; 
    //int* numNeuronsPerLayer = new int[numLayers];

    int countertotal[2] = {0,0};
    int countergtotal[2] = {0,0};
    
protected:
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
    Settings                      settings;
    CutoffFunction::CutoffType    cutoffType;
    std::vector<Element>          elements;
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

KOKKOS_INLINE_FUNCTION double Mode::scale(int attype, double value, int k, d_t_SFscaling SFscaling)
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

KOKKOS_INLINE_FUNCTION void Mode::calculateSFGR(System* s, AoSoA_NNP nnp_data, d_t_SF SF, d_t_SFscaling SFscaling, d_t_SFGmemberlist SFGmemberlist, int attype, int groupIndex, t_verletlist_full_2D neigh_list, T_INT i)
{
    printf("Beginning slicing\n");
    auto x = Cabana::slice<Positions>(s->xvf);
    //printf("Sliced X\n");
    auto id = Cabana::slice<IDs>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto G = Cabana::slice<NNPNames::G>(nnp_data);

    printf("Done slicing\n");
    //pick e1, rc from first member (since they are all the same)
    //SFGmemberlist(attype,groupIndex,0): second index = groupIndex, third index  = for first SF in group
    int e1 = SF(attype, SFGmemberlist(attype,groupIndex,0), 2);
    double rc = SF(attype, SFGmemberlist(attype,groupIndex,0), 7);
    
    int size;
    for (int l=0; l < MAX_SF; ++l)
    {
        if (SFGmemberlist(attype,groupIndex,l) == 0 && SFGmemberlist(attype,groupIndex,l+1) == 0)
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
            // Energy calculation.
            // Calculate cutoff function and derivative.
            double temp = tanh(1.0 - rij / rc);
            double temp2 = temp * temp;
            pfc = temp * temp2;
            
            //double const* const d1 = n.dr.r;
            //TODO: use subview for size calculation
            //auto SFGmemberlist_subview = Kokkos::subview(SFGmemberlist, make_pair(0,1), make_pair(0,Kokkos::ALL));
            for (size_t k = 0; k < size; ++k) 
            {
                double eta = SF(attype, SFGmemberlist(attype,groupIndex,k), 4);
                double rs = SF(attype, SFGmemberlist(attype,groupIndex,k), 8);
                double pexp = exp(-eta * (rij - rs) * (rij - rs));
                G(i,SFGmemberlist(attype,groupIndex,k)) += pexp*pfc;
            }
        }
    }

    double raw_value;
    for (size_t k = 0; k < size; ++k)
    {
        raw_value = G(i,SFGmemberlist(attype,groupIndex,k)); 
        G(i,SFGmemberlist(attype,groupIndex,k)) = scale(attype, raw_value, SFGmemberlist(attype,groupIndex,k), SFscaling);
    }
    return;
}



KOKKOS_INLINE_FUNCTION void Mode::calculateSFGAN(System* s, AoSoA_NNP nnp_data, d_t_SF SF, d_t_SFscaling SFscaling, d_t_SFGmemberlist SFGmemberlist, int attype, int groupIndex, t_verletlist_full_2D neigh_list, T_INT i) 
{
    auto x = Cabana::slice<Positions>(s->xvf);
    auto id = Cabana::slice<IDs>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto G = Cabana::slice<NNPNames::G>(nnp_data);
    
    //pick e1, rc from first member (since they are all the same)
    //SFGmemberlist(1,0): first index for angular narrow, second index for first SF in angular narrow group
    int e1 = SF(attype, SFGmemberlist(attype,groupIndex,0), 2);
    int e2 = SF(attype, SFGmemberlist(attype,groupIndex,0), 3);
    double rc = SF(attype, SFGmemberlist(attype,groupIndex,0), 7);
    //CutoffFunction fc(rc);
    
    int size;
    for (int l=0; l < MAX_SF; ++l)
    {
        if (SFGmemberlist(attype,groupIndex,l) == 0 && SFGmemberlist(attype,groupIndex,l+1) == 0)
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
        //Atom::Neighbor& nj = atom.neighbors[j];
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
            
            // SIMPLE EXPRESSIONS:
            //Vec3D const drij(atom.neighbors[j].dr);
            //double const* const dr1 = drij.r;

            for (size_t kk = jj + 1; kk < numNeighbors; kk++)
            {
                int k = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, kk);
                //Atom::Neighbor& nk = atom.neighbors[k];
                size_t const nek = type(k);
                
                if ((e1 == nej && e2 == nek) ||
                    (e2 == nej && e1 == nek))
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
                        // SIMPLE EXPRESSIONS:
                        //Vec3D const drjk(atom.neighbors[k].dr
        
                        //double rjk = drjk.norm2();
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

                            // SIMPLE EXPRESSIONS:
                            //Vec3D const drik(atom.neighbors[k].dr);
                            //double const* const dr2 = drik.r;
                            //double const* const dr3 = drjk.r;
                            double const rinvijik = 1.0 / rij / rik;
                            // SIMPLE EXPRESSIONS:
                            //double const costijk = (drij * drik) * rinvijik;
                            double const costijk = (dxij*dxik + dyij*dyik + dzij*dzik)* rinvijik;
                            double const pfc = pfcij * pfcik * pfcjk;
                            double const r2sum = r2ij + r2ik + r2jk;
                            double vexp = 0.0;
                            double rijs = 0.0;
                            double riks = 0.0;
                            double rjks = 0.0;
                            for (size_t l = 0; l < size; ++l)
                            {
                              if (SF(attype,SFGmemberlist(attype,groupIndex,l),8) > 0.0)
                              {  
                                rijs = rij - SF(attype,SFGmemberlist(attype,groupIndex,l),8);
                                riks = rik - SF(attype,SFGmemberlist(attype,groupIndex,l),8);
                                rjks = rjk - SF(attype,SFGmemberlist(attype,groupIndex,l),8);
                                vexp = exp(-SF(attype,SFGmemberlist(attype,groupIndex,l),4) * (rijs * rijs
                                                    + riks * riks
                                                    + rjks * rjks));
                              }
                              else
                                  vexp = exp(-SF(attype,SFGmemberlist(attype,groupIndex,l),4) * r2sum);
                              double const plambda = 1.0
                                                   + SF(attype,SFGmemberlist(attype,groupIndex,l),5) * costijk;
                              double fg = vexp;
                              if (plambda <= 0.0) fg = 0.0;
                              else
                              {
                                  fg *= pow(plambda, (SF(attype,SFGmemberlist(attype,groupIndex,l),6) - 1.0));
                              }
                              G(i,SFGmemberlist(attype,groupIndex,l)) += fg * plambda * pfc;

                            } // l
                        } // rjk <= rc
                    } // rik <= rc
                } // elem
            } // k
        } // rij <= rc
    } // j

    double raw_value = 0.0;
    //printf("Raw values: ");
    for (size_t k = 0; k < size; ++k)
    {
        //printf("%f ", G(i,SFGmemberlist(attype,groupIndex,k)));
        raw_value = G(i,SFGmemberlist(attype,groupIndex,k)) * pow(2,(1-SF(attype,k,6))); 
        G(i,SFGmemberlist(attype,groupIndex,k)) = scale(attype, raw_value, SFGmemberlist(attype,groupIndex,k), SFscaling);
    }
    //printf("\n");
    return;
}



KOKKOS_INLINE_FUNCTION void Mode::calculateSFGRD(System* s, AoSoA_NNP nnp_data, d_t_SF SF, d_t_SFscaling SFscaling, d_t_SFGmemberlist SFGmemberlist, t_dGdr dGdr, int attype, int groupIndex, t_verletlist_full_2D neigh_list, T_INT i)
{
    auto x = Cabana::slice<Positions>(s->xvf);
    auto id = Cabana::slice<IDs>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto G = Cabana::slice<NNPNames::G>(nnp_data);
  
    //pick e1, rc from first member (since they are all the same)
    //SFGmemberlist(attype,groupIndex,0): second index = groupIndex, third index  = for first SF in group
    int e1 = SF(attype, SFGmemberlist(attype,groupIndex,0), 2);
    double rc = SF(attype, SFGmemberlist(attype,groupIndex,0), 7);
    //CutoffFunction fc(rc);
    
    int size;
    for (int l=0; l < MAX_SF; ++l)
    {
        if (SFGmemberlist(attype,groupIndex,l) == 0 && SFGmemberlist(attype,groupIndex,l+1) == 0)
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
        //Atom::Neighbor& n = atom.neighbors[j];
        int j = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, jj);
        //std::cout << "i: " << id(i) << " j: " << id(j) << std::endl; 
        //std::cout << "xi: " << x(i,0) << " " << x(i,1) << " " << x(i,2) << std::endl; 
        //std::cout << "xj: " << x(j,0) << " " << x(j,1) << " " << x(j,2) << std::endl; 
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
            //double const* const d1 = n.dr.r;
            //TODO: use subview for size calculation
            //auto SFGmemberlist_subview = Kokkos::subview(SFGmemberlist, make_pair(0,1), make_pair(0,Kokkos::ALL));
            for (size_t k = 0; k < size; ++k) 
            {
                double eta = SF(attype, SFGmemberlist(attype,groupIndex,k), 4);
                double rs = SF(attype, SFGmemberlist(attype,groupIndex,k), 8);
                double pexp = exp(-eta * (rij - rs) * (rij - rs));
                //if (k == 0)
                //std::cout << "member: " << SFGmemberlist(attype,groupIndex,k) << " " ;
                //std::cout << eta << " " << rs << " " << rij << " " << pexp << " " << pfc << " " << pexp*pfc << std::endl;
                G(i,SFGmemberlist(attype,groupIndex,k)) += pexp*pfc;
                // Force calculation.
                double const p1 = SFscaling(attype,k,6) * (pdfc - 2.0 * eta * (rij - rs) * pfc) * pexp / rij;
                // SIMPLE EXPRESSIONS:
                //Vec3D const dij = p1 * atom.neighbors[j].dr;
                //double const p1drijx = p1 * d1[0];
                //double const p1drijy = p1 * d1[1];
                //double const p1drijz = p1 * d1[2];

                // Save force contributions in Atom storage.
                size_t const ki = SFGmemberlist(attype,groupIndex,k);
                // SIMPLE EXPRESSIONS:
                //atom.dGdr[ki]              += dij;
                //atom.neighbors[j].dGdr[ki] -= dij;
                dGdr(i,ki,1) += (p1*dyij);
                dGdr(i,ki,2) += (p1*dzij);

                dGdr(j,ki,0) -= (p1*dxij);
                dGdr(j,ki,1) -= (p1*dyij);
                dGdr(j,ki,2) -= (p1*dzij);
            }
        }
    }

    return;
}


KOKKOS_INLINE_FUNCTION void Mode::calculateSFGAND(System* s, AoSoA_NNP nnp_data, d_t_SF SF, d_t_SFscaling SFscaling, d_t_SFGmemberlist SFGmemberlist, t_dGdr dGdr, int attype, int groupIndex, t_verletlist_full_2D neigh_list, T_INT i) 
{
    auto x = Cabana::slice<Positions>(s->xvf);
    auto id = Cabana::slice<IDs>(s->xvf);
    auto type = Cabana::slice<Types>(s->xvf);
    auto G = Cabana::slice<NNPNames::G>(nnp_data);
 
    //pick e1, rc from first member (since they are all the same)
    //SFGmemberlist(1,0): first index for angular narrow, second index for first SF in angular narrow group
    int e1 = SF(attype, SFGmemberlist(attype,groupIndex,0), 2);
    int e2 = SF(attype, SFGmemberlist(attype,groupIndex,0), 3);
    double rc = SF(attype, SFGmemberlist(attype,groupIndex,0), 7);
    //CutoffFunction fc(rc);
    
    int size;
    for (int l=0; l < MAX_SF; ++l)
    {
        if (SFGmemberlist(attype,groupIndex,l) == 0 && SFGmemberlist(attype,groupIndex,l+1) == 0)
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
        //Atom::Neighbor& nj = atom.neighbors[j];
        int j = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, jj);
        size_t const nej = type(j);
        //std::cout << "i: " << id(i) << " j: " << id(j) << std::endl; 
        //std::cout << "xi: " << x(i,0) << " " << x(i,1) << " " << x(i,2) << std::endl; 
        //std::cout << "xj: " << x(j,0) << " " << x(j,1) << " " << x(j,2) << std::endl; 
        
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
            // SIMPLE EXPRESSIONS:
            //Vec3D const drij(atom.neighbors[j].dr);
            //double const* const dr1 = drij.r;

            for (size_t kk = jj + 1; kk < numNeighbors; kk++)
            {
                int k = Cabana::NeighborList<t_verletlist_full_2D>::getNeighbor(neigh_list, i, kk);
                //Atom::Neighbor& nk = atom.neighbors[k];
                size_t const nek = type(k);
                
                if ((e1 == nej && e2 == nek) ||
                    (e2 == nej && e1 == nek))
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
                        // SIMPLE EXPRESSIONS:
                        //Vec3D const drjk(atom.neighbors[k].dr
                        //               - atom.neighbors[j].dr);
                        //double rjk = drjk.norm2();
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
                            //fc.fdf(rjk, pfcjk, pdfcjk);

                            // SIMPLE EXPRESSIONS:
                            //Vec3D const drik(atom.neighbors[k].dr);
                            //double const* const dr2 = drik.r;
                            //double const* const dr3 = drjk.r;
                            double const rinvijik = 1.0 / rij / rik;
                            // SIMPLE EXPRESSIONS:
                            //double const costijk = (drij * drik) * rinvijik;
                            double const costijk = (dxij*dxik + dyij*dyik + dzij*dzik)* rinvijik;
                            double const pfc = pfcij * pfcik * pfcjk;
                            double const r2sum = r2ij + r2ik + r2jk;
                            double const pr1 = pfcik * pfcjk * pdfcij / rij;
                            double const pr2 = pfcij * pfcjk * pdfcik / rik;
                            double const pr3 = pfcij * pfcik * pdfcjk / rjk;
                            double vexp = 0.0;
                            double rijs = 0.0;
                            double riks = 0.0;
                            double rjks = 0.0;
                            double rs, eta, lambda, zeta;
                            for (size_t l = 0; l < size; ++l)
                            {
                              rs = SF(attype,SFGmemberlist(attype,groupIndex,l),8);
                              eta = SF(attype,SFGmemberlist(attype,groupIndex,l),4);
                              lambda = SF(attype,SFGmemberlist(attype,groupIndex,l),5);
                              zeta = SF(attype,SFGmemberlist(attype,groupIndex,l),6);
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
                              
                              fg *= pow(2,(1-zeta)) * SFscaling(attype,SFGmemberlist(attype,groupIndex,l),6);
                              double const pfczl = pfc * zeta * lambda; 
                              double factorDeriv = 2.0 * eta / zeta / lambda;
                              double const p2etapl = plambda * factorDeriv;
                              double p1;
                              double p2;
                              double p3;
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
                              // Save force contributions in Atom storage.
                              //
                              // SIMPLE EXPRESSIONS:
                              //size_t const li = members[l]->getIndex();
                              //atom.dGdr[li] += p1 * drij + p2 * drik;
                              //atom.neighbors[j].dGdr[li] -= p1 * drij
                              //                            + p3 * drjk;
                              //atom.neighbors[k].dGdr[li] -= p2 * drik
                              //                            - p3 * drjk;
                              /*
                              double const p1drijx = p1 * dxij;
                              double const p1drijy = p1 * dyij;
                              double const p1drijz = p1 * dzij;

                              double const p2drikx = p2 * dxik;
                              double const p2driky = p2 * dyik;
                              double const p2drikz = p2 * dzik;

                              double const p3drjkx = p3 * dxjk;
                              double const p3drjky = p3 * dyjk;
                              double const p3drjkz = p3 * dzjk;*/
                              
                              size_t const li = SFGmemberlist(attype,groupIndex,l);
                              dGdr(i,li,0) += (p1*dxij + p2*dxik);
                              dGdr(i,li,1) += (p1*dyij + p2*dyik);
                              dGdr(i,li,2) += (p1*dzij + p2*dzik);

                              dGdr(j,li,0) -= (p1*dxij + p3*dxjk);
                              dGdr(j,li,1) -= (p1*dyij + p3*dyjk);
                              dGdr(j,li,2) -= (p1*dzij + p3*dzjk);

                              dGdr(k,li,0) -= (p2*dxik - p3*dxjk);
                              dGdr(k,li,1) -= (p2*dyik - p3*dyjk);
                              dGdr(k,li,2) -= (p2*dzik - p3*dzjk);
                            
                            } // l
                        } // rjk <= rc
                    } // rik <= rc
                } // elem
            } // k
        } // rij <= rc
    } // j

    return;
}


}

#endif
