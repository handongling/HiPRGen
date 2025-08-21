# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import os
import copy
from itertools import combinations_with_replacement
import time
from pathlib import Path
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Tuple
from monty.serialization import dumpfn, loadfn

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender

import joblauncher
from joblauncher.job_setup import job_setup
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
import numpy as np
from pymatgen.core import Molecule
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from monty.serialization import loadfn, dumpfn
from ase.io.jsonio import decode, encode
import numpy as np
from pymatgen.core import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from scipy.spatial.transform import Rotation as R

import rdkit
from rdkit import Chem
from io import StringIO
from pymatgen.io.xyz import XYZ
import numpy as np

from rdkit import Chem

def get_charge(mol):
    """
    Returns the sum of formal charges for all atoms in an RDKit molecule.
    
    Parameters:
        mol (rdkit.Chem.Mol): The RDKit molecule object.
        
    Returns:
        int: Total formal charge of the molecule.
    """
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def pymatgen_to_rdkit_mol(mol):
    buffer = StringIO(mol.to(fmt='mol'))
    return(Chem.MolFromMolBlock(buffer.getvalue()))

def has_element(mg, elements={"P", "F"}):
    mol_elements = {site.specie.symbol for site in mg.molecule.sites}
    return elements.intersection(mol_elements)

def no_large_imaginary_values(lst, threshold=1000.0):
    """
    Returns True if no imaginary component in the list exceeds the given threshold.

    Parameters:
        lst (list): A list of numbers (real or complex).
        threshold (float): Imaginary threshold magnitude.

    Returns:
        bool: True if all imaginary parts are <= threshold, False otherwise.
    """
    for val in lst:
        if isinstance(val, complex) and abs(val.imag) > threshold:
            return False
    return True

from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.analysis.molecule_structure_comparator import MoleculeStructureComparator
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
import matplotlib.pyplot as plt

def build_molecular_hull(mol_graphs, energies, normalize=False):
    """
    Build a convex hull from molecule graphs and their energies.

    Parameters:
        mol_graphs (List[MoleculeGraph]): List of molecule graphs.
        energies (List[float]): Corresponding energies in eV.
        normalize (bool): Whether to normalize energies per atom.

    Returns:
        phase_diagram (PhaseDiagram): Pymatgen phase diagram object.
    """
    entries = []

    for mol_graph, energy in zip(mol_graphs, energies):
        comp = mol_graph.molecule.composition
        if normalize:
            energy /= comp.num_atoms  # optional normalization per atom
        entry = ComputedEntry(composition=comp, energy=energy)
        entries.append(entry)

    phase_diagram = PhaseDiagram(entries)
    # plotter = PDPlotter(phase_diagram, show_unstable=True)

    # plotter.get_plot()
    # plt.title("Molecular Convex Hull")
    # plt.show()

    return phase_diagram


def random_unit_vector():
    """Generate a random 3D unit vector."""
    vec = np.random.randn(3)
    return vec / np.linalg.norm(vec)

def combine_molecules_with_alignment(mg1: MoleculeGraph, mg2: MoleculeGraph, idx1: int, idx2: int, bond_length: float = 1.5) -> MoleculeGraph:
    """
    Combine two molecules by bonding atom idx1 of mg1 to atom idx2 of mg2.
    Rotates mol2 so that its bonding vector points opposite to mol1's, then positions mol2 along the axis for bonding.
    If a molecule has only one atom, a random vector is used for alignment.

    Args:
        mg1 (MoleculeGraph): First molecule graph.
        mg2 (MoleculeGraph): Second molecule graph.
        idx1 (int): Atom index in mg1 to form bond from.
        idx2 (int): Atom index in mg2 to form bond to.
        bond_length (float): Desired bond length in angstroms.

    Returns:
        MoleculeGraph: Combined MoleculeGraph with bond between idx1 and idx2.
    """
    mol1 = mg1.molecule
    mol2 = mg2.molecule

    pos1 = mol1[idx1].coords
    pos2 = mol2[idx2].coords

    # Get direction vectors
    vec1 = mol1.center_of_mass - pos1 if len(mol1) > 1 else random_unit_vector()
    vec2 = mol2.center_of_mass - pos2 if len(mol2) > 1 else random_unit_vector()

    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)

    # Rotate mol2 so vec2 → -vec1
    target_vec = -vec1
    source_vec = vec2

    try:
        rotation, _ = R.align_vectors([target_vec], [source_vec])
    except Exception:
        print("Warning: Rotation failed. Using identity rotation.")
        rotation = R.identity()

    rotated_coords = rotation.apply(mol2.cart_coords - pos2) + pos2

    # Translate mol2 so that atom idx2 is `bond_length` away from pos1 along -vec1
    new_pos2 = pos1 - bond_length * vec1
    shift = new_pos2 - rotated_coords[idx2]
    shifted_coords = rotated_coords + shift

    # Combine
    new_coords = np.vstack([mol1.cart_coords, shifted_coords])
    new_species = mol1.species + mol2.species
    new_mol = Molecule(new_species, new_coords)

    # Construct new MoleculeGraph
    new_mg = MoleculeGraph.with_empty_graph(new_mol)

    for u, v, d in mg1.graph.edges(data=True):
        new_mg.add_edge(u, v, d)
    offset = len(mol1)
    for u, v, d in mg2.graph.edges(data=True):
        new_mg.add_edge(u + offset, v + offset, d)

    new_mg.add_edge(idx1, idx2 + offset, {'weight': 1.0})
    return new_mg


def combine_mol_graphs(molgraph_1: MoleculeGraph, molgraph_2: MoleculeGraph) -> MoleculeGraph:
    """
    Create a combined MoleculeGraph based on two initial MoleculeGraphs.

    Args:
        molgraph_1 (MoleculeGraph)
        molgraph_2 (MoleculeGraph)

    Returns:
        copy_1 (MoleculeGraph)
    """
    # This isn't strictly necessary, but we center both molecules and shift the second
    # For 3D structure generation, having the two molecules appropriately separated is
    # helpful

    radius_1 = np.amax(molgraph_1.molecule.distance_matrix)
    radius_2 = np.amax(molgraph_2.molecule.distance_matrix)

    copy_1 = copy.deepcopy(molgraph_1)
    copy_1.molecule.translate_sites(list(range(len(molgraph_1.molecule))), -1 * molgraph_1.molecule.center_of_mass)

    copy_2 = copy.deepcopy(molgraph_2)
    copy_2.molecule.translate_sites(
        list(range(len(copy_2.molecule))),
        -1 * copy_2.molecule.center_of_mass + np.array([radius_1 + radius_2 + 1.0, 0.0, 0.0]),
    )

    for site in copy_2.molecule:
        copy_1.insert_node(len(copy_1.molecule), site.specie, site.coords)

    for edge in copy_2.graph.edges():
        side_1 = edge[0] + len(molgraph_1.molecule)
        side_2 = edge[1] + len(molgraph_1.molecule)
        copy_1.add_edge(side_1, side_2)

    copy_1.molecule.set_charge_and_spin(molgraph_1.molecule.charge + molgraph_2.molecule.charge)

    return copy_1


def identify_connectable_heavy_atoms(mol_graphs: List[MoleculeGraph]) -> List[List[int]]:
    """
    Identify the heavy atoms in a molecule that can form additional bonds,
    based on valence rules

    Args:
        mol_graphs (List[MoleculeGraph]): List of initial (fragment) MoleculeGraphs

    Returns:
        heavy_atoms_index_list (List[List[int]]): List of the appropriate indices for
        each molecule graph in mol_graphs.
    """

    bond_max = {"C": 4, "P": 6, "S": 6, "O": 2, "N": 3, "F": 1}

    heavy_atoms_index_list = list()
    for i, mol_graph in enumerate(mol_graphs):
        heavy_atoms_in_mol = list()
        num_atoms = len(mol_graph.molecule)
        for j in range(num_atoms):
            connected_sites = mol_graph.get_connected_sites(j)
            num_connected_sites = len(connected_sites)
            element = str(mol_graph.molecule[j].specie)

            if element in ["Li", "H"]:
                if num_connected_sites == 0 and num_atoms == 1:
                    heavy_atoms_in_mol.append(j)

            else:
                metal_count = 0

                for k, site in enumerate(connected_sites):
                    if str(site.site.specie) == "Li":
                        metal_count += 1

                if num_connected_sites - metal_count < bond_max[element]:
                    heavy_atoms_in_mol.append(j)

        heavy_atoms_index_list.append(heavy_atoms_in_mol)

    return heavy_atoms_index_list

def generate_combinations_fast(mol_graphs: List[MoleculeGraph], charges: List[int], directory: Path, max_size: Optional[int] = None):
    """
    Generate all combination of molecule/atom indices that can participate in recombination
    by looping through all molecule pairs(including a mol and itself) and all connectable
    heavy atoms in each molecule.

    Args:
        mol_graphs (List[MoleculeGraph]): List of initial (fragment) MoleculeGraphs
        directory (Path): Path in which to place the output files
        max_size (Optional[int]): If not None (default), only recombinant molecules
        with less than this number of electrons will be allowed.

    Returns:
        final_list (List[MoleculeGraph]): List of all generated recombinant molecules
    """

    combinations_file = directory / "combinations_test.txt"
    mol_graphs_file = directory / "mol_graphs_recombination.json"
    forbidden_connections = [ frozenset(("Li", "Li")),
                             frozenset(("F", "P")),
                             frozenset(("F", "S")), 
                             frozenset(("F", "F")), 
                             frozenset(("P", "P")),
                             frozenset(("S", "S")),
                            frozenset(("H", "P")),
                            frozenset(("H", "Li")),
                            frozenset(("H", "F")),
                            frozenset(("O", "F"))]
    
    with open(combinations_file.as_posix(), "w") as combos:
        combos.write("mol_1\tatom_1\tmol_2\tatom_2\n")
        final_list = list()
        count = 0
        
        eq_atoms = []
        for graph in mol_graphs:
            if len(graph.molecule) == 1:
                eq_atoms.append([0])
            else:
                pga = PointGroupAnalyzer(graph.molecule)
                eq_atoms.append(pga.get_equivalent_atoms()['eq_sets'])
        heavy_atoms_index_list = identify_connectable_heavy_atoms(mol_graphs)
        heavy_atoms_index_list_sym = []
        for ii, heavy_atoms in enumerate(heavy_atoms_index_list):
            heavy_atoms_index_list_sym.append([atom for atom in heavy_atoms if atom in eq_atoms[ii]])
        num_mols = len(mol_graphs)
        all_mol_pair_index = list(combinations_with_replacement(range(num_mols), 2))
        comp_buckets = {}
        for pair_index in tqdm(all_mol_pair_index):
            mol_graph1 = mol_graphs[pair_index[0]]
            mol_graph2 = mol_graphs[pair_index[1]]

            # total_charge = mol_graph1.molecule.charge + mol_graph2.molecule.charge
            total_charge = charges[pair_index[0]] + charges[pair_index[1]]
            total_electrons = mol_graph1.molecule._nelectrons + mol_graph2.molecule._nelectrons

            if int(total_charge) not in {-2,-1, 0, 1,2}:
                continue

            if max_size is not None:
                if total_electrons > max_size:
                    continue

            heavy_atoms_1 = heavy_atoms_index_list_sym[pair_index[0]]
            heavy_atoms_2 = heavy_atoms_index_list_sym[pair_index[1]]
            
            if len(heavy_atoms_1) == 0 or len(heavy_atoms_2) == 0:
                continue
            else:
                for i, atom1 in enumerate(heavy_atoms_1):
                    for j, atom2 in enumerate(heavy_atoms_2):
                        specie1 = str(mol_graph1.molecule[atom1].specie)
                        specie2 = str(mol_graph2.molecule[atom2].specie)


                        if frozenset({specie1, specie2}) in forbidden_connections:
                            continue
                        
                        combined_mol_graph = combine_mol_graphs(mol_graph1, mol_graph2)
                        combined_mol_graph.add_edge(atom1, atom2 + len(mol_graph1.molecule))

                        match = False
                        
                        #check if the recombination is something in the original fragment list
                        for entry in mol_graphs:
                            if (
                                combined_mol_graph.isomorphic_to(entry)
                                and combined_mol_graph.molecule.charge == entry.molecule.charge
                            ):
                                match = True
                                break
                        if match:
                            continue
                        
                        #check if the recombination is something in the new combinations
                        # index = None
                        comp = tuple(sorted(combined_mol_graph.molecule.composition.element_composition.get_el_amt_dict().items()))
                        
                        if comp not in comp_buckets:
                            comp_buckets[comp] = [(combined_mol_graph, pair_index[0], atom1, pair_index[1], atom2, total_charge)]
                            final_list.append(combined_mol_graph)
                            # index = count
                            count+=1
                        else:
                            # for mol_graph in comp_buckets[comp]:
                            #     if (
                            #     mol_graph[0].isomorphic_to(combined_mol_graph)
                            #     and combined_mol_graph.molecule.charge == mol_graph[0].molecule.charge
                            # ):
                            #         # index = mol_graph[1]
                            #         break
                            
                            # if index is None:
                                # index = count
                            comp_buckets[comp].append((combined_mol_graph, pair_index[0], atom1, pair_index[1], atom2, total_charge))
                            final_list.append(combined_mol_graph)
                            count+=1
                        # combos.write("{}\t{}\t{}\t{}\t{}\n".format(pair_index[0], atom1, pair_index[1], atom2, index))
    
    # dumpfn(final_list, mol_graphs_file.as_posix())

    

    
    return comp_buckets, final_list

def parse_combinations_file(filepath: Path) -> List[Tuple[int, int, int, int, int]]:
    """
    Parse a text file to extract reaction information.

    Args:
       filepath (Path)

    Return:
        reactions (List[Tuple[int, int, int, int, int]]): List of reactions. The five elements
            in each reaction are, in order:
                molecule_1_index
                atom_1_index
                molecule_2_index
                atom_2_index
                product_index

    """
    with open(filepath.as_posix()) as combo_file:
        lines = combo_file.readlines()

        reactions = list()
        # Skip first line - header
        for line in lines[1:]:
            line_parsed = [int(x) for x in line.strip().split("\t")]
            reactions.append(tuple(line_parsed))

        return reactions
def generate_combinations_fast_distinct(mol_graphs: List[MoleculeGraph], charges: List[int], mol_graphs2: List[MoleculeGraph], charges2: List[int], directory: Path, max_size: Optional[int] = None):
    """
    Generate all combination of molecule/atom indices that can participate in recombination
    by looping through all molecule pairs(including a mol and itself) and all connectable
    heavy atoms in each molecule.

    Args:
        mol_graphs (List[MoleculeGraph]): List of initial (fragment) MoleculeGraphs
        directory (Path): Path in which to place the output files
        max_size (Optional[int]): If not None (default), only recombinant molecules
        with less than this number of electrons will be allowed.

    Returns:
        final_list (List[MoleculeGraph]): List of all generated recombinant molecules
    """

    combinations_file = directory / "combinations_test.txt"
    mol_graphs_file = directory / "mol_graphs_recombination.json"
    forbidden_connections = [ frozenset(("Li", "Li")),
                             frozenset(("F", "P")),
                             frozenset(("F", "S")), 
                             frozenset(("F", "F")), 
                             frozenset(("P", "P")),
                             frozenset(("S", "S")),
                            frozenset(("H", "P")),
                            frozenset(("H", "Li")),
                            frozenset(("H", "F")),
                            frozenset(("O", "F"))]
    
    with open(combinations_file.as_posix(), "w") as combos:
        combos.write("mol_1\tatom_1\tmol_2\tatom_2\n")
        final_list = list()
        count = 0
        
        eq_atoms = []
        for graph in mol_graphs:
            if len(graph.molecule) == 1:
                eq_atoms.append([0])
            else:
                pga = PointGroupAnalyzer(graph.molecule)
                eq_atoms.append(pga.get_equivalent_atoms()['eq_sets'])
        heavy_atoms_index_list = identify_connectable_heavy_atoms(mol_graphs)
        heavy_atoms_index_list2 = identify_connectable_heavy_atoms(mol_graphs2)

        heavy_atoms_index_list_sym = []
        for ii, heavy_atoms in enumerate(heavy_atoms_index_list):
            heavy_atoms_index_list_sym.append([atom for atom in heavy_atoms if atom in eq_atoms[ii]])
        heavy_atoms_index_list_sym2 = []
        for ii, heavy_atoms in enumerate(heavy_atoms_index_list2):
            heavy_atoms_index_list_sym2.append([atom for atom in heavy_atoms if atom in eq_atoms[ii]])
        num_mols = len(mol_graphs)
        num_mols2 = len(mol_graphs2)
        all_mol_pair_index = [(i, j) for i in range(num_mols) for j in range(num_mols2)]
        
        comp_buckets = {}
        for pair_index in tqdm(all_mol_pair_index):
            mol_graph1 = mol_graphs[pair_index[0]]
            mol_graph2 = mol_graphs[pair_index[1]]

            # total_charge = mol_graph1.molecule.charge + mol_graph2.molecule.charge
            total_charge = charges[pair_index[0]] + charges2[pair_index[1]]
            total_electrons = mol_graph1.molecule._nelectrons + mol_graph2.molecule._nelectrons

            if int(total_charge) not in {-2,-1, 0, 1,2}:
                continue

            if max_size is not None:
                if total_electrons > max_size:
                    continue

            heavy_atoms_1 = heavy_atoms_index_list_sym[pair_index[0]]
            heavy_atoms_2 = heavy_atoms_index_list_sym2[pair_index[1]]
            
            if len(heavy_atoms_1) == 0 or len(heavy_atoms_2) == 0:
                continue
            else:
                for i, atom1 in enumerate(heavy_atoms_1):
                    for j, atom2 in enumerate(heavy_atoms_2):
                        specie1 = str(mol_graph1.molecule[atom1].specie)
                        specie2 = str(mol_graph2.molecule[atom2].specie)


                        if frozenset({specie1, specie2}) in forbidden_connections:
                            continue
                        
                        combined_mol_graph = combine_mol_graphs(mol_graph1, mol_graph2)
                        combined_mol_graph.add_edge(atom1, atom2 + len(mol_graph1.molecule))

                        match = False
                        
                        #check if the recombination is something in the original fragment list
                        # for entry in mol_graphs:
                        #     if (
                        #         combined_mol_graph.isomorphic_to(entry)
                        #         and combined_mol_graph.molecule.charge == entry.molecule.charge
                        #     ):
                        #         match = True
                        #         break
                        # if match:
                        #     continue
                        
                        #check if the recombination is something in the new combinations
                        # index = None
                        comp = tuple(sorted(combined_mol_graph.molecule.composition.element_composition.get_el_amt_dict().items()))
                        
                        if comp not in comp_buckets:
                            comp_buckets[comp] = [(combined_mol_graph, pair_index[0], atom1, pair_index[1], atom2, total_charge)]
                            final_list.append(combined_mol_graph)
                            # index = count
                            count+=1
                        else:
                            # for mol_graph in comp_buckets[comp]:
                            #     if (
                            #     mol_graph[0].isomorphic_to(combined_mol_graph)
                            #     and combined_mol_graph.molecule.charge == mol_graph[0].molecule.charge
                            # ):
                            #         # index = mol_graph[1]
                            #         break
                            
                            # if index is None:
                                # index = count
                            comp_buckets[comp].append((combined_mol_graph, pair_index[0], atom1, pair_index[1], atom2, total_charge))
                            final_list.append(combined_mol_graph)
                            count+=1
                        # combos.write("{}\t{}\t{}\t{}\t{}\n".format(pair_index[0], atom1, pair_index[1], atom2, index))
    
    # dumpfn(final_list, mol_graphs_file.as_posix())

    

    
    return comp_buckets, final_list

def generate_bondnet_files(
    orig_mol_graphs: List[MoleculeGraph],
    recombinant_mol_graphs: List[MoleculeGraph],
    combinations: List[Tuple[int, int, int, int, int]],
    output_directory: Path,
):
    """
    Generate input files for BonDNet

    Args:
        orig_mol_graphs (List[MoleculeGraph]): List of molecule graphs used to generate the recombinant molecules
        recombinant_mol_graphs (List[MoleculeGraph]): List of recombinant molecule graphs
        combinations (List[Tuple[int, int, int, int, int]]): List of reaction tuples of format
                molecule_1_index
                atom_1_index
                molecule_2_index
                atom_2_index
                product_index
        output_directory (Path)

    Returns:
        None

    """
    total_mol_graphs = orig_mol_graphs + recombinant_mol_graphs

    bookmark = len(orig_mol_graphs)

    with open((output_directory / "reactions.csv").as_posix(), "w") as rxn_file:
        rxn_file.write("reactant,product1,product2\n")
        for ii, combination in enumerate(combinations):
            rxn_file.write("{},{},{}\n".format(bookmark + combination[4], combination[0], combination[2]))

    dumpfn(total_mol_graphs, (output_directory / "mol_graphs.json").as_posix())

def get_graph(molecule):
    graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())
    graph = metal_edge_extender(graph)
    return graph

def self_isomorphic_check(graphs, charges):
    molecule_mg_dict = {}
    for ii, graph in enumerate(graphs):
        comp = sorted(graph.molecule.composition.element_composition.get_el_amt_dict().items())
        comp.append(charges[ii])
        comp = tuple(comp)
        if comp in molecule_mg_dict:
            molecule_mg_dict[comp].append((graph,charges[ii]))
        else:
            molecule_mg_dict[comp]= [(graph,charges[ii])]
    
    final_entries = []
    
    for comp in tqdm(molecule_mg_dict):
        unique = []
        duplicates = []
        for ii, graph in enumerate(molecule_mg_dict[comp]):
            if ii in duplicates:
                continue
            unique.append(graph)
            for graph_compare_index in range(ii+1, len(molecule_mg_dict[comp])):
                # if False:
                if graph_compare_index in duplicates:
                    continue
                if graph[0].isomorphic_to(molecule_mg_dict[comp][graph_compare_index][0]):
                    duplicates.append(graph_compare_index)
    
        final_entries.extend(unique)
    return final_entries

def ref_isomorphic_check(graphs, charges, ref_graphs, ref_charges):
    molecule_mg_dict = {}
    for ii, graph in enumerate(ref_graphs):
        comp = sorted(graph.molecule.composition.element_composition.get_el_amt_dict().items())
        comp.append(charges[ii])
        comp = tuple(comp)
        
        if comp in molecule_mg_dict:
            molecule_mg_dict[comp].append((ref_graph,ref_charges[ii]))
        else:
            molecule_mg_dict[comp]= [(ref_graph,ref_charges[ii])]
    
    final_entries = []

    for ii, graph in tqdm(enumerate(graphs)):
        comp = sorted(graph.molecule.composition.element_composition.get_el_amt_dict().items())
        comp.append(charges[ii])
        comp = tuple(comp)
        unique = []
        match = False
        for ii, ref_graph in enumerate(molecule_mg_dict[comp]):
            if graph.isomorphic_to(ref_graph[0]):
                match = True 
                break
        if match == False:
            final_entries.extend((graph, charges[ii]))
    return final_entries

def energy_filter(graphs, charges, energies, etol = 0.5):
    molecule_mg_dict = {}
    for ii, graph in enumerate(graphs):
        comp = sorted(graph.molecule.composition.element_composition.get_el_amt_dict().items())
        comp.append(charges[ii])
        comp = tuple(comp)
        if comp in molecule_mg_dict:
            molecule_mg_dict[comp].append((graph,charges[ii], energies[ii]))
        else:
            molecule_mg_dict[comp]= [(graph,charges[ii], energies[ii])]

    final_entries = []
    for comp in molecule_mg_dict:
        min_energy = 10000
        for graph in molecule_mg_dict[comp]:
            if graph[2] < min_energy:
                min_energy = graph[2]

        for graph in molecule_mg_dict[comp]:
            if graph[2] - min_energy < etol:
                final_entries.append(graph)
    
    return final_entries
    

from math import ceil
from collections import defaultdict
import networkx as nx
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Molecule
from itertools import combinations
def frequent_substructures_pmg(
    mol_graphs,
    min_k=3,
    max_k=5,
    min_support=0.2,         # float in (0,1] for fraction of molecules, or int >=1
    node_label_fn=None,      # function(site) -> label; default: site.specie.symbol
    top_n=None,
    connected_only=True
):
    """
    Find frequent connected, induced subgraphs (node-labeled; unlabeled edges)
    across a list of pymatgen MoleculeGraph objects. Returns pymatgen MoleculeGraphs
    as exemplars of each frequent pattern.

    Parameters
    ----------
    mol_graphs : list[MoleculeGraph]
    min_k, max_k : int
        Min/max subgraph size (in nodes). Keep max_k small (3-5) for performance.
    min_support : float or int
        If float in (0,1], interpreted as fraction of molecules.
        If int >= 1, interpreted as absolute number of molecules.
    node_label_fn : callable or None
        Function mapping a pymatgen Site -> hashable label. Default is element symbol.
        (Labels define isomorphism equivalence.)
    top_n : int or None
        Return only the top_n patterns by support (then by size desc).
    connected_only : bool
        Only consider connected subgraphs (recommended).

    Returns
    -------
    list[dict] with keys:
        - 'size' : int, number of nodes
        - 'support' : int, number of molecules containing the pattern
        - 'support_frac' : float
        - 'hash' : str, WL hash
        - 'molecule_graph' : MoleculeGraph exemplar of the pattern
    """
    if not mol_graphs:
        return []

    if node_label_fn is None:
        node_label_fn = lambda site: getattr(site.specie, "symbol", str(site.specie))

    n_graphs = len(mol_graphs)
    # Resolve min support threshold
    if isinstance(min_support, float):
        if not (0.0 < min_support <= 1.0):
            raise ValueError("min_support as float must be in (0,1].")
        min_sup_abs = max(1, ceil(min_support * n_graphs))
    elif isinstance(min_support, int):
        if min_support < 1:
            raise ValueError("min_support as int must be >= 1.")
        min_sup_abs = min_support
    else:
        raise TypeError("min_support must be float in (0,1] or int >= 1.")

    # Build simple NetworkX graphs (undirected, no multi-edges) with node labels
    # and keep mapping back to the original MoleculeGraph.
    nx_graphs = []
    for mg in mol_graphs:
        G = nx.Graph()
        mol = mg.molecule
        # Nodes: index with label attr
        for i, site in enumerate(mol.sites):
            G.add_node(i, label=str(node_label_fn(site)))
        # Edges: collapse MultiGraph edges to simple undirected edges
        seen = set()
        for u, v, *_ in mg.graph.edges(keys=True, data=True):
            a, b = (u, v) if u <= v else (v, u)
            if (a, b) not in seen:
                seen.add((a, b))
                G.add_edge(a, b)
        nx_graphs.append(G)

    def wl_hash(H):
        # WL hash using node labels (as strings)
        C = H.copy()
        for n, d in C.nodes(data=True):
            d["label"] = str(d.get("label", ""))
        return nx.weisfeiler_lehman_graph_hash(C, node_attr="label")

    pattern_support_sets = defaultdict(set)   # hash -> set of graph IDs containing it
    exemplars = {}                            # hash -> (gid, tuple(orig_nodes), NX subgraph)
    sizes = {}                                # hash -> size (nodes)

    for gid, G in enumerate(nx_graphs):
        n = G.number_of_nodes()
        if n < min_k:
            continue

        per_graph_hashes = set()

        max_k_eff = min(max_k, n)
        for k in range(min_k, max_k_eff + 1):
            # Enumerate induced k-node subgraphs
            for nodes in combinations(G.nodes, k):
                H = G.subgraph(nodes)
                if connected_only and not nx.is_connected(H):
                    continue
                h = wl_hash(H)
                if h not in per_graph_hashes:
                    per_graph_hashes.add(h)
                    if h not in exemplars:
                        # Store an exemplar with original node indices & subgraph
                        exemplars[h] = (gid, tuple(sorted(nodes)), H.copy())
                        sizes[h] = k

        # Update global support
        for h in per_graph_hashes:
            pattern_support_sets[h].add(gid)

    # Prepare results meeting min support
    results = []
    for h, gids in pattern_support_sets.items():
        sup = len(gids)
        if sup >= min_sup_abs and sizes.get(h, 0) >= min_k:
            gid_ex, orig_nodes, H = exemplars[h]
            parent_mg = mol_graphs[gid_ex]
            parent_mol = parent_mg.molecule

            # Build Molecule for the exemplar (keep real coordinates/species)
            sub_sites = [parent_mol.sites[i] for i in orig_nodes]
            new_mol = Molecule(
                species=[s.specie for s in sub_sites],
                coords=[s.coords for s in sub_sites],
                site_properties={k: [s.properties.get(k) for s in sub_sites]
                                 for k in set().union(*(s.properties.keys() for s in sub_sites))}
                if any(s.properties for s in sub_sites) else None
            )

            # Build MoleculeGraph for the exemplar
            mg_sub = MoleculeGraph.with_empty_graph(new_mol)
            idx_map = {orig: new for new, orig in enumerate(orig_nodes)}
            # Add edges from the NX subgraph (which uses original indices)
            for u, v in H.edges():
                uu, vv = idx_map[u], idx_map[v]
                try:
                    mg_sub.add_edge(uu, vv)  # typical signature
                except TypeError:
                    # Fallbacks for older/newer pymatgen versions
                    try:
                        mg_sub.add_edge(uu, vv, to_jimage=(0, 0, 0), weight=1)
                    except TypeError:
                        mg_sub.add_edge(uu, vv, order=1)

            results.append({
                "size": sizes[h],
                "support": sup,
                "support_frac": sup / n_graphs,
                "hash": h,
                "molecule_graph": mg_sub
            })

    # Sort by support desc, then size desc, then hash for stability
    results.sort(key=lambda r: (-r["support"], -r["size"], r["hash"]))
    if top_n is not None:
        results = results[:top_n]
    return results
