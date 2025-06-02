"""
Script modified from: 
    G. S. Deshmukh, “ChemGCN: A Graph Convolutional Network for Chemical Property
    Prediction,” GitHub repository, 2020. Available: https://github.com/gauravsdeshmukh/ChemGCN
"""
import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem import rdmolops
from torch.utils.data import Dataset

class Graph:  
    def __init__(
        self,
        molecule_smiles: str,
        node_vec_len: int,
        atom_types_dict: dict,
        aromaticity_dict: dict,
        hybridization_dict: dict,
        formal_charge_dict: dict,
        ea_dict: dict,
        ip_dict: dict,
        atomic_mass_dict: dict,
        total_valence_e_dict: dict,
        en_dict: dict,
        atomic_radius_dict: dict,
        max_atoms: int = None
    ):
        self.smiles = molecule_smiles
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.atom_types_len = len(atom_types_dict)
        self.atom_types_dict = atom_types_dict
        self.aromaticity_dict = aromaticity_dict
        self.hybridization_dict = hybridization_dict
        self.formal_charge_dict = formal_charge_dict
        self.ea_dict = ea_dict
        self.ip_dict = ip_dict
        self.atomic_mass_dict = atomic_mass_dict
        self.total_valence_e_dict = total_valence_e_dict
        self.en_dict = en_dict
        self.atomic_radius_dict = atomic_radius_dict

        self.smiles_to_mol()
        if self.mol is not None:
            self.smiles_to_graph()

    def smiles_to_mol(self):
        if self.smiles is None:
            raise ValueError("Error: Encountered None for a SMILES string.")
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            self.mol = None
            raise ValueError(f"Error: RDKit could not convert SMILES '{self.smiles}'.")
        self.mol = Chem.AddHs(mol)
        Chem.SetHybridization(self.mol)
        AllChem.ComputeGasteigerCharges(self.mol)

    def smiles_to_graph(self):
        atoms = self.mol.GetAtoms()

        # Handle max_atoms
        if self.max_atoms is None:
            n_atoms = len(list(atoms))
        else:
            n_atoms = self.max_atoms

        # Create empty node matrix
        node_mat = np.zeros((n_atoms, self.node_vec_len))

        # Fill node_mat
        for atom in atoms:
            atom_index = atom.GetIdx()
            atom_no = atom.GetAtomicNum()
            # Atom type (one-hot)
            atom_type_vec = np.zeros(self.atom_types_len)
            atom_type_vec[self.atom_types_dict.get(str(atom_no))] = 1

            # Aromaticity
            atom_aromaticity = str(atom.GetIsAromatic())
            atom_aromaticity_scalar = int(self.aromaticity_dict.get(atom_aromaticity, 0))

            # Hybridization
            atom_hybridization = atom.GetHybridization().name
            atom_hybridization_vec = np.zeros(len(self.hybridization_dict))
            atom_hybridization_vec[self.hybridization_dict.get(atom_hybridization)] = 1

            # Formal charge
            atom_formal_charge = str(atom.GetFormalCharge())
            atom_formal_charge_vec = np.zeros(len(self.formal_charge_dict))
            atom_formal_charge_vec[self.formal_charge_dict.get(atom_formal_charge)] = 1

            # Electron affinity
            atom_ea = self.ea_dict.get(str(atom_no), 0.0)

            # Ionization potential
            atom_ip = self.ip_dict.get(str(atom_no), 0.0)

            # Gasteiger charge
            atom_charge = float(atom.GetProp('_GasteigerCharge'))

            # Atomic mass
            atom_mass = self.atomic_mass_dict.get(str(atom_no), 0.0)

            # Total valence electrons
            total_valence = str(atom.GetTotalValence())
            total_valence_norm = self.total_valence_e_dict.get(total_valence, 0.0)

            # Electronegativity
            atom_en = self.en_dict.get(str(atom_no), 0.0)

            # Atomic radius
            atomic_radius = self.atomic_radius_dict.get(str(atom_no), 0.0)

            feature_vector = np.concatenate([
                atom_type_vec,
                [atom_aromaticity_scalar],
                atom_hybridization_vec,
                atom_formal_charge_vec,
                [atom_ea],
                [atom_ip],
                [atom_charge],
                [atom_mass],
                [total_valence_norm],
                [atom_en],
                [atomic_radius]
            ])

            node_mat[atom_index] = feature_vector

        # Get adjacency matrix
        adj_mat = rdmolops.GetAdjacencyMatrix(self.mol).astype(float)

        dist_mat = molDG.GetMoleculeBoundsMatrix(self.mol)
        dist_mat[dist_mat == 0.0] = 1.0  # avoid division by zero
        adj_mat = adj_mat * (1 / dist_mat)

        # Pad adjacency matrix if needed
        dim_add = n_atoms - adj_mat.shape[0]
        adj_mat = np.pad(adj_mat, pad_width=((0, dim_add), (0, dim_add)), mode="constant")

        # Add self-loops
        adj_mat = adj_mat + np.eye(n_atoms)

        # Save
        self.node_mat = node_mat
        self.adj_mat = adj_mat


class GraphData(Dataset):
    def __init__(
        self,
        dataset_df: pd.DataFrame,
        node_vec_len: int,
        max_atoms: int,
        atom_types_dict: dict,
        aromaticity_dict: dict,
        hybridization_dict: dict,
        formal_charge_dict: dict,
        ea_dict: dict,
        ip_dict: dict,
        atomic_mass_dict: dict,
        total_valence_e_dict: dict,
        en_dict: dict,
        atomic_radius_dict: dict
    ):
        # Store attributes
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.atom_types_len = len(atom_types_dict)
        self.atom_types_dict = atom_types_dict
        self.aromaticity_dict = aromaticity_dict
        self.hybridization_dict = hybridization_dict
        self.formal_charge_dict = formal_charge_dict
        self.ea_dict = ea_dict
        self.ip_dict = ip_dict
        self.atomic_mass_dict = atomic_mass_dict
        self.total_valence_e_dict = total_valence_e_dict
        self.en_dict = en_dict
        self.atomic_radius_dict = atomic_radius_dict

        df = dataset_df

        # Create index lists / dictionaries
        self.indices = df.index.to_list()
        self.smiles = df["smiles"].to_dict()
        self.outputs = df["solv_oxidation_potential"].to_dict()

        # === NEW: Precompute node_mat, adj_mat, and output for each sample ===
        self.precomputed = {}  # dict to store (node_mat, adj_mat, output)
        for i in self.indices:
            smile = self.smiles[i]
            try:
                graph = Graph(
                    molecule_smiles=smile,
                    node_vec_len=self.node_vec_len,
                    atom_types_dict=self.atom_types_dict,
                    aromaticity_dict=self.aromaticity_dict,
                    hybridization_dict=self.hybridization_dict,
                    formal_charge_dict=self.formal_charge_dict,
                    ea_dict=self.ea_dict,
                    ip_dict=self.ip_dict,
                    atomic_mass_dict=self.atomic_mass_dict,
                    total_valence_e_dict=self.total_valence_e_dict,
                    en_dict=self.en_dict,
                    atomic_radius_dict=self.atomic_radius_dict,
                    max_atoms=self.max_atoms
                )
                # Convert to torch.Tensor once
                node_mat_tensor = torch.Tensor(graph.node_mat)
                adj_mat_tensor = torch.Tensor(graph.adj_mat)
                output_tensor = torch.Tensor([self.outputs[i]])

                self.precomputed[i] = (node_mat_tensor, adj_mat_tensor, output_tensor)

            except ValueError as e:
                # If RDKit fails, store None
                print(f"[Warning] Could not process SMILES with index {i}: {e}")
                self.precomputed[i] = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: str):
        precomp = self.precomputed[i]
        if precomp is None:
            raise ValueError(f"No precomputed data found for index {i}. Possibly invalid SMILES.")
        node_mat, adj_mat, output = precomp
        smile_str = self.smiles[i]
        return (node_mat, adj_mat), output, smile_str, i  # same structure as before


def collate_graph_dataset(dataset: Dataset):
    node_mats = []
    adj_mats = []
    outputs = []
    smiles = []
    ids = []

    for i in range(len(dataset)):
        (node_mat, adj_mat), output, smile, idx = dataset[i]
        node_mats.append(node_mat)
        adj_mats.append(adj_mat)
        outputs.append(output)
        smiles.append(smile)
        ids.append(idx)

    node_mats_tensor = torch.cat(node_mats, dim=0)
    adj_mats_tensor = torch.cat(adj_mats, dim=0)
    outputs_tensor = torch.stack(outputs, dim=0)

    return (node_mats_tensor, adj_mats_tensor), outputs_tensor, smiles, ids
