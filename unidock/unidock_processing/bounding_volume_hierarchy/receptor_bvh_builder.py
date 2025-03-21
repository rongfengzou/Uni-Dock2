import os

import numpy as np
from scipy.spatial import KDTree

import MDAnalysis as mda
mda.topology.tables.vdwradii['FE'] = 2.48

import networkx as nx

from rdkit import Chem
from rdkit.Chem import GetMolFrags, FragmentOnBonds

from unidock.unidock_processing.utils.protein_topology import prepare_protein_residue_mol_list
from unidock.unidock_processing.ligand_topology.generic_rotatable_bond import GenericRotatableBond
from unidock.unidock_processing.bounding_volume_hierarchy.utils import construct_oriented_bounding_box, mol2image, create_network_view

class ReceptorBVHBuilder(object):
    def __init__(self,
                 receptor_pdb_file_name,
                 kept_ligand_resname_list=None,
                 target_center=(0.0, 0.0, 0.0),
                 box_size=(22.5, 22.5, 22.5),
                 create_tree_visualization=True,
                 working_dir_name='.'):

        self.receptor_pdb_file_name = os.path.abspath(receptor_pdb_file_name)
        self.kept_ligand_resname_list = kept_ligand_resname_list
        self.target_center = np.array(target_center, dtype=np.float32)
        self.box_size = np.array(box_size, dtype=np.float32)
        self.pocket_radius = np.max(self.box_size) / 2.0

        self.create_tree_visualization = create_tree_visualization
        self.working_dir_name = os.path.abspath(working_dir_name)

        self.rotatable_bond_finder = GenericRotatableBond()

        self.receptor_ag = mda.Universe(self.receptor_pdb_file_name).atoms
        protein_ag = self.receptor_ag.select_atoms('protein')
        protein_pdb_file_name = os.path.join(self.working_dir_name, 'protein.pdb')
        protein_ag.write(protein_pdb_file_name, bonds=None)

        if self.kept_ligand_resname_list is not None:
            self.mda_to_rdkit = mda._CONVERTERS['RDKIT']().convert
            self.small_molecule_mol_list = []

            for ligand_resname in self.kept_ligand_resname_list:
                self.small_molecule_mol_list.extend(self.__split_small_molecules__(ligand_resname))

        #########################################################################################################
        #########################################################################################################
        ## Search for closed pocket residues
        self.protein_mol, self.protein_residue_mol_list = prepare_protein_residue_mol_list(protein_pdb_file_name)

        protein_atom_positions = self.protein_mol.GetConformer().GetPositions().astype(np.float32)
        protein_kdtree = KDTree(protein_atom_positions)
        closed_neighbors_idx_list = protein_kdtree.query_ball_point(self.target_center, self.pocket_radius)
        unique_closed_neighbors_idx_list = list(np.unique(closed_neighbors_idx_list))
        protein_pocket_residue_idx_list = [self.protein_mol.GetAtomWithIdx(int(unique_closed_neighbors_idx)).GetIntProp('internal_residue_idx') for unique_closed_neighbors_idx in unique_closed_neighbors_idx_list]
        self.protein_pocket_residue_idx_list = list(np.sort(np.unique(protein_pocket_residue_idx_list)))
        #########################################################################################################
        #########################################################################################################

    def __build_protein_BVH_tree__(self):
        phi_atoms_pattern_mol = Chem.MolFromSmarts('[N]!@[C@;H1,H2;$(C[C;H0;$(C=O)])]')
        psi_atoms_pattern_mol = Chem.MolFromSmarts('[C@;H1,H2;$(CN)][C;H0;$(C=O)]')
        backbone_pattern_mol = Chem.MolFromSmarts('C(=O)[N;!$(N([H])[H])]')

        phi_matched_atom_idx_list = list(self.protein_mol.GetSubstructMatches(phi_atoms_pattern_mol, maxMatches=100000))
        psi_matched_atom_idx_list = list(self.protein_mol.GetSubstructMatches(psi_atoms_pattern_mol, maxMatches=100000))

        rotatable_bond_atom_idx_list = phi_matched_atom_idx_list + psi_matched_atom_idx_list
        num_rotatable_bonds = len(rotatable_bond_atom_idx_list)
        bond_idx_list = [None] * num_rotatable_bonds

        for rotatable_bond_idx in range(num_rotatable_bonds):
            rotatable_bond_atom_idx_tuple = rotatable_bond_atom_idx_list[rotatable_bond_idx]
            rotatable_bond = self.protein_mol.GetBondBetweenAtoms(rotatable_bond_atom_idx_tuple[0], rotatable_bond_atom_idx_tuple[1])
            bond_idx = rotatable_bond.GetIdx()
            bond_idx_list[rotatable_bond_idx] = bond_idx

        splitted_protein_mol = FragmentOnBonds(self.protein_mol, bond_idx_list, addDummies=False)
        primary_fragment_mol_list = list(GetMolFrags(splitted_protein_mol, asMols=True, sanitizeFrags=False))
        num_primary_fragments = len(primary_fragment_mol_list)

        for primary_fragment_idx in range(num_primary_fragments):
            primary_fragment_mol = primary_fragment_mol_list[primary_fragment_idx]
            Chem.GetSymmSSSR(primary_fragment_mol)
            primary_fragment_mol.UpdatePropertyCache(strict=False)

            #########################################################################################################
            #########################################################################################################
            ## Only collect pocket residues
            pocket_residue_flag = False
            num_primary_fragment_atoms = primary_fragment_mol.GetNumAtoms()
            for atom_idx in range(num_primary_fragment_atoms):
                atom = primary_fragment_mol.GetAtomWithIdx(atom_idx)
                if atom.GetIntProp('internal_residue_idx') in self.protein_pocket_residue_idx_list:
                    pocket_residue_flag = True
                    break

            if not pocket_residue_flag:
                continue

            #########################################################################################################
            #########################################################################################################

            if primary_fragment_mol.HasSubstructMatch(backbone_pattern_mol):
                primary_fragment_mol.SetIntProp('chain_tree_node_idx', self.current_node_idx)
                primary_fragment_coords_array = primary_fragment_mol.GetConformer().GetPositions()

                primary_fragment_obb, primary_fragment_coord_point_list = construct_oriented_bounding_box(primary_fragment_coords_array)
                primary_fragment_obb_center = primary_fragment_obb.Center().Coord()

                self.receptor_chain_tree.add_node(self.current_node_idx,
                                                  mol_type='protein',
                                                  mol=primary_fragment_mol,
                                                  coord_point_list=primary_fragment_coord_point_list,
                                                  obb=primary_fragment_obb,
                                                  obb_center=primary_fragment_obb_center)

                self.receptor_chain_tree.add_edge(0,
                                                  self.current_node_idx,
                                                  parent_node_idx=0,
                                                  children_node_idx=self.current_node_idx)

                if self.create_tree_visualization:
                    self.receptor_chain_tree_visualized.add_node(self.current_node_idx,
                                                                 img=mol2image(primary_fragment_mol),
                                                                 hac=primary_fragment_mol.GetNumAtoms())

                    self.receptor_chain_tree_visualized.add_edge(0,
                                                                 self.current_node_idx,
                                                                 parent_node_idx=0,
                                                                 children_node_idx=self.current_node_idx)

                self.current_node_idx += 1

            else:
                side_chain_rotatable_bond_info_list = self.rotatable_bond_finder.identify_rotatable_bonds(primary_fragment_mol)
                side_chain_bond_list = list(primary_fragment_mol.GetBonds())
                side_chain_rotatable_bond_idx_list = []
                for bond in side_chain_bond_list:
                    bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
                    if bond_info in side_chain_rotatable_bond_info_list or bond_info_reversed in side_chain_rotatable_bond_info_list:
                        side_chain_rotatable_bond_idx_list.append(bond.GetIdx())

                if len(side_chain_rotatable_bond_idx_list) > 0:
                    ## Find cuttable side chain torsions, build side chain nodes with C alpha as parent
                    splitted_primary_fragment_mol = FragmentOnBonds(primary_fragment_mol, side_chain_rotatable_bond_idx_list, addDummies=False)
                    secondary_fragment_mol_list = list(GetMolFrags(splitted_primary_fragment_mol, asMols=True, sanitizeFrags=False))
                    num_secondary_fragments = len(secondary_fragment_mol_list)

                    ## Find C alpha group index
                    c_alpha_group_idx = None
                    for secondary_fragment_idx in range(num_secondary_fragments):
                        if c_alpha_group_idx is not None:
                            break
                        else:
                            secondary_fragment_mol = secondary_fragment_mol_list[secondary_fragment_idx]
                            for atom in secondary_fragment_mol.GetAtoms():
                                if atom.GetProp('atom_name') == 'CA':
                                    c_alpha_group_idx = secondary_fragment_idx
                                    break

                    if c_alpha_group_idx is None:
                        raise ValueError('Bugs in C alpha finding codes or problematic protein mol.')

                    ## Assign fragments as nodes to chain tree
                    c_alpha_fragment_mol =  secondary_fragment_mol_list[c_alpha_group_idx]
                    c_alpha_fragment_mol.SetIntProp('chain_tree_node_idx', self.current_node_idx)
                    c_alpha_fragment_coords_array = c_alpha_fragment_mol.GetConformer().GetPositions()

                    c_alpha_fragment_obb, c_alpha_fragment_coord_point_list = construct_oriented_bounding_box(c_alpha_fragment_coords_array)
                    c_alpha_fragment_obb_center = c_alpha_fragment_obb.Center().Coord()

                    self.receptor_chain_tree.add_node(self.current_node_idx,
                                                      mol_type='protein',
                                                      mol=c_alpha_fragment_mol,
                                                      coord_point_list=c_alpha_fragment_coord_point_list,
                                                      obb=c_alpha_fragment_obb,
                                                      obb_center=c_alpha_fragment_obb_center)

                    self.receptor_chain_tree.add_edge(0,
                                                      self.current_node_idx,
                                                      parent_node_idx=0,
                                                      children_node_idx=self.current_node_idx)

                    if self.create_tree_visualization:
                        self.receptor_chain_tree_visualized.add_node(self.current_node_idx,
                                                                     img=mol2image(c_alpha_fragment_mol),
                                                                     hac=c_alpha_fragment_mol.GetNumAtoms())

                        self.receptor_chain_tree_visualized.add_edge(0,
                                                                     self.current_node_idx,
                                                                     parent_node_idx=0,
                                                                     children_node_idx=self.current_node_idx)

                    c_alpha_node_idx = self.current_node_idx
                    self.current_node_idx += 1

                    for secondary_fragment_idx in range(num_secondary_fragments):
                        if secondary_fragment_idx == c_alpha_group_idx:
                            continue

                        side_chain_fragment_mol =  secondary_fragment_mol_list[secondary_fragment_idx]
                        side_chain_fragment_mol.SetIntProp('chain_tree_node_idx', self.current_node_idx)
                        side_chain_fragment_coords_array = side_chain_fragment_mol.GetConformer().GetPositions()

                        side_chain_fragment_obb, side_chain_fragment_coord_point_list = construct_oriented_bounding_box(side_chain_fragment_coords_array)
                        side_chain_fragment_obb_center = side_chain_fragment_obb.Center().Coord()

                        self.receptor_chain_tree.add_node(self.current_node_idx,
                                                          mol_type='protein',
                                                          mol=side_chain_fragment_mol,
                                                          coord_point_list=side_chain_fragment_coord_point_list,
                                                          obb=side_chain_fragment_obb,
                                                          obb_center=side_chain_fragment_obb_center)

                        self.receptor_chain_tree.add_edge(c_alpha_node_idx,
                                                          self.current_node_idx,
                                                          parent_node_idx=c_alpha_node_idx,
                                                          children_node_idx=self.current_node_idx)

                        if self.create_tree_visualization:
                            self.receptor_chain_tree_visualized.add_node(self.current_node_idx,
                                                                         img=mol2image(side_chain_fragment_mol),
                                                                         hac=side_chain_fragment_mol.GetNumAtoms())

                            self.receptor_chain_tree_visualized.add_edge(c_alpha_node_idx,
                                                                         self.current_node_idx,
                                                                         parent_node_idx=c_alpha_node_idx,
                                                                         children_node_idx=self.current_node_idx)

                        self.current_node_idx += 1

                else:
                    ## No cuttable torsions, group all side chain as one group.
                    primary_fragment_mol.SetIntProp('chain_tree_node_idx', self.current_node_idx)
                    primary_fragment_coords_array = primary_fragment_mol.GetConformer().GetPositions()

                    primary_fragment_obb, primary_fragment_coord_point_list = construct_oriented_bounding_box(primary_fragment_coords_array)
                    primary_fragment_obb_center = primary_fragment_obb.Center().Coord()

                    self.receptor_chain_tree.add_node(self.current_node_idx,
                                                      mol_type='protein',
                                                      mol=primary_fragment_mol,
                                                      coord_point_list=primary_fragment_coord_point_list,
                                                      obb=primary_fragment_obb,
                                                      obb_center=primary_fragment_obb_center)

                    self.receptor_chain_tree.add_edge(0,
                                                      self.current_node_idx,
                                                      parent_node_idx=0,
                                                      children_node_idx=self.current_node_idx)

                    if self.create_tree_visualization:
                        self.receptor_chain_tree_visualized.add_node(self.current_node_idx,
                                                                     img=mol2image(primary_fragment_mol),
                                                                     hac=primary_fragment_mol.GetNumAtoms())

                        self.receptor_chain_tree_visualized.add_edge(0,
                                                                     self.current_node_idx,
                                                                     parent_node_idx=0,
                                                                     children_node_idx=self.current_node_idx)

                    self.current_node_idx += 1

    def __split_small_molecules__(self, ligand_resname):
        molecule_ag = self.receptor_ag.select_atoms(f'resname {ligand_resname}')
        num_residues = molecule_ag.n_residues
        mol_list = [None] * num_residues

        for residue_idx in range(num_residues):
            res = molecule_ag.residues[residue_idx]
            mol_universe = mda.Merge(res.atoms)
            mol = self.mda_to_rdkit(mol_universe, NoImplicit=False)
            mol.SetProp('residue_name', ligand_resname)
            mol_list[residue_idx] = mol

        return mol_list

    def __build_small_molecule_BVH_tree__(self):
        for mol in self.small_molecule_mol_list:

            #########################################################################################################
            #########################################################################################################
            ## Check if this molecule is closed to pocket center
            mol_atom_positions = mol.GetConformer().GetPositions().astype(np.float32)
            mol_kdtree = KDTree(mol_atom_positions)
            closed_neighbors_idx_list = mol_kdtree.query_ball_point(self.target_center, self.pocket_radius)
            if len(closed_neighbors_idx_list) == 0:
                continue
            #########################################################################################################
            #########################################################################################################

            if mol.GetProp('residue_name') in ['HOH']:
                mol_h = Chem.AddHs(mol, addCoords=True)
                mol_h.SetIntProp('chain_tree_node_idx', self.current_node_idx)
                mol_coords_array = mol_h.GetConformer().GetPositions()

                mol_obb, mol_coord_point_list = construct_oriented_bounding_box(mol_coords_array)
                mol_obb_center = mol_obb.Center().Coord()

                self.receptor_chain_tree.add_node(self.current_node_idx,
                                                  mol_type='small_molecule',
                                                  mol=mol_h,
                                                  coord_point_list=mol_coord_point_list,
                                                  obb=mol_obb,
                                                  obb_center=mol_obb_center)

                self.receptor_chain_tree.add_edge(0,
                                                  self.current_node_idx,
                                                  parent_node_idx=0,
                                                  children_node_idx=self.current_node_idx)

                if self.create_tree_visualization:
                    self.receptor_chain_tree_visualized.add_node(self.current_node_idx,
                                                                 img=mol2image(mol_h),
                                                                 hac=mol_h.GetNumAtoms())

                    self.receptor_chain_tree_visualized.add_edge(0,
                                                                 self.current_node_idx,
                                                                 parent_node_idx=0,
                                                                 children_node_idx=self.current_node_idx)

                self.current_node_idx += 1

            else:
                mol_rotatable_bond_info_list = self.rotatable_bond_finder.identify_rotatable_bonds(mol)
                mol_bond_list = list(mol.GetBonds())
                mol_rotatable_bond_idx_list = []
                for bond in mol_bond_list:
                    bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
                    if bond_info in mol_rotatable_bond_info_list or bond_info_reversed in mol_rotatable_bond_info_list:
                        mol_rotatable_bond_idx_list.append(bond.GetIdx())

                if len(mol_rotatable_bond_idx_list) > 0:
                    splitted_fragment_mol = FragmentOnBonds(mol, mol_rotatable_bond_idx_list, addDummies=False)
                    fragment_mol_list = list(GetMolFrags(splitted_fragment_mol, asMols=True, sanitizeFrags=False))
                else:
                    fragment_mol_list = [mol]

                for fragment_mol in fragment_mol_list:
                    fragment_mol.SetIntProp('chain_tree_node_idx', self.current_node_idx)
                    fragment_coords_array = fragment_mol.GetConformer().GetPositions()

                    fragment_obb, fragment_coord_point_list = construct_oriented_bounding_box(fragment_coords_array)
                    fragment_obb_center = fragment_obb.Center().Coord()

                    self.receptor_chain_tree.add_node(self.current_node_idx,
                                                      mol_type='small_molecule',
                                                      mol=fragment_mol,
                                                      coord_point_list=fragment_coord_point_list,
                                                      obb=fragment_obb,
                                                      obb_center=fragment_obb_center)

                    self.receptor_chain_tree.add_edge(0,
                                                      self.current_node_idx,
                                                      parent_node_idx=0,
                                                      children_node_idx=self.current_node_idx)

                    if self.create_tree_visualization:
                        self.receptor_chain_tree_visualized.add_node(self.current_node_idx,
                                                                     img=mol2image(fragment_mol),
                                                                     hac=fragment_mol.GetNumAtoms())

                        self.receptor_chain_tree_visualized.add_edge(0,
                                                                     self.current_node_idx,
                                                                     parent_node_idx=0,
                                                                     children_node_idx=self.current_node_idx)

                    self.current_node_idx += 1

    def build_whole_receptor_BVH_tree(self):
        self.receptor_chain_tree = nx.Graph()
        self.receptor_chain_tree_visualized = nx.Graph()
        self.current_node_idx = 0

        root_node_info_dict = {}
        root_node_info_dict['protein_mol'] = self.protein_mol
        root_node_info_dict['chain_tree_node_idx'] = self.current_node_idx

        self.receptor_chain_tree.add_node(self.current_node_idx, node_info_dict=root_node_info_dict)

        if self.create_tree_visualization:
            self.receptor_chain_tree_visualized.add_node(self.current_node_idx, img=mol2image(self.protein_mol), hac=self.protein_mol.GetNumAtoms())

        self.current_node_idx += 1

        self.__build_protein_BVH_tree__()
        self.__build_small_molecule_BVH_tree__()

    def visualize_receptor_chain_tree(self):
        return create_network_view(self.receptor_chain_tree_visualized,
                                   color_mapper='',
                                   scale_factor=30,
                                   to_undirected=True,
                                   layout='preset')
