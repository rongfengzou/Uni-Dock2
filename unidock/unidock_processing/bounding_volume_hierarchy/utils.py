import typing as t
from urllib import parse
import ipycytoscape

import numpy as np
from numba import jit
import networkx as nx

from rdkit.Chem import GetMolFrags, FragmentOnBonds
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import PrepareMolForDrawing

try:
    from OCP.TColgp import TColgp_Array1OfPnt
    from OCP.TColStd import TColStd_Array1OfReal
    from OCP.gp import gp_Pnt
    from OCP.Bnd import Bnd_OBB
except:
    print("Warning: import OCP failed")

def point_list_to_TColgp_Array1OfPnt(li):
    pts = TColgp_Array1OfPnt(0, len(li) - 1)
    for n, i in enumerate(li):
        pts.SetValue(n, gp_Pnt(*i))
    return pts

def point_list_to_TColStd_Array1OfReal(num_atoms, tolerance):
    pts = TColStd_Array1OfReal(0, num_atoms - 1)
    for atom_idx in range(num_atoms):
        pts.SetValue(atom_idx, tolerance)
    return pts

def construct_oriented_bounding_box(coords_array, tolerance=None):
    num_atoms = coords_array.shape[0]
    coord_point_list = [None] * num_atoms
    for atom_idx in range(num_atoms):
        coord_point_list[atom_idx] = gp_Pnt(*coords_array[atom_idx, :])

    obb_points = point_list_to_TColgp_Array1OfPnt(coords_array)

    if tolerance is not None:
        obb_tolerances = point_list_to_TColStd_Array1OfReal(num_atoms, tolerance)
    else:
        obb_tolerances = None

    obb = Bnd_OBB()
    obb.ReBuild(theListOfPoints=obb_points,
                theListOfTolerances=obb_tolerances,
                theIsOptimal=True)

    return obb, coord_point_list

def construct_oriented_bounding_box_list(fragment_mol, tolerance=None):
    num_atoms = fragment_mol.GetNumAtoms()
    for atom_idx in range(num_atoms):
        atom = fragment_mol.GetAtomWithIdx(atom_idx)
        atom.SetIntProp('fragment_atom_idx', atom_idx)

    ring_info = fragment_mol.GetRingInfo()
    ring_atom_idx_list = list(ring_info.AtomRings())

    num_unit_rings = len(ring_atom_idx_list)
    identified_ring_atom_idx_nested_list = [None] * num_unit_rings
    cuttable_bond_idx_list = []

    for unit_ring_idx in range(num_unit_rings):
        ring_atom_idx_tuple = ring_atom_idx_list[unit_ring_idx]
        identified_ring_atom_idx_list = list(ring_atom_idx_tuple)
        for atom_idx in ring_atom_idx_tuple:
            atom = fragment_mol.GetAtomWithIdx(atom_idx)
            for neighbor_atom in atom.GetNeighbors():
                if neighbor_atom.IsInRing():
                    continue
                else:
                    cuttable_bond = fragment_mol.GetBondBetweenAtoms(atom_idx, neighbor_atom.GetIdx())
                    cuttable_bond_idx = cuttable_bond.GetIdx()
                    if cuttable_bond_idx not in cuttable_bond_idx_list:
                        cuttable_bond_idx_list.append(cuttable_bond_idx)

        identified_ring_atom_idx_nested_list[unit_ring_idx] = identified_ring_atom_idx_list

    ####################################################################
    ## deal with phosphorus groups: always too large for obb
    for atom_idx in range(num_atoms):
        atom = fragment_mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'P':
            for neighbor_atom in atom.GetNeighbors():
                cuttable_bond = fragment_mol.GetBondBetweenAtoms(atom_idx, neighbor_atom.GetIdx())
                if not cuttable_bond.IsInRing():
                    cuttable_bond_idx = cuttable_bond.GetIdx()
                    if cuttable_bond_idx not in cuttable_bond_idx_list:
                        cuttable_bond_idx_list.append(cuttable_bond_idx)
    ####################################################################

    if len(cuttable_bond_idx_list) > 0:
        splitted_mol = FragmentOnBonds(fragment_mol, cuttable_bond_idx_list, addDummies=False)
        splitted_mol_list = list(GetMolFrags(splitted_mol, asMols=True, sanitizeFrags=False))
    else:
        splitted_mol_list = [fragment_mol]

    total_ring_atom_idx_set = set()
    for identified_ring_atom_idx_list in identified_ring_atom_idx_nested_list:
        for ring_atom_idx in identified_ring_atom_idx_list:
            total_ring_atom_idx_set.add(ring_atom_idx)

    total_ring_atom_idx_list = list(total_ring_atom_idx_set)

    fragment_atom_idx_nested_list = []
    for splitted_mol in splitted_mol_list:
        ring_fragment_flag = False
        for atom in splitted_mol.GetAtoms():
            if atom.GetIntProp('fragment_atom_idx') in total_ring_atom_idx_list:
                ring_fragment_flag = True
                break

        if not ring_fragment_flag:
            num_fragment_atoms = splitted_mol.GetNumAtoms()
            fragment_atom_idx_list = [None] * num_fragment_atoms
            for fragment_atom_idx in range(num_fragment_atoms):
                fragment_atom = splitted_mol.GetAtomWithIdx(fragment_atom_idx)
                fragment_atom_idx_list[fragment_atom_idx] = fragment_atom.GetIntProp('fragment_atom_idx')

            fragment_atom_idx_nested_list.append(fragment_atom_idx_list)

    separated_group_atom_idx_nested_list = identified_ring_atom_idx_nested_list + fragment_atom_idx_nested_list
    num_separated_groups = len(separated_group_atom_idx_nested_list)
    separated_group_obb_list = [None] * num_separated_groups
    separated_group_obb_info_dict_list = [None] * num_separated_groups
    fragment_coords_array = fragment_mol.GetConformer().GetPositions()

    for separated_group_idx in range(num_separated_groups):
        separated_group_atom_idx_list = separated_group_atom_idx_nested_list[separated_group_idx]
        num_separated_group_atoms = len(separated_group_atom_idx_list)
        separated_group_coords_array = fragment_coords_array[separated_group_atom_idx_list, :]

        obb_points = point_list_to_TColgp_Array1OfPnt(separated_group_coords_array)

        if tolerance is not None:
            obb_tolerances = point_list_to_TColStd_Array1OfReal(num_separated_group_atoms, tolerance)
        else:
            obb_tolerances = None

        obb = Bnd_OBB()
        obb.ReBuild(theListOfPoints=obb_points,
                    theListOfTolerances=obb_tolerances,
                    theIsOptimal=True)

        separated_group_obb_list[separated_group_idx] = obb

        obb_info_dict = {}
        obb_info_dict['center'] = np.array(obb.Center().Coord())
        obb_info_dict['x_axis_vector'] = np.array(obb.XDirection().Coord())
        obb_info_dict['y_axis_vector'] = np.array(obb.YDirection().Coord())
        obb_info_dict['z_axis_vector'] = np.array(obb.ZDirection().Coord())
        obb_info_dict['x_axis_size'] = obb.XHSize()
        obb_info_dict['y_axis_size'] = obb.YHSize()
        obb_info_dict['z_axis_size'] = obb.ZHSize()

        separated_group_obb_info_dict_list[separated_group_idx] = obb_info_dict

    return separated_group_obb_list, separated_group_obb_info_dict_list

def check_self_contact_conformation(coords_array):
    collision_flag = False
    num_atoms = coords_array.shape[0]
    for i in range(num_atoms-1):
        coords_i = coords_array[i ,:]
        coords_j_array = coords_array[i+1:num_atoms ,:]
        num_j_atoms = coords_j_array.shape[0]
        coords_i_broadcast = np.broadcast_to(coords_i, (num_j_atoms, 3))
        i_j_diff = coords_i_broadcast - coords_j_array
        i_j_dist = np.linalg.norm(i_j_diff, axis=1)

        if np.sum(i_j_dist < 0.9) > 0:
            collision_flag = True
            break

    return collision_flag

def check_complex_contact_conformation(protein_coords_array, ligand_coords_array):
    collision_flag = False
    free_flag = True
    num_protein_atoms = protein_coords_array.shape[0]
    num_ligand_atoms = ligand_coords_array.shape[0]

    for atom_idx in range(num_ligand_atoms):
        coords = ligand_coords_array[atom_idx ,:]
        coords_broadcast = np.broadcast_to(coords, (num_protein_atoms, 3))
        atom_residue_diff = coords_broadcast - protein_coords_array
        atom_residue_dist = np.linalg.norm(atom_residue_diff, axis=1)

        if np.sum(atom_residue_dist < 0.9) > 0:
            collision_flag = True

        if np.sum(atom_residue_dist < 6.5) > 0:
            free_flag = False

        if collision_flag:
            break

    return collision_flag, free_flag

@jit('bool_(float32[:,:])', nopython=True, nogil=True, fastmath=True)
def check_self_contact_conformation_jit(coords_array):
    collision_flag = False
    num_atoms = np.int32(coords_array.shape[0])
    cutoff = np.float32(0.9)
    dim = 3

    for i in range(num_atoms):
        coords_i = coords_array[i ,:]
        for j in range(i+1, num_atoms):
            coords_j = coords_array[j ,:]
            coords_diff = coords_i - coords_j

            s = np.float32(0.0)
            for k in range(dim):
                s += coords_diff[k]**2

            if np.sqrt(s) < cutoff:
                collision_flag = True
                break

        if collision_flag:
            break

    return collision_flag

def mol2svg(mol):
    """Transform a rdMol to svg
    Args:
        mol: rdMol
        
    Returns:
        svg: svg
    """

    drawed_mol = PrepareMolForDrawing(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)
    drawer.DrawMolecule(drawed_mol)
    drawer.AddMoleculeMetadata(drawed_mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return svg

def mol2image(mol):
    """Mol to `data:image` transformation

    Args:
        mol: RDKit Mol object

    Returns:
        str: `data:image` string
    """
    svg_string = mol2svg(mol)
    impath = 'data:image/svg+xml;charset=utf-8,' + parse.quote(svg_string, safe='')

    return impath

def nx_to_ipycyto(
    graph: nx.Graph,
    scale_factor: int = 10,
    to_undirected: bool = True,
    calculate_positions: bool = True
) -> t.Dict:
    """transform a networkx to cytoscape dict

    Args:
        graph (nx.Graph): graph 
        scale_factor (int, optional): scale factor. Defaults to 10.
        to_undirected (bool, optional): whether to_undirected, default to True
        calculate_positions (bool, optional): 
          whether to calculate the fruchterman_reingold_layout, default to True
    Returns:
        t.Dict: cyto scape json data
    """
    SCALE_FACTOR = min(
        30000,
        graph.number_of_nodes() * scale_factor
    )
    if calculate_positions:
        if to_undirected:
            pos = nx.fruchterman_reingold_layout(
                nx.to_undirected(graph),
                iterations=300
            )
        else:
            pos = nx.fruchterman_reingold_layout(
                graph,
                iterations=300
            ) 
    ipycyto_data = {}
    node_positions = {}
    ipycyto_data['nodes'] = []
    ipycyto_data['edges'] = []
    
    for node in graph.nodes:
        graph.nodes[node]['id'] = node 
        if calculate_positions:
            position = {
                'x': pos[int(node)][0] * SCALE_FACTOR,
                'y': pos[int(node)][1] * SCALE_FACTOR
            }
            node_positions[node] = position
        ipycyto_data['nodes'].append(
            {'data': graph.nodes[node]}
        )
    for edge in graph.edges:
        start_id = edge[0]
        end_id = edge[1]
        edge_data = {
            'data': {
                'source': start_id,
                'target': end_id
            }
        }
        ipycyto_data['edges'].append(edge_data)

    return ipycyto_data, node_positions


def create_network_view(
    graph: nx.Graph,
    color_mapper: str,
    scale_factor: int = 10,
    to_undirected: bool = True,
    layout: str = 'preset'
) -> ipycytoscape.cytoscape.CytoscapeWidget:
    """create a jupyter graph view

    Args:
        graph (nx.Graph): graph
        color_mapper (str): a string color map, generated by ColorMap
        scale_factor (int, optional): node scale factor. Defaults to 10.
        layout (str, optional): graph layout. Defaults to 'preset'.

    Returns:
        ipycytoscape.cytoscape.CytoscapeWidget: the jupyter widget
    """
    ipyto_data, node_positions = nx_to_ipycyto(
        graph=graph,
        scale_factor=scale_factor,
        to_undirected=to_undirected,
        calculate_positions=(layout == 'preset')
    )
    cyto = ipycytoscape.CytoscapeWidget()
    cyto.graph.add_graph_from_json(ipyto_data)
    cyto.set_style([
        {
            'css': {
                'background-color': color_mapper,
                'shape': 'circle',
                'width': '200px',
                'height': '200px',
                'border-color': 'rgb(0,0,0)',
                'border-opacity': 1.0,
                'border-width': 1.0,
                'color': '#4579e8',
                'background-image': 'data(img)',
                'background-fit': 'contain'
            },
            'selector': 'node'
        },
        {
            'css': {'width': 20.0, },
            'style': {
                    'width': 4,
                    'curve-style': 'bezier',
                    'line-color': '#9dbaea',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': '#9dbaea',
                },
            'selector': 'edge'
        }
    ])
    if layout == 'preset':
        cyto.set_layout(name=layout, positions=node_positions)
    else:
        cyto.set_layout(name=layout)

    return cyto
