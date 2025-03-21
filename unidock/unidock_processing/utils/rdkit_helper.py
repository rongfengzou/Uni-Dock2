from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol


def write_conformer_to_mol(mol,
                           conf_idx,
                           conf_energy,
                           ligand_smiles,
                           ligand_molecule_name,
                           ligand_conf_name):

    conf = mol.GetConformer(conf_idx)
    conf_mol = Chem.Mol(mol)
    conf_mol.RemoveAllConformers()
    conf_mol.AddConformer(conf)
    conf_mol.SetProp('conformer_energy', '%.2f' % conf_energy)
    conf_mol.SetProp('ligand_smiles', ligand_smiles)
    conf_mol.SetProp('_Name', ligand_conf_name)
    conf_mol.SetProp('ligand_conf_name', ligand_conf_name)
    conf_mol.SetProp('ligand_molecule_name', ligand_molecule_name)
    conf_mol = PropertyMol(conf_mol)

    return conf_mol

def reorder_mol_heavy_atoms(mol):
    num_atoms = mol.GetNumAtoms()
    renumbered_heavy_atom_idx_list = []
    renumbered_hydrogen_atom_idx_list = []

    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != 'H':
            renumbered_heavy_atom_idx_list.append(atom_idx)
        else:
            renumbered_hydrogen_atom_idx_list.append(atom_idx)

    renumbered_atom_idx_list = renumbered_heavy_atom_idx_list + renumbered_hydrogen_atom_idx_list
    if len(renumbered_atom_idx_list) != num_atoms:
        raise IndexError('Bugs in heavy atom reordering code!!')

    mol_reordered = Chem.RenumberAtoms(mol, renumbered_atom_idx_list)

    return mol_reordered, renumbered_atom_idx_list

def set_properties(mol: Chem.Mol, props: dict):
    for k, v in props.items():
        if isinstance(v, int):
            mol.SetIntProp(k, v)
        if isinstance(v, float):
            mol.SetDoubleProp(k, v)
        else:
            mol.SetProp(k, str(v))