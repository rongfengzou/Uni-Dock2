from rdkit import Chem


def get_is_ring_aromatic(mol:Chem.rdchem.Mol, bond_ring):
    for bond_idx in bond_ring:
        if not mol.GetBondWithIdx(bond_idx).GetIsAromatic():
            return False
    return True


def get_is_mol_with_complex_and_saturated_rings(mol:Chem.rdchem.Mol):
    ring_info = mol.GetRingInfo()
    atom_ring_list = list(ring_info.AtomRings())
    bond_ring_list = list(ring_info.BondRings())
    num_rings = len(atom_ring_list)
    if num_rings == 0:
        return False
    ring_size_list = [None] * num_rings
    for ring_idx in range(num_rings):
        atom_ring = atom_ring_list[ring_idx]
        ring_size_list[ring_idx] = len(atom_ring)
    if max(ring_size_list) <= 4:
        return False
    is_aromatic_ring_set = set()
    for ring_idx in range(num_rings):
        bond_ring = bond_ring_list[ring_idx]
        is_aromatic_ring_set.add(get_is_ring_aromatic(mol, bond_ring))
    is_aromatic_ring_list = list(is_aromatic_ring_set)
    if len(is_aromatic_ring_list) == 1 and is_aromatic_ring_list[0] == True:
        return False
    return True
