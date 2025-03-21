import hashlib
import collections

from rdkit import Chem


def StandardizeSmiles(smi):
    # from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    try:
        return rdMolStandardize.StandardizeSmiles(smi)
        # return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        return None


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol)
    return None


def get_smarts_mol(smiles, return_smarts=False):
    """
    Transform a smiles to a smarts molecule.
    """
    if smiles.startswith("?"):
        smarts = smiles.strip("?")
    else:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        smarts = Chem.MolToSmarts(mol)
        smarts = "*".join(smarts.split("#0"))
    if return_smarts:
        return smarts
    smarts_mol = Chem.MolFromSmarts(smarts)
    return smarts_mol


def smi_strhash(smi):
    """
    Get string hash for SMILES (usually canonical). Used for fast ID
    generation for molecules and scaffolds.

    Parameters
    ----------
    smi : str
        Input SMILES.

    Returns
    -------
    strhash : str
        36-based hash of SMILES with 12 digits. First 6 digits for
        structural features of SMILES and last 6 digits as uniform hash
        to avoid collisions.
            1 : Length of SMILES (divided by 8).
            2 : Count of aliphatic carbons (divided by 4).
            3 : Count of aromatic carbons (divided by 4).
            4 : Count of double/triple bonds.
            5 : Count of branched chains.
            6 : Count of rings.
            7 ~ 12 : base36 hash.

    Examples
    --------
    >>> smi_strhash("c1cccc(CC2=NNC(=O)c3ccccc23)c1")
    '303223pzgwhm'
    """
    char36 = "0123456789abcdefghijklmnopqrstuvwxyz"
    char_d = collections.Counter(smi.replace("Cl", "Q"))
    rawints = [
        len(smi) >> 3,
        char_d["C"] >> 2,
        char_d["c"] >> 2,
        char_d["="] + char_d["#"],
        char_d["("],
        sum([char_d[key] for key in "123456789"]) >> 1,
    ] + list(hashlib.blake2s(bytes(smi, "utf-8"), digest_size=6).digest())
    strhash = "".join([char36[i % 36] for i in rawints])
    return strhash


def get_RGroup_d(core_mols, mol, strict=False):
    """
    To find the core and substituents for the input molecule
    RGroupDecompose may have problems when input multi-cores and multi-mols

    Parameters
    ----------
    core_mols : List[rdkit.Chem.rdchem.Mol]
        input core candidates
    mol : rdkit.Chem.rdchem.Mol
        input molecule
    strcit : bool
        whether to avoid more substituents than expected.

    Returns
    -------
    sub_d : Dict[str]
        list of substituents.

    Examples
    --------
    >>> smiles = 'CN(c1ncccc1CNc1nc(Nc2ccc(CN)cc2)ncc1C(F)(F)F)S(C)(=O)=O'
    >>> core = "c1nc(N[*:3])nc(N[*:2])c1[*:1]"
    >>> mol = Chem.MolFromSmiles(smiles)
    >>> core_mol = get_smarts_mol(core)
    >>> get_RGroup_d([core_mol], mol)
    {'Core': 'c1nc(N[*:3])nc(N[*:2])c1[*:1]', 'R1': 'FC(F)(F)[*:1]', 'R2': 'CN(c1ncccc1C[*:2])S(C)(=O)=O', 'R3': 'NCc1ccc([*:3])cc1'}
    """
    from rdkit.Chem import rdRGroupDecomposition

    for core_mol in core_mols:
        if not mol.HasSubstructMatch(core_mol):
            continue
        subs, _ = rdRGroupDecomposition.RGroupDecompose(
            [core_mol], [mol], asSmiles=True
        )
        if subs:
            sub_d = subs[0]
            n_sites = Chem.MolToSmiles(core_mol).count("*") if strict else -1
            if check_sub_d(sub_d, n_sites):
                if Chem.MolFromSmiles(sub_d["Core"]):
                    if sub_d["Core"].count(":") > n_sites:
                        sub_d = get_sub_d(core_mol, mol)
                        if sub_d:
                            return sub_d
                        else:
                            return None
                    return sub_d
                else:
                    sub_d = get_sub_d(core_mol, mol)
                    if sub_d:
                        return sub_d
    return None


def get_core_info(core_mol):
    """
    Get core info for substituent recognition.
    """
    core_info = []
    for atom in core_mol.GetAtoms():
        if atom.GetSymbol() == "*":
            idx = atom.GetIdx()
            nei_idx = atom.GetNeighbors()[0].GetIdx()
            core_info.append((idx, nei_idx, atom.GetAtomMapNum()))
    core_info = sorted(core_info, key=lambda x: x[-1])
    return core_info


def get_sub_d(core_mol, mol, keep_map_num=True):
    """
    It is limited when there have more substituents than expected
    or the matched core is connected to a ring.
    So it's useful only when rdRGroupDecomposition.RGroupDecompose not work.

    Parameters
    ----------
    core_mol : rdkit.Chem.rdchem.Mol
    mol : rdkit.Chem.rdchem.Mol
    keep_map_num : bool
        whether to keep the map number for subs

    Returns
    -------
    sub_d : Dict[str]
        list of substituents.

    Examples
    --------
    >>> smiles = 'CN(c1ncccc1CNc1nc(Nc2ccc(CN)cc2)ncc1C(F)(F)F)S(C)(=O)=O'
    >>> core = "c1nc(N[*:3])nc(N[*:2])c1[*:1]"
    >>> mol = Chem.MolFromSmiles(smiles)
    >>> core_mol = get_smarts_mol(core)
    >>> get_sub_d(core_mol, mol)
    {'Core': 'c1nc(N[*:3])nc(N[*:2])c1[*:1]', 'R1': 'FC(F)(F)[*:1]', 'R2': 'CN(c1ncccc1C[*:2])S(C)(=O)=O', 'R3': 'NCc1ccc([*:3])cc1'}
    """
    core_info = get_core_info(core_mol)
    if not core_info:
        return None
    matched_ids = mol.GetSubstructMatch(core_mol)

    bond_ids = [
        mol.GetBondBetweenAtoms(matched_ids[idx], matched_ids[nei_idx]).GetIdx()
        for idx, nei_idx, site_id in core_info
    ]
    mol = Chem.FragmentOnBonds(mol, bond_ids, addDummies=True)
    frag_mols = list(Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    frag_smiles = [Chem.MolToSmiles(mol) for mol in frag_mols]
    sub_d = {}
    for smi in frag_smiles:
        match_str = f"[{matched_ids[core_info[0][0]]}*]"
        if match_str not in smi:
            if match_str == '[0*]' and not '*]' in smi:
                smi = smi.replace('*', '[0*]')
                if match_str not in smi:
                    continue
            else:
                continue
        for idx, nei_idx, map_num in core_info:
            map_info = f":{map_num}" if map_num > 0 else ""
            smi = smi.replace(f"[{matched_ids[idx]}*]", f"[*{map_info}]")
        core = canonical_smiles(smi)
        if not core:
            return None
        sub_d["Core"] = core
        break
    for _, nei_idx, map_num in core_info:
        sub_smi = None
        for smi in frag_smiles:
            if f"[{matched_ids[nei_idx]}*]" in smi:
                map_info = f":{map_num}" if map_num > 0 and keep_map_num else ""
                smi = smi.replace(f"[{matched_ids[nei_idx]}*]", f"[*{map_info}]")
                sub_smi = canonical_smiles(smi)
                sub_d[f"R{map_num}"] = sub_smi
                break
        if not sub_smi:
            return None
    return sub_d


def check_sub_d(sub_d, n_sites):
    if n_sites > 0 and n_sites != sub_d["Core"].count("*"):
        # matched connection atom in ring or undefined substituent existence
        return False
    for key, frag in sub_d.items():
        if key != "Core" and frag.count("*") != 1:
            return False
    return True


def core_filter(smiles_list, core):
    core_mols_smiles = Chem.MolFromSmiles(core)
    core_mols_smarts = get_smarts_mol(core)
    def core_match(smi):
        mol = Chem.MolFromSmiles(smi)
        sub_d = get_RGroup_d([core_mols_smiles], mol, strict=True)
        if sub_d is None:
            sub_d = get_RGroup_d([core_mols_smarts], mol, strict=True)
        return bool(sub_d)
    res = list()
    for smiles in smiles_list:
        if core_match(smiles):
            res.append(smiles)
    return res
