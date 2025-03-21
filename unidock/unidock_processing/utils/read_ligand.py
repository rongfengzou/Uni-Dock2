from typing import List
from pathlib import Path
import os
import logging
import traceback
from rdkit import Chem
from unidock.unidock_processing.utils.rdkit_helper import set_properties

def read_smi_file(file_path:str) -> List[Chem.Mol]:
    try:
        fname = Path(file_path).stem
        mols = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if line.strip():
                    try:
                        line_list_1 = line.strip().split(' ')
                        line_list_2 = line.strip().split('\t')
                        line_list_3 = line.strip().split(',')

                        if len(line_list_1) > 1:
                            parsed_line_list = line_list_1
                        elif len(line_list_2) > 1:
                            parsed_line_list = line_list_2
                        elif len(line_list_3) > 1:
                            parsed_line_list = line_list_3
                        else:
                            parsed_line_list = line_list_1

                        smi = parsed_line_list[0]
                        if len(parsed_line_list) > 1:
                            name = parsed_line_list[1]
                        else:
                            name = f'{fname}_{i}'

                        mol = Chem.MolFromSmiles(smi, sanitize=False)
                        mol.SetProp('_Name', name)
                        mols.append(mol)
                    except:
                        continue
        return mols
    except:
        logging.error(f'Read SMI file error: {traceback.format_exc()}')
        return []

def read_mol_file(file_path:str) -> List[Chem.Mol]:
    try:
        mol = Chem.MolFromMolFile(file_path, removeHs=False, sanitize=False, strictParsing=False)
        return [mol]
    except:
        logging.error(f"Read MOL file error: {traceback.format_exc()}")
        return []

def read_sdf_file(file_path:str) -> List[Chem.Mol]:
    try:
        mols = [m for m in Chem.SDMolSupplier(file_path, removeHs=False, sanitize=False, strictParsing=False)]
        return mols
    except:
        logging.error(f"Read SDF file error: {traceback.format_exc()}")
        return []

def read_pdb_file(file_path:str) -> List[Chem.Mol]:
    try:
        mol = Chem.MolFromPDBFile(file_path, removeHs=False, sanitize=False)
        return [mol]
    except:
        logging.error(f"Read SDF file error: {traceback.format_exc()}")
        return []

def read_ligand(ligand_path:str, **kwargs) -> List[Chem.Mol]:
    if not os.path.exists(ligand_path):
        raise FileNotFoundError('%s does not exist' % ligand_path)

    mols = []
    file_name, ext = os.path.splitext(os.path.basename(ligand_path))
    if ext == ".sdf":
        mols = read_sdf_file(ligand_path)
    elif ext == ".mol":
        mols = read_mol_file(ligand_path)
    elif ext == ".smi":
        mols = read_smi_file(ligand_path)
    elif ext == ".pdb":
        mols = read_pdb_file(ligand_path)
    else:
        raise KeyError("Invalid ligand format")

    for ind, mol in enumerate(mols):
        try:
            mol.SetProp("filename", f"{file_name}_{ind}" if len(mols) > 1 else file_name)
            if kwargs:
                set_properties(mol, kwargs)
        except:
            logging.error(f"Failed to process ligand file {file_name} idx {ind}: {traceback.format_exc()}")
    return mols
