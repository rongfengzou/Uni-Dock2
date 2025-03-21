#!/usr/bin/env python
# coding=utf-8

import os
from rdkit import Chem


def sdf2smi(sdf_path):
    from .smi_tools import StandardizeSmiles
    res = set()
    for mol in Chem.SDMolSupplier(sdf_path, removeHs=False):
        try:
            smi = StandardizeSmiles(Chem.MolToSmiles(mol))
            if smi:
                res.add(smi)
        except:
            pass
    return list(res)


def split_sdfs(sdf_files, n_mols=500):
    """ split the sdf files to new sdf files, so that number of molecules ≤ n_mols in each output sdf

    :param sdf_files: a list of file name, each sdf may have multiple molecules
    :param n_mols: maximum number of molecules in each output sdf
    :return: sdf_files_new
    """
    i, j = 0, 0
    lines, file_paths = [], []
    for sdf_file in sdf_files:
        sdf_lines = open(sdf_file).readlines()
        for line in sdf_lines:
            if line.startswith('$$$$'):
                lines.append(line)
                i += 1
                if i % n_mols == 0:
                    file_path = os.path.abspath('split_%s.sdf' % j)
                    file_paths.append(file_path)
                    with open(file_path, 'w') as fp:
                        fp.write(''.join(lines))
                    j += 1
                    lines = []
            else:
                lines.append(line)
    if lines:
        file_path = os.path.abspath('split_%s.sdf' % j)
        file_paths.append(file_path)
        with open(file_path, 'w') as fp:
            fp.write(''.join(lines))

    return file_paths, i


def extract_mols_from_sdf(sdf_block, name_dic, split=False, merge_conf=False):
    """ Extract molecules in names from sdf_block

    :param sdf_block: sdf file content with multiple molecules
    :param name_dic: {name: name_new} of molecules to extract from the sdf_block
    :param split: whether to split into different sdf files
    :param merge_conf: merge conformations of the same molecule into one sdf file
    :return: {file_name: file_content}
    """
    name_sdf_dic = {}
    for sdf in sdf_block.split("$$$$\n")[:-1]:  # ignore the last empty record
        items = sdf.splitlines()[0].strip().split("~")
        if items[0] in name_dic:
            name_new = name_dic[items[0]]
            if not name_new:
                name_new = items[0]
            tmp_name = name_new + ".sdf"
            if len(items) > 1:
                name_new += "_" + "".join(items[1:])
            lines = sdf.split("\n")   # rename the mol
            lines[0] = name_new
            lines[-1] = "$$$$\n"      # replace the empty record with $$$$
            sdf_str = "\n".join(lines)
            if split and merge_conf:
                if tmp_name in name_sdf_dic:
                    name_sdf_dic[tmp_name] = name_sdf_dic[tmp_name] + sdf_str
                else:
                    name_sdf_dic[tmp_name] = sdf_str
            else:
                name_sdf_dic[name_new + ".sdf"] = sdf_str
    if split:
        return name_sdf_dic
    else:
        return {"merged.sdf": ''.join(name_sdf_dic.values())}


def merge_sdf(sdf_path, sdf_list):
    sdf_writer = Chem.SDWriter(sdf_path)
    for mini_sdf in sdf_list:
        mol = Chem.SDMolSupplier(mini_sdf, removeHs=False)[0]
        sdf_writer.write(mol)
    sdf_writer.close()


def core_filter_sdf(sdf_path, core_smi):
    from unidock.unidock_processing.utils.smi_tools import get_smarts_mol, get_RGroup_d
    core_mols_smiles = Chem.MolFromSmiles(core_smi)
    core_mols_smarts = get_smarts_mol(core_smi)

    def core_match(mol):
        sub_d = get_RGroup_d([core_mols_smiles], mol, strict=True)
        if sub_d is None:
            sub_d = get_RGroup_d([core_mols_smarts], mol, strict=True)
        return bool(sub_d)

    res = list()
    for mol in Chem.SDMolSupplier(sdf_path, removeHs=False):
        if core_match(Chem.RemoveHs(mol)):
            res.append(mol)
    new_sdf_path = 'saved.sdf'
    writer = Chem.SDWriter(new_sdf_path)
    for mol in res:
        writer.write(mol)
    writer.close()
    return new_sdf_path 


def write_mol_to_sdf_file(mol:Chem.Mol, filepath:str):
    if mol.GetNumConformers() == 0:
        with Chem.SDWriter(filepath) as writer:
            writer.write(mol)
    else:
        name = ""
        if mol.HasProp("_Name"):
            name = mol.GetProp('_Name')
        with Chem.SDWriter(filepath) as writer:
            for i, conf in enumerate(mol.GetConformers()):
                if mol.HasProp("_Name"):
                    mol.SetProp("_Name", f"{name}_con{(i+1):0>2}")
                writer.write(mol, confId=conf.GetId())