import os
import time
import json
import shutil
import logging
import numpy as np
import pandas as pd
from glob import glob

from rdkit import Chem, RDLogger
from unidock.unidock_processing.utils.smi_tools import StandardizeSmiles
from unidock.unidock_processing.utils.constant import IN_DB_PROPS
from xmoleverse_client.api import molecule

rdlg = RDLogger.logger()
rdlg.setLevel(RDLogger.CRITICAL)
used_names = []


def multi_run(inputs, func, core_num=None):
    from multiprocessing import Pool

    if len(inputs) == 1:
        return [func(inputs[0])]
    if core_num:
        pool = Pool(processes=core_num)
    else:
        pool = Pool()
    contents = pool.map(func, inputs)
    pool.close()
    return contents


def filter_by_rmsd(ene_mol, rmsd_cutoff=1.0):
    """ remove conform with minimum RMSD to other conforms less than rmsd_cutoff

    :param ene_mol: list of (VinaScore, RDKit Mol)
    :param rmsd_cutoff: minimum rmsd allowed between any two conformations
    :return: list of (VinaScore, RDKit Mol)
    """
    ene_mol.sort(key=lambda x: x[0])  # make sure to keep configuration with the lowest vina_score
    mols_noh = [Chem.RemoveHs(m[1]) for m in ene_mol]
    ref_mol = mols_noh[0]
    matches = ref_mol.GetSubstructMatches(ref_mol, useChirality=True, uniquify=False)
    n_atoms = ref_mol.GetNumAtoms()
    rmsd_cutoff_2 = rmsd_cutoff ** 2
    for i in range(len(ene_mol)):
        if len(ene_mol) - 1 <= i:
            break
        crd_i = mols_noh[i].GetConformer(0).GetPositions()
        for j in range(len(ene_mol) - 1, i, -1):
            crd_j = mols_noh[j].GetConformer(0).GetPositions()
            # calc best rmsd between mol i and j
            rmsd_list = [np.sum((crd_i - crd_j[np.array(match)]) ** 2) / n_atoms for match in matches]
            if np.min(rmsd_list) < rmsd_cutoff_2:
                ene_mol.remove(ene_mol[j])
    logging.debug("remove %d poses for rmsd_cutoff" % (len(mols_noh) - len(ene_mol)))
    return ene_mol


def parse_ifp_data(ifp_str=''):
    """ Parse IFP data from ifp_str.
        e.g. HYD_*_LEU_1284...12_15_11_22_14_20,ACC_*_HIS_1224...0,DON_*_HIS_1224...7
    """
    ifp_data = []
    for item in ifp_str.strip().split(','):
        if '_' not in item or '...' not in item:
            logging.warning(f"Invalid IFP string: {item}, just ignore!")
            continue
        key, atoms = item.split('...')
        items = key.split('_')
        ifp_data.append({
            "type": items[0],
            "residue": '_'.join(items[1:]),
            "atoms": list(map(int, atoms.split('_')))
        })
    return ifp_data


def get_sdf_props(mol, name_dic=None):
    """ Get docking score, ifp data from RDKit Mol
    """
    ifp_str, dock_score, vina_score, vina_score_le = '', None, None, None
    prop_names = mol.GetPropNames()
    if 'Interaction_Sum' in prop_names:
        ifp_str = mol.GetProp('Interaction_Sum')
    if 'VINA.SCORE' in prop_names:
        vina_score = float(mol.GetProp('VINA.SCORE'))
        vina_score_le = vina_score / mol.GetNumHeavyAtoms()
    if 'SCORE.INTER' in prop_names:
        dock_score = float(mol.GetProp('SCORE.INTER'))
    if 'Score' in prop_names:
        dock_score = float(mol.GetProp('Score'))
    prop_dic = {"dock_score": dock_score, "vina_score": vina_score,
                "vina_score_le": vina_score_le, "ifp_data": parse_ifp_data(ifp_str)}
    if name_dic:
        name = mol.GetProp('_Name')[:12]
        if name in name_dic:
            prop_dic["name"] = gen_name(name_dic[name])
    return prop_dic


def gen_name(name):
    if name in used_names:
        for i in range(50):
            tmp = f"{name}-{i}"
            if tmp not in used_names:
                used_names.append(tmp)
                return tmp
    else:
        used_names.append(name)
        return name


class MergeResults(object):
    def __init__(self, sdf_list, csv_list, max_num=30000, rmsd_threshold=1):
        logging.info(f'sdf_list {len(sdf_list)}, csv_list {len(csv_list)}')
        save_dir_path = 'save'
        os.makedirs(save_dir_path, exist_ok=True)
        self.sdf_list = sdf_list
        self.csv_list = csv_list
        self.save_dir_path = save_dir_path
        self.max_num = max_num
        self.rmsd_threshold = rmsd_threshold
        self.tmp_path = os.path.join(save_dir_path, 'tmp')
        os.makedirs(self.tmp_path, exist_ok=True)
        self.mol_names = []

    def filter_vina_score(self):
        """ 1. Group molecules by {original_name}_isomer_{idx}
            2. Get molecule names with minimum vina_score ranking before max_num
        """
        tmp_dic = {}
        for sdf in self.sdf_list:
            for mol in Chem.SDMolSupplier(sdf, removeHs=False):
                if not mol or 'VINA.SCORE' not in mol.GetPropNames():
                    continue
                score = float(mol.GetProp('VINA.SCORE')) / mol.GetNumHeavyAtoms()
                name = mol.GetProp('_Name').split('_conf_')[0]
                if name not in tmp_dic:
                    tmp_dic[name] = []
                tmp_dic[name].append(score)
        tmp_list = [(min(v), k) for k, v in tmp_dic.items()]
        self.mol_names = [t[1] for t in sorted(tmp_list)[:self.max_num]]

    def write_split_sdf(self, sdf):
        """ Write sdf for name in mol_names
        """
        for mol in Chem.SDMolSupplier(sdf, removeHs=False):
            if not mol:
                continue
            name = mol.GetProp('_Name')
            if name.split('_conf_')[0] not in self.mol_names:
                continue
            sdw = Chem.SDWriter(os.path.join(self.tmp_path, f'{name}.sdf'))
            sdw.write(mol)
            sdw.close()

    def filter_rmsd(self, mol_name):
        """ 1. Delete configurations of the same molecule with RMSD < rmsd_threshold
            2. Write the molecules into split sdf
        """
        prefix = os.path.join(self.tmp_path, mol_name)
        sdfs = glob(prefix + '_conf_*.sdf')
        if len(sdfs) > 1:
            ene_mol = []
            for sdf in sdfs:
                mol = Chem.SDMolSupplier(sdf, removeHs=False)[0]
                ene_mol.append((float(mol.GetProp('VINA.SCORE')), mol))
                os.remove(sdf)
            ene_mol = filter_by_rmsd(ene_mol, rmsd_cutoff=self.rmsd_threshold)
            sdw = Chem.SDWriter(prefix + '.sdf')
            for ene, mol in ene_mol:
                sdw.write(mol)
            sdw.close()
        elif len(sdfs) == 1:
            os.rename(sdfs[0], prefix + '.sdf')
        else:
            if not os.path.exists(prefix + '.sdf'):
                logging.info(f"No sdf file found for {mol_name}!")

    def save_to_db(self, name_dic=None):
        res_df = pd.concat([pd.read_csv(csv) for csv in self.csv_list])
        cols = ["NAME", "SMILES"] + IN_DB_PROPS
        props_list = res_df[cols].to_dict("records")
        del res_df
        mols_info = list()
        smiles_name_dic = dict()
        for props in props_list:
            name = props.pop("NAME")
            if name not in self.mol_names:  # skip the molecules not in mol_names
                continue
            smi_ori = props.pop("SMILES")
            smiles = StandardizeSmiles(smi_ori)
            if smiles in smiles_name_dic:
                continue
            if smi_ori != smiles:
                logging.warning(f"smiles of {name} changed from {smi_ori} to {smiles} after standardize!")
            mols_info.append({"smiles": smiles, "data": props})
            smiles_name_dic[smiles] = name

        logging.info(f"Number of Mols before in_db: {len(mols_info)}")
        data = molecule.batch_add_molecule(mols_info)
        logging.info(f"Number of Mols after in_db: {len(data)}")
        mol_ids = list()
        sdf_props = dict()
        sdw = Chem.SDWriter(os.path.join(self.save_dir_path, 'results.sdf'))
        for d in data:
            _id = d["id"]
            _smiles = d["smiles"]
            if _smiles in smiles_name_dic:
                mol_ids.append(_id)
                sdf_path = os.path.join(self.tmp_path, f"{smiles_name_dic[_smiles]}.sdf")
                for i, mol in enumerate(Chem.SDMolSupplier(sdf_path, removeHs=False)):
                    if i == 0:
                        sdf_props[_id] = get_sdf_props(mol, name_dic=name_dic)
                        mol.SetProp('_Name', _id)
                    else:
                        mol.SetProp('_Name', f"{_id}~conf{i}")
                    sdw.write(mol)
            else:
                logging.warning(f"Changed to {_smiles} after saving to database, ignored!!!")
        sdw.close()
        json.dump(mol_ids, open(os.path.join(self.save_dir_path, "mol_ids.json"), "w"))
        json.dump(sdf_props, open(os.path.join(self.save_dir_path, "sdf_props.json"), "w"))
        shutil.rmtree(self.tmp_path)

    def run(self, n_cpu=4, name_dic=None):
        t0 = time.time()
        # group by name and filter by vina_score
        self.filter_vina_score()
        t1 = time.time()
        logging.info(f'filter_vina_score {t1 - t0}')

        # write sdf for name in mol_names
        multi_run(self.sdf_list, self.write_split_sdf, core_num=n_cpu)
        t2 = time.time()
        logging.info(f'write_split_sdf {t2 - t1}')

        # filter configurations by rmsd
        multi_run(self.mol_names, self.filter_rmsd, core_num=n_cpu)
        t3 = time.time()
        logging.info(f'filter_rmsd {t3 - t2}')

        # save_to_db
        self.save_to_db(name_dic=name_dic)
        t4 = time.time()
        logging.info(f'save_to_db {t4 - t3}')

        return self.save_dir_path
