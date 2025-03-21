import os

from rdkit import Chem

def get_real_core(core_sdf, core_smi, save_path='save'):
    os.makedirs(save_path, exist_ok=True)
    real_core_sdf = os.path.join(save_path, 'real_core.sdf')
    core_smi_mol = Chem.MolFromSmiles(core_smi)
    core_smi_mol = Chem.AddHs(core_smi_mol)
    core_smi_mol = Chem.DeleteSubstructs(core_smi_mol, Chem.MolFromSmiles('*'))
    mol = Chem.SDMolSupplier(core_sdf, removeHs=False)[0]
    core = Chem.ReplaceSidechains(mol, core_smi_mol)
    core = Chem.DeleteSubstructs(core, Chem.MolFromSmiles('*'))
    assert core
    Chem.SDWriter(real_core_sdf).write(core)
    return real_core_sdf
