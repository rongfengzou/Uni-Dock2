import os
import sys
import argparse
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from unidock.unidock_processing.unidocktools.unidock_protocol_runner import UnidockProtocolRunner
from unidock import __version__

logo_description = r"""

    ██╗   ██╗██████╗ ██████╗ 
    ██║   ██║██╔══██╗╚════██╗
    ██║   ██║██║  ██║ █████╔╝
    ██║   ██║██║  ██║██╔═══╝ 
    ╚██████╔╝██████╔╝███████╗
     ╚═════╝ ╚═════╝ ╚══════╝

    DP Technology Docking Toolkit

"""

def main():
    parser = argparse.ArgumentParser(
        prog='unidock2',
        description=logo_description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-r', '--receptor', required=True,
                        help='Receptor structure file in PDB or DMS format')

    parser.add_argument('-l', '--ligand',
                        default=None,
                        help='Single ligand structure file in SDF format')
    parser.add_argument('-lb', '--ligand_batch',
                        default=None,
                        help='Recorded batch text file of ligand SDF file path')

    parser.add_argument('-w', '--working_dir',
                        default=None,
                        help='Working directory path')

    parser.add_argument('--center', nargs=3, type=float, metavar=('center_x', 'center_y', 'center_z'),
                        default=[0.0, 0.0, 0.0],
                        help='Docking box center coordinates')
    parser.add_argument('--size', nargs=3, type=float, metavar=('size_x', 'size_y', 'size_z'),
                        default=[30.0, 30.0, 30.0],
                        help='Docking box dimensions')

    parser.add_argument('--constraint', action='store_true',
                        help='Enable constraint docking mode')
    parser.add_argument('--covalent', action='store_true',
                        help='Enable covalent docking mode')
    
    parser.add_argument('--reference',
                        default=None,
                        help='Reference molecule SDF file name for constraint docking mode')
    parser.add_argument('--atom_mapping',
                        default=None,
                        help='Custom atom mapping dict specification for each molecules as a list of dict in JSON file')
    parser.add_argument('--covalent_residue',
                        default=None,
                        help='Covalent residue atom info specification as a list in JSON file')

    parser.add_argument('-v', '--version', action='version',
                        version=f'%(prog)s {__version__}',
                        help='Show program version')

    args = parser.parse_args()

    receptor_file_name = os.path.abspath(args.receptor)

    if args.ligand:
        ligand_sdf_file_name = os.path.abspath(args.ligand)
    else:
        ligand_sdf_file_name = None

    if args.ligand_batch:
        with open(args.ligand_batch, 'r') as ligand_batch_file:
            ligand_batch_line_list = ligand_batch_file.readlines()

        batch_ligand_sdf_file_name_list = []
        for ligand_batch_line in ligand_batch_line_list:
            batch_ligand_sdf_file_name = ligand_batch_line.strip()
            if len(batch_ligand_sdf_file_name) != 0:
                batch_ligand_sdf_file_name_list.append(os.path.abspath(batch_ligand_sdf_file_name))
    else:
        batch_ligand_sdf_file_name_list = []

    if ligand_sdf_file_name:
        total_ligand_sdf_file_name_list = [ligand_sdf_file_name] + batch_ligand_sdf_file_name_list
    else:
        total_ligand_sdf_file_name_list = batch_ligand_sdf_file_name_list

    if len(total_ligand_sdf_file_name_list) == 0:
        raise ValueError('Ligand SDF file input not found !!')

    if args.working_dir:
        working_dir_name = os.path.abspath(args.working_dir)
    else:
        working_dir_name = os.path.abspath('.')

    if args.ref and args.constraint:
        reference_sdf_file_name = os.apth.abspath(args.ref)
    else:
        reference_sdf_file_name = None

    if args.atom_map and args.constraint:
        with open(args.atom_map, 'r') as atom_mapping_file:
            core_atom_mapping_dict_list = json.load(atom_mapping_file)
    else:
        core_atom_mapping_dict_list = None

    if args.covalent_residue and args.covalent:
        with open(args.covalent_residue, 'r') as covalent_residue_file:
            covalent_residue_atom_info_list = json.load(covalent_residue_file)
    else:
        covalent_residue_atom_info_list = None

    docking_runner = UnidockProtocolRunner(
        receptor_pdb_file=receptor_file_name,
        ligand_sdf_files=total_ligand_sdf_file_name_list,
        target_center=tuple(args.center),
        box_size=tuple(args.size),
        template_docking=args.constraint,
        covalent_ligand=args.covalent,
        working_dir_name=working_dir_name,
        reference_sdf_file_name=reference_sdf_file_name,
        core_atom_mapping_dict_list=core_atom_mapping_dict_list,
        covalent_residue_atom_info_list=covalent_residue_atom_info_list
    )

    docking_runner.run_unidock_protocol()

if __name__ == '__main__':
    main()
