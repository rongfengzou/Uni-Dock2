from rdkit import Chem

ATOM_TYPE_DEFINITION_LIST = [{'smarts': '[#1]',                                                                                           'atype': 'H'},
                             {'smarts': '[#5]',                                                                                           'atype': 'B'},
                             {'smarts': '[#6]',                                                                                           'atype': 'C'},
                             {'smarts': '[#6;$([#6][#5,#7,#8,#9,#14,#15,#16,#17,#34,#35,#53])]',                                          'atype': 'CP'},
                             {'smarts': '[#7]',                                                                                           'atype': 'N'},
                             {'smarts': '[$([#7;v3,v4&+1]);!H0]',                                                                         'atype': 'ND'},
                             {'smarts': '[#7&!$([nX3])&!$([NX3]-[*]=,:[#7,#8,#15,#16])&!$([NX3]-[a])&!$([#7v4&+1])]',                     'atype': 'NA'},
                             {'smarts': '[#7&$([#7;v3])&!$([nX3])&!$([NX3]-[*]=,:[#7,#8,#15,#16])&!$([NX3]-[a])&!$([#7v4&+1])&!H0]',      'atype': 'NDA'},
                             {'smarts': '[#8]',                                                                                           'atype': 'O'},
                             {'smarts': '[O;+0;!H0]',                                                                                     'atype': 'OD'},
                             {'smarts': '[O,o&+0;!$(O~N~[O-]);!$([OX2](C)C=O);!$(O(~a)~a);!$(O=N-*);!$([O-]-N=O)]',                       'atype': 'OA'},
                             {'smarts': '[O;+0;!$(O~N~[O-]);!$([OX2](C)C=O);!$(O(~a)~a);!$(O=N-*);!$([O-]-N=O);!H0]',                     'atype': 'ODA'},
                             {'smarts': '[#9]',                                                                                           'atype': 'F'},
                             {'smarts': '[F;$(Fc),$(F[C;v4;$(C(F)(-[a,C,#1])(-[a,C,#1])-[a,C,#1])])]',                                    'atype': 'FA'},
                             {'smarts': '[#14]',                                                                                          'atype': 'Si'},
                             {'smarts': '[#15]',                                                                                          'atype': 'P'},
                             {'smarts': '[#16]',                                                                                          'atype': 'S'},
                             {'smarts': '[S;+0;!H0]',                                                                                     'atype': 'SD'},
                             {'smarts': '[#17]',                                                                                          'atype': 'Cl'},
                             {'smarts': '[#35]',                                                                                          'atype': 'Br'},
                             {'smarts': '[#53]',                                                                                          'atype': 'I'},
                             {'smarts': '[#12]',                                                                                          'atype': 'Mg'},
                             {'smarts': '[#20]',                                                                                          'atype': 'Ca'},
                             {'smarts': '[#25]',                                                                                          'atype': 'Mn'},
                             {'smarts': '[#26]',                                                                                          'atype': 'Fe'},
                             {'smarts': '[#30]',                                                                                          'atype': 'Zn'}]

class AtomType(object):
    def __init__(self):
        self.atom_type_definition_list = ATOM_TYPE_DEFINITION_LIST

    def assign_atom_types(self, mol):
        for atom_type_dict in self.atom_type_definition_list:
            smarts = atom_type_dict['smarts']
            atom_type = atom_type_dict['atype']

            pattern_mol = Chem.MolFromSmarts(smarts)
            pattern_matches = mol.GetSubstructMatches(pattern_mol)
            for pattern_match in pattern_matches:
                atom = mol.GetAtomWithIdx(pattern_match[0])
                atom.SetProp('vina_atom_type', atom_type)
