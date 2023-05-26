import torch
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, GetPeriodicTable
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from torch_geometric.data import Data
import scipy.spatial as spa

periodic_table = GetPeriodicTable()

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
}

feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
])), 1)  # number of scalar features

def count_ats(mol):
    count = 0
    for atom in mol.GetAtoms():
        count += 1
    return count

def canon_mol(mol):
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    smi = Chem.MolToSmiles(mol)
    smi = Chem.CanonSmiles(smi)
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    #Chem.SanitizeMol(mol)
    return mol

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def reader(xyz):
    with open(xyz, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    nat = int(lines[0])
    start_idx = 2
    end_idx = start_idx + nat

    atomtypes = []
    coords = []

    for line_idx in range(start_idx, end_idx):
        line = lines[line_idx]
        atomtype, x, y, z = line.split()
        atomtypes.append(str(atomtype))
        coords.append([float(x), float(y), float(z)])

    assert len(atomtypes) == nat
    assert len(coords) == nat
    return np.array(atomtypes), np.array(coords)

def atom_featurizer(mol):
    ComputeGasteigerCharges(mol)  # they are Nan for 93 molecules in all of PDBbind. We put a 0 in that case.
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        g_charge = atom.GetDoubleProp('_GasteigerCharge')
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.
        ])

    return torch.tensor(atom_features_list)


def get_graph(mol, atomtypes, coords, y, radius=20, max_neighbor=24, device='cpu'):
    """
    Builds graph using specified coords. Only using distances.

    data.x -> Node features
    data.pos -> xyz coordinates
    data.edge_index -> edge_src, edge_dst
    data.edge_attr -> distance
    data.y -> Energy
    """
   # mol = canon_mol(mol)

  #  atoms = []
  #  for atom in mol.GetAtoms():
   #     sym = atom.GetSymbol()
   #     atoms.append(sym)
  #  print('atoms in canon SMILES', atoms_canon)
  #  print('atoms in xyz', atomtypes)
   # count = len(atoms)

    # now compare to atomtypes and coords
   # print('atoms in smiles', atoms)
   # print('atomtypes', atomtypes)
   # assert np.all(atoms == atomtypes), 'atoms from xyz and smiles dont match!'

    num_nodes = coords.shape[0]
   # assert num_nodes == count, "rdkit atom count different from num nodes"
    assert coords.shape[1] == 3

    distance = spa.distance.cdist(coords, coords)

    src_list = []
    dst_list = []
    dist_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbor + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            print(
                f'The radius {radius} was too small for one atom such that it had no neighbors. So we connected {i} to the closest other atom {dst}')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distance[i, dst])
        dist_list.extend(valid_dist)

    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)

    x = atom_featurizer(mol)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(dist_list)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=torch.tensor(coords, dtype=torch.float32))

    return data.to(device)
