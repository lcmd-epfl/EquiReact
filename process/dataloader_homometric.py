import numpy as np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from process.create_graph import get_graph


class HomometricHe(Dataset):
    def __init__(self, atom_mapping=False):
        self.max_number_of_reactants = 1
        self.max_number_of_products = 1
        self.labels = torch.tensor([6.666])
        self.indices = [0]
        self.nreactions = len(self.labels)
        self.process()
        self.std = torch.tensor(1.0)
        self.atom_mapping = atom_mapping


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        r_graph = self.reactant_graphs[idx]
        p_graph = self.product_graphs[idx]
        label = self.labels[idx]
        if self.atom_mapping:
            r_map = self.reactants_maps[idx]
            return label, idx, r_graph, p_graph, r_map
        else:
            return label, idx, r_graph, p_graph


    def process(self):
        atoms  = np.array(['He', 'He', 'He', 'He'])
        rcoords = np.array([[0,  0, 0,], [1,  0, 0], [0,  2, 0], [1,  2, 0]])
        pcoords = np.array([[0,  0, 0,], [1,  0, 0], [-1,  0, 0], [0,  2, 0]])
        mol = Chem.MolFromSmiles('.'.join([f'[{a}]' for a in atoms]))
        self.reactant_graphs = [get_graph(mol, atoms, rcoords, 0)]
        self.product_graphs  = [get_graph(mol, atoms, pcoords, 0)]
        self.reactants_maps  = [torch.arange(len(atoms))]


    def process_chirality_test(self):
        atoms = ['C', 'H', 'F', 'Cl', 'Br']
        mol = Chem.MolFromSmiles('.'.join([f'[{a}]' for a in atoms]))
        import io
        f_handler = io.StringIO(""" 0.05928331  -0.12694591  -0.06683713
 0.10695432  -0.39791163  -1.12442036
 0.21441193  -1.20602809   0.72154781
 1.36247485   1.07091085   0.25382473
-1.74312440   0.65997478   0.21588495""")
        rcoords = np.loadtxt(f_handler)
        pcoords = -rcoords
        f_handler.close()
        self.reactant_graphs = [get_graph(mol, atoms, rcoords, 0)]
        self.product_graphs  = [get_graph(mol, atoms, pcoords, 0)]
        self.reactants_maps  = [torch.arange(len(atoms))]
