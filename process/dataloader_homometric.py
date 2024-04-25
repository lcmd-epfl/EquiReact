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

