import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from process.create_graph import reader, get_graph, canon_mol, atom_featurizer


class GDB722TS(Dataset):
    def __init__(self, files_dir='data/gdb7-22-ts/xyz/', csv_path='data/gdb7-22-ts/ccsdtf12_dz.csv',
                 radius=20, max_neighbor=24, processed_dir='data/gdb7-22-ts/processed/', process=True):

        self.max_number_of_reactants = 1
        self.max_number_of_products = 3

        self.max_neighbor = max_neighbor
        self.radius = radius
        self.files_dir = files_dir + '/'
        self.processed_dir = processed_dir + '/'

        print("Loading data into memory...")

        self.df = pd.read_csv(csv_path)
        labels = torch.tensor(self.df['dE0'].values)

        mean = torch.mean(labels)
        std = torch.std(labels)
        self.std = std
        #TODO to normalise in train/test/val split
        self.labels = (labels - mean)/std

        self.indices = self.df['idx'].to_list()

        if (not os.path.exists(os.path.join(self.processed_dir, 'reactant_graphs.pt')) or
            not os.path.exists(os.path.join(self.processed_dir, 'products_graphs.pt'))):
            print("Processed data not found, processing data...")
            self.process()

        elif process == True:
            print("Processing by request...")
            self.process()

        else:
            self.reactant_graphs = torch.load(os.path.join(self.processed_dir, 'reactant_graphs.pt'))
            self.products_graphs = torch.load(os.path.join(self.processed_dir, 'products_graphs.pt'))
            print(f"Coords and graphs successfully read from {self.processed_dir}")
        print()


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r = self.reactant_graphs[idx]
        p = self.products_graphs[idx]
        label = self.labels[idx]
        return label, idx, r, *p


    def process(self):

        print(f"Processing xyz files and saving coords to {self.processed_dir}")
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            print(f"Creating processed directory {self.processed_dir}")

        reactant_coords_list    = []
        reactant_atomtypes_list = []
        products_coords_list    = []
        products_atomtypes_list = []

        for idx in tqdm(self.indices, desc='reading xyz files'):
            rxn_dir = f'{self.files_dir}{idx:06}/'
            # 1 reactant
            r_file = f'{rxn_dir}r{idx:06}.xyz'
            atomtypes, coords = reader(r_file, bohr=True)
            reactant_atomtypes_list.append(atomtypes)
            reactant_coords_list.append(coords)
            # multiple products
            p_files = sorted(glob(rxn_dir +"p*.xyz"))
            products_atomtypes_list.append([])
            products_coords_list.append([])
            assert len(p_files) <= self.max_number_of_products, 'more products than the maximum number of products'
            for p_file in p_files:
                atomtypes, coords = reader(p_file, bohr=True)
                products_atomtypes_list[-1].append(atomtypes)
                products_coords_list[-1].append(coords)

        assert len(reactant_coords_list)    == len(products_coords_list), 'not as many products as reactants'
        assert len(products_atomtypes_list) == len(products_coords_list), 'not as many atomtypes as coords for products'
        assert len(reactant_atomtypes_list) == len(reactant_coords_list), 'not as many atomtypes as coords for reactants'


        # Graphs
        def make_graph(smi, atoms, coords, idx):
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"mol obj {idx} is None from smi {smi}"
            mol = Chem.AddHs(mol)
            assert len(atoms)==mol.GetNumAtoms(), f"nats don't match in idx {idx}"

            try:
                ats = [at.GetSymbol() for at in mol.GetAtoms()]
                assert np.all(ats == atoms), "atomtypes don't match"
            except:
                try:
                    mol = canon_mol(mol)
                    ats = [at.GetSymbol() for at in mol.GetAtoms()]
                    assert np.all(ats == atoms), "atomtypes don't match"
                except:
                    unq_atoms = np.unique(atoms)
                    # coords from xyz
                    coord_dict = {}
                    element_dict = {}
                    for unq_atom in unq_atoms:
                        for count, i in enumerate(np.where(atoms==unq_atom)[0]):
                            label = unq_atom + str(count+1)
                            coord_dict[label] = coords[i]
                            element_dict[label] = unq_atom

                    ordered_coords = []
                    ordered_elements = []
                    ats = [at.GetSymbol() for at in mol.GetAtoms()]
                    count = {unq_atom: 0 for unq_atom in unq_atoms}
                    for atom in ats:
                        label = atom + str(count[atom]+1)
                        ordered_coords.append(coord_dict[label])
                        ordered_elements.append(element_dict[label])
                        count[atom] += 1

                    assert np.all(ats == ordered_elements), "reordering went wrong"

                    assert len(coords) == len(ordered_coords), "coord lengths don't match"
                    coords = np.array(ordered_coords)
                    assert len(atoms) == len(ats), "atoms lengths don't match"
                    atoms = np.array(ats)
            return get_graph(mol, atoms, coords, self.labels[idx],
                             radius=self.radius, max_neighbor=self.max_neighbor)

        print(f"Processing csv file and saving graphs to {self.processed_dir}")

        products_graphs_list = []
        reactant_graphs_list = []

        num_node_feat = atom_featurizer(Chem.MolFromSmiles('C')).shape[-1]
        empty = Data(x=torch.zeros((0, num_node_feat)), edge_index=torch.zeros((2,0)),
                     edge_attr=torch.zeros(0), y=torch.tensor(0.0), pos=torch.zeros((0,3)))

        for i, idx in enumerate(tqdm(self.indices, desc="making graphs")):
            #print(f'{idx=}')
            rxnsmi = self.df[self.df['idx'] == idx]['rxn_smiles'].item()
            rsmi, psmis = rxnsmi.split('>>')

            # reactant
            reactant_graphs_list.append(make_graph(rsmi, reactant_atomtypes_list[i], reactant_coords_list[i], i))

            # products
            psmis = psmis.split('.')
            nprod = len(products_coords_list[i])
            assert len(psmis) == nprod, 'number of products doesnt match'
            pgraphs = [make_graph(*args, i) for args in zip(psmis, products_atomtypes_list[i], products_coords_list[i])]
            padding = [empty] * (self.max_number_of_products-nprod)
            products_graphs_list.append(pgraphs + padding)

        assert len(reactant_graphs_list) == len(products_graphs_list), 'not as many products as reactants'

        self.reactant_graphs = reactant_graphs_list
        self.products_graphs = products_graphs_list
        rgraphsavename = self.processed_dir + 'reactant_graphs.pt'
        pgraphsavename = self.processed_dir + 'products_graphs.pt'
        torch.save(reactant_graphs_list, rgraphsavename)
        torch.save(products_graphs_list, pgraphsavename)
        print(f"Saved graphs to {rgraphsavename} and {pgraphsavename}")
