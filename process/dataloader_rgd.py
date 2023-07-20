import os
from os.path import exists, join
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import h5py
from process.create_graph import get_graph, get_empty_graph


class RGD1(Dataset):

    def __init__(self, files_dir='data/rgd1/',
                 radius=20, max_neighbor=24, processed_dir='data/rgd1/processed/', process=True,
                 split_complexes=False,
                 noH=False, atom_mapping=False, rxnmapper=False, reverse=False):

        h5_path = files_dir + '/RGD1_CHNO.h5'
        csv_path = files_dir + '/RGD1CHNO_smiles.csv'

        self.version = 1  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        if noH or rxnmapper or reverse:
            raise NotImplementedError
        if split_complexes:
            self.max_number_of_reactants = 4
            self.max_number_of_products = 4
        else:
            self.max_number_of_reactants = 1
            self.max_number_of_products = 1

        self.max_neighbor = max_neighbor
        self.radius = radius
        self.processed_dir = processed_dir + '/'
        self.h5_path = h5_path
        self.atom_mapping = atom_mapping
        self.noH = noH
        self.split_complexes = split_complexes

        dataset_prefix = os.path.splitext(os.path.basename(h5_path))[0]
        if split_complexes:
            dataset_prefix += '.split'
        if noH:
            dataset_prefix += '.noH'
        self.paths = SimpleNamespace(
                rg = join(self.processed_dir, f'{dataset_prefix}.v{self.version}.reactants_graphs.pt'),
                pg = join(self.processed_dir, f'{dataset_prefix}.v{self.version}.products_graphs.pt'),
                mp = join(self.processed_dir, f'{dataset_prefix}.v{self.version}.p2r_mapping.pt'),
                )

        print("Loading data into memory...")

        self.df = pd.read_csv(csv_path)
        self.nreactions = len(self.df)
        self.indices = self.df['reaction'].to_list()
        self.labels = torch.tensor(self.df['DE_F'].values)

        if process == True:
            print("Processing by request...")
            self.process()
        else:
            if exists(self.paths.rg) and exists(self.paths.pg) and exists(self.paths.mp):
                self.reactants_graphs = torch.load(self.paths.rg)
                self.products_graphs = torch.load(self.paths.pg)
                self.p2r_maps        = torch.load(self.paths.mp)
                print(f"Coords and graphs successfully read from {self.processed_dir}")
            else:
                print("Processed data not found, processing data...")
                self.process()

        self.standardize_labels()


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        r = self.reactants_graphs[idx]
        p = self.products_graphs[idx]
        label = self.labels[idx]
        if self.atom_mapping:
            return label, idx, *r, *p, self.p2r_maps[idx]
        else:
            return label, idx, *r, *p


    def process(self):

        print(f"Processing xyz files and saving coords to {self.processed_dir}")
        if not exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            print(f"Creating processed directory {self.processed_dir}")

        hf = h5py.File(self.h5_path, 'r')

        mend = Chem.GetPeriodicTable()
        self.empty = get_empty_graph()

        self.products_graphs = []
        self.reactants_graphs = []
        self.p2r_maps = []

        for i, idx in enumerate(tqdm(self.indices, desc="making graphs")):
            rxn = hf[idx]
            atomtypes = np.array([*map(lambda x: mend.GetElementSymbol(int(x)), rxn.get('elements'))])
            reactant_coords = np.array(rxn.get('RG'))
            product_coords  = np.array(rxn.get('PG'))
            assert len(reactant_coords) == len(atomtypes), f'{idx}'
            assert len(product_coords) == len(atomtypes), f'{idx}'

            dfrow = self.df[self.df['reaction'] == idx]
            rsmi = dfrow['reactant'].item()
            psmi = dfrow['product'].item()

            if self.split_complexes:
                rsmis = rsmi.split('.')
                psmis = psmi.split('.')
            else:
                rsmis = [rsmi]
                psmis = [psmi]

            rgraphs, ratoms, rmaps = self.make_graph_wrapper(rsmis, reactant_coords, self.max_number_of_reactants, f'r{idx}', atomtypes, i)
            pgraphs, patoms, pmaps = self.make_graph_wrapper(psmis, product_coords, self.max_number_of_products, f'p{idx}', atomtypes, i)
            assert len(ratoms)==len(atomtypes), f'wrong number of reactant atoms in {idx}'
            assert len(patoms)==len(atomtypes), f'wrong number of product atoms in {idx}'
            self.reactants_graphs.append(rgraphs)
            self.products_graphs.append(pgraphs)

            assert np.all(sorted(rmaps)==sorted(pmaps)), f'atoms missing from mapping {idx}'
            p2rmap = np.hstack([np.where(pmaps==j)[0] for j in rmaps])
            assert np.all(rmaps == pmaps[p2rmap])
            assert np.all(ratoms == patoms[p2rmap])
            self.p2r_maps.append(p2rmap)

        torch.save(self.reactants_graphs, self.paths.rg)
        torch.save(self.products_graphs, self.paths.pg)
        torch.save(self.p2r_maps, self.paths.mp)
        print(f"Saved graphs to {self.paths.rg} and {self.paths.pg}")


    def make_graph_wrapper(self, smis, coords, nmol_max, tag, atomtypes, ireact):
        graphs = []
        maps = []
        atoms = []
        nmol = len(smis)
        assert nmol <= nmol_max, f'{tag}: too many molecules'
        for i, smi in enumerate(smis):
            graph, mapping, atom = self.make_graph(smi, atomtypes, coords, ireact, f'{tag}_{i}')
            graphs.append(graph)
            maps.append(mapping)
            atoms.append(atom)
        padding = [self.empty] * (nmol_max-nmol)
        return graphs+padding, np.hstack(atoms), np.hstack(maps)


    def make_graph(self, smi, atoms, coords, ireact, idx):
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        assert mol is not None, f"mol obj {idx} is None from smi {smi}"
        Chem.SanitizeMol(mol)

        atom_map = np.array([at.GetAtomMapNum() for at in mol.GetAtoms()])
        assert np.all(atom_map > 0), f"mol {idx} is not atom-mapped"
        atom_map -= 1

        new_atoms = atoms[atom_map]
        new_coords = coords[atom_map]

        graph = get_graph(mol, new_atoms, new_coords, ireact,
                          radius=self.radius, max_neighbor=self.max_neighbor)
        return graph, atom_map, new_atoms


    def standardize_labels(self):
        mean = torch.mean(self.labels)
        std = torch.std(self.labels)
        self.std = std
        self.labels = (self.labels - mean)/std
