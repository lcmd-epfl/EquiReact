import os
from os.path import exists, join
from glob import glob
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import networkx
import networkx.algorithms.isomorphism as iso
from process.create_graph import reader, get_graph, canon_mol, get_empty_graph


class GDB722TS(Dataset):

    def __init__(self, files_dir='data/gdb7-22-ts/xyz/', csv_path='data/gdb7-22-ts/ccsdtf12_dz_cleaned.csv',
                 radius=20, max_neighbor=24, processed_dir='data/gdb7-22-ts/processed/', process=True):

        self.version = 4.0  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.max_number_of_reactants = 1
        self.max_number_of_products = 3

        self.max_neighbor = max_neighbor
        self.radius = radius
        self.files_dir = files_dir + '/'
        self.processed_dir = processed_dir + '/'

        self.paths = SimpleNamespace(
                rg = join(self.processed_dir, 'reactant_graphs.pt'),
                pg = join(self.processed_dir, 'products_graphs.pt'),
                v  = join(self.processed_dir, 'version.pt')
                )

        print("Loading data into memory...")

        self.df = pd.read_csv(csv_path)
        labels = torch.tensor(self.df['dE0'].values)

        mean = torch.mean(labels)
        std = torch.std(labels)
        self.std = std
        #TODO to normalise in train/test/val split
        self.labels = (labels - mean)/std

        self.indices = self.df['idx'].to_list()

        if process == True:
            print("Processing by request...")
            self.process()
        elif not exists(self.paths.v) or torch.load(self.paths.v) != self.version:
            print("Processed data is outdated, processing data...")
            self.process()
        else:
            if exists(self.paths.rg) and exists(self.paths.pg):
                self.reactant_graphs = torch.load(self.paths.rg)
                self.products_graphs = torch.load(self.paths.pg)
                print(f"Coords and graphs successfully read from {self.processed_dir}")
            else:
                print("Processed data not found, processing data...")
                self.process()


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        r = self.reactant_graphs[idx]
        p = self.products_graphs[idx]
        label = self.labels[idx]
        return label, idx, r, *p


    def process(self):

        print(f"Processing xyz files and saving coords to {self.processed_dir}")
        if not exists(self.processed_dir):
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
        print(f"Processing csv file and saving graphs to {self.processed_dir}")

        empty = get_empty_graph()

        self.products_graphs = []
        self.reactant_graphs = []
        for i, idx in enumerate(tqdm(self.indices, desc="making graphs")):
            rxnsmi = self.df[self.df['idx'] == idx]['rxn_smiles'].item()
            rsmi, psmis = rxnsmi.split('>>')
            # reactant
            self.reactant_graphs.append(self.make_graph(rsmi, reactant_atomtypes_list[i], reactant_coords_list[i], i, f'r{idx:06d}'))
            # products
            psmis = psmis.split('.')
            nprod = len(products_coords_list[i])
            assert len(psmis) == nprod, 'number of products doesnt match'
            pgraphs = [self.make_graph(*args, i, f'p{idx:06d}_{ip}') for ip, args in enumerate(zip(psmis, products_atomtypes_list[i], products_coords_list[i]))]
            padding = [empty] * (self.max_number_of_products-nprod)
            self.products_graphs.append(pgraphs + padding)

        assert len(self.reactant_graphs) == len(self.products_graphs), 'not as many products as reactants'

        torch.save(self.reactant_graphs, self.paths.rg)
        torch.save(self.products_graphs, self.paths.pg)
        torch.save(self.version, self.paths.v)
        print(f"Saved graphs to {self.paths.rg} and {self.paths.pg}")


    def get_xyz_bonds(self, nbonds, atoms, coords):
        def get_xyz_bonds_inner(atoms, coords, rscal=1.0):
            rad = {'H': 0.455, 'C':  0.910, 'N': 0.845, 'O': 0.780}
            xyz_bonds = []
            for i1, (a1, r1) in enumerate(zip(atoms, coords)):
                for i2, (a2, r2) in enumerate(list(zip(atoms, coords))[i1+1:], start=i1+1):
                    rmax = (rad[a1]+rad[a2]) * rscal
                    if np.linalg.norm(r1-r2) < rmax:
                        xyz_bonds.append((i1, i2))
            return np.array(xyz_bonds)
        for rscal in [1.0, 1.1, 1.0/1.1, 1.05, 1.0/1.05]:
            xyz_bonds = get_xyz_bonds_inner(atoms, coords, rscal=rscal)
            if len(xyz_bonds) == nbonds:
                return xyz_bonds
        else:
            return None


    def make_nx_graph(self, atoms, bonds):
        G = networkx.Graph()
        G.add_nodes_from([(i, {'q': q}) for i, q in enumerate(atoms)])
        G.add_edges_from(bonds)
        return G


    def make_graph(self, smi, atoms, coords, ireact, idx):
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        assert mol is not None, f"mol obj {idx} is None from smi {smi}"
        Chem.SanitizeMol(mol)

        assert len(atoms)==mol.GetNumAtoms(), f"nats don't match in idx {idx}"

        rdkit_bonds = np.array(sorted(sorted((i.GetBeginAtomIdx(), i.GetEndAtomIdx())) for i in mol.GetBonds()))
        rdkit_atoms = np.array([at.GetSymbol() for at in mol.GetAtoms()])

        xyz_bonds = self.get_xyz_bonds(len(rdkit_bonds), atoms, coords)
        assert xyz_bonds is not None, f"different number of bonds in {idx}"

        if np.all(rdkit_atoms==atoms) and np.all(rdkit_bonds==xyz_bonds):
            # Don't search for a match because the first one doesn't have to be the shortest one
            new_atoms = atoms
            new_coords = coords
        else:
            G1 = self.make_nx_graph(rdkit_atoms, rdkit_bonds)
            G2 = self.make_nx_graph(atoms, xyz_bonds)
            GM = iso.GraphMatcher(G1, G2, node_match=iso.categorical_node_match('q', None))
            assert GM.is_isomorphic(), f"smiles and xyz graphs are not isomorphic in {idx}"

            match = next(GM.match())
            src, dst = np.array(sorted(match.items(), key=lambda match: match[0])).T
            assert np.all(src==np.arange(G1.number_of_nodes()))

            new_atoms = atoms[dst]
            new_coords = coords[dst]

        return get_graph(mol, new_atoms, new_coords, self.labels[ireact],
                         radius=self.radius, max_neighbor=self.max_neighbor)
