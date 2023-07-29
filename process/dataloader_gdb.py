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

    def __init__(self,
                 processed_dir='data/gdb7-22-ts/processed/', process=True,
                 xtb=False, noH=False, atom_mapping=False, rxnmapper=False, reverse=False):

        if xtb:
            self.bohr = False
            files_dir = 'data/gdb7-22-ts/xyz-xtb'
            self.xyz_files = lambda x: (f'{files_dir}/{x}/Reactant_{x}_0_opt.xyz',
                                        sorted(glob(f'{files_dir}/{x}/Product_{x}_*_opt.xyz')))
        else:
            self.bohr = True
            files_dir='data/gdb7-22-ts/xyz'
            self.xyz_files = lambda x: (f'{files_dir}/{x:06}/r{x:06}.xyz',
                                        sorted(glob(f'{files_dir}/{x:06}/p{x:06}*.xyz')))

        if rxnmapper is True:
            if noH:
                csv_path = 'data/gdb7-22-ts/rxnmapper-noH.csv'
            else:
                csv_path = 'data/gdb7-22-ts/rxnmapper.csv'
        else:
            csv_path = 'data/gdb7-22-ts/ccsdtf12_dz_cleaned.csv'
        print(f'{csv_path=}')

        self.version = 10  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.max_number_of_reactants = 1
        self.max_number_of_products = 3

        self.processed_dir = processed_dir + '/'
        self.atom_mapping = atom_mapping
        self.noH = noH

        dataset_prefix = os.path.splitext(os.path.basename(csv_path))[0]
        if noH:
            dataset_prefix += '.noH'
        if xtb:
            dataset_prefix += '.xtb'
        dataset_prefix += f'.v{self.version}'
        print(f'{dataset_prefix=}')

        self.paths = SimpleNamespace(
                rg = join(self.processed_dir, f'{dataset_prefix}.reactants_graphs.pt'),
                pg = join(self.processed_dir, f'{dataset_prefix}.products_graphs.pt'),
                mp = join(self.processed_dir, f'{dataset_prefix}.p2r_mapping.pt'),
                )

        print("Loading data into memory...")

        self.df = pd.read_csv(csv_path)
        if xtb:
            bad_idx = np.loadtxt('data/gdb7-22-ts/bad-xtb.dat', dtype=int)
            for idx in bad_idx:
                self.df.drop(self.df[self.df['idx']==idx].index, axis=0, inplace=True)
        self.nreactions = len(self.df)
        self.labels = torch.tensor(self.df['dE0'].values)
        self.indices = self.df['idx'].to_list()

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

        if reverse:
            self.add_reverse()

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

        reactant_coords_list    = []
        reactant_atomtypes_list = []
        products_coords_list    = []
        products_atomtypes_list = []

        for idx in tqdm(self.indices, desc='reading xyz files'):

            r_file, p_files = self.xyz_files(idx)
            # 1 reactant
            atomtypes, coords = reader(r_file, bohr=self.bohr)
            reactant_atomtypes_list.append(atomtypes)
            reactant_coords_list.append(coords)
            # multiple products
            products_atomtypes_list.append([])
            products_coords_list.append([])
            assert len(p_files) <= self.max_number_of_products, 'more products than the maximum number of products'
            for p_file in p_files:
                atomtypes, coords = reader(p_file, bohr=self.bohr)
                products_atomtypes_list[-1].append(atomtypes)
                products_coords_list[-1].append(coords)

        assert len(reactant_coords_list)    == len(products_coords_list), 'not as many products as reactants'
        assert len(products_atomtypes_list) == len(products_coords_list), 'not as many atomtypes as coords for products'
        assert len(reactant_atomtypes_list) == len(reactant_coords_list), 'not as many atomtypes as coords for reactants'

        # Graphs
        print(f"Processing csv file and saving graphs to {self.processed_dir}")

        empty = get_empty_graph()

        self.products_graphs = []
        self.reactants_graphs = []
        self.p2r_maps = []
        for i, idx in enumerate(tqdm(self.indices, desc="making graphs")):
            rxnsmi = self.df[self.df['idx'] == idx]['rxn_smiles'].item()
            rsmi, psmis = rxnsmi.split('>>')

            # reactant
            rgraph, rmap, ratom = self.make_graph(rsmi, reactant_atomtypes_list[i], reactant_coords_list[i], i, f'r{idx:06d}')
            self.reactants_graphs.append([rgraph])

            # products
            psmis = psmis.split('.')
            nprod = len(products_coords_list[i])
            assert len(psmis) == nprod, 'number of products doesnt match'

            pgraphs = []
            pmaps = []
            patoms = []
            for ip, args in enumerate(zip(psmis, products_atomtypes_list[i], products_coords_list[i])):
                pgraph, pmap, patom = self.make_graph(*args, i, f'p{idx:06d}_{ip}')
                if pgraph is None:
                    nprod -= 1
                    continue
                pgraphs.append(pgraph)
                pmaps.append(pmap)
                patoms.append(patom)
            padding = [empty] * (self.max_number_of_products-nprod)
            self.products_graphs.append(pgraphs + padding)

            pmaps = np.hstack(pmaps)
            assert np.all(sorted(rmap)==sorted(pmaps)), f'atoms missing from mapping {idx}'
            if not self.noH:
                assert np.all(sorted(rmap)==np.arange(len(ratom))), f'atoms missing from mapping {idx}'
            p2rmap = np.hstack([np.where(pmaps==j)[0] for j in rmap])
            assert np.all(rmap == pmaps[p2rmap])
            assert np.all(ratom == np.hstack(patoms)[p2rmap])
            self.p2r_maps.append(p2rmap)

            # how to get r2pmap from p2rmap
            r2pmap = np.hstack([np.where(rmap==j)[0] for j in pmaps])
            assert np.all(pmaps == rmap[r2pmap]), f'{idx}'
            assert np.all(ratom[r2pmap] == np.hstack(patoms)), f'{idx}'
            assert np.all(r2pmap == np.argsort(p2rmap))


        assert len(self.reactants_graphs) == len(self.products_graphs), 'not as many products as reactants'

        torch.save(self.reactants_graphs, self.paths.rg)
        torch.save(self.products_graphs, self.paths.pg)
        torch.save(self.p2r_maps, self.paths.mp)
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

        if self.noH:
            natoms_relevant = np.count_nonzero(np.array([at.GetSymbol() for at in mol.GetAtoms()])!='H')
            mol = Chem.RemoveAllHs(mol)
            mol = Chem.AddHs(mol)
            Chem.SanitizeMol(mol)
        else:
            natoms_relevant = mol.GetNumAtoms()
        if natoms_relevant==0:
            return None, None, None

        assert len(atoms)==mol.GetNumAtoms(), f"nats don't match in idx {idx}"

        atom_map = np.array([at.GetAtomMapNum() for at in mol.GetAtoms()])
        assert np.all(atom_map[:natoms_relevant] > 0), f"mol {idx} is not atom-mapped"

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

        if self.noH:
            mol = Chem.RemoveAllHs(mol)
            assert natoms_relevant == mol.GetNumAtoms(), f'different number of atoms before adding/removing Hs and after in {idx}'
            noH_idx = np.where(new_atoms!='H')
            new_atoms = new_atoms[noH_idx]
            new_coords = new_coords[noH_idx]
            atom_map = atom_map[noH_idx]

        graph = get_graph(mol, new_atoms, new_coords, ireact)
        return graph, atom_map-1, new_atoms


    def add_reverse(self):
        max_number_of_mol = max(self.max_number_of_reactants, self.max_number_of_products)
        empty = get_empty_graph()
        for i in range(self.nreactions):
            self.reactants_graphs[i].extend([empty] * (max_number_of_mol - len(self.reactants_graphs[i])))
            self.products_graphs[i].extend([empty] * (max_number_of_mol - len(self.products_graphs[i])))

        self.max_number_of_reactants = max_number_of_mol
        self.max_number_of_products = max_number_of_mol

        self.reactants_graphs += self.products_graphs
        self.products_graphs  += self.reactants_graphs[:self.nreactions]
        self.p2r_maps         += [np.argsort(p2rmap) for p2rmap in self.p2r_maps]
        self.labels = torch.hstack((self.labels, self.labels-torch.tensor(self.df['dHrxn298'].values)))


    def standardize_labels(self):
        #TODO to normalise in train/test/val split
        mean = torch.mean(self.labels)
        std = torch.std(self.labels)
        self.std = std
        self.labels = (self.labels - mean)/std
