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
from process.create_graph import reader, get_graph, canon_mol


class Cyclo23TS(Dataset):
    def __init__(self, files_dir='data/cyclo/xyz/', csv_path='data/cyclo/mod_dataset.csv',
                 map_dir='data/cyclo/matches/',
                 processed_dir='data/cyclo/processed/', process=True,
                 noH=False,
                 atom_mapping=False):

        self.version = 2.6  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.max_number_of_reactants = 2
        self.max_number_of_products = 1

        self.files_dir = files_dir + '/'
        self.processed_dir = processed_dir + '/'
        self.map_dir = map_dir
        self.atom_mapping = atom_mapping
        self.noH = noH

        dataset_prefix = os.path.splitext(os.path.basename(csv_path))[0]
        if noH:
            dataset_prefix += '.noH'
        dataset_prefix += f'.v{self.version}'
        print(f'{dataset_prefix=}')

        self.paths = SimpleNamespace(
                r0g = join(self.processed_dir, f'{dataset_prefix}.reactant_0_graphs.pt'),
                r1g = join(self.processed_dir, f'{dataset_prefix}.reactant_1_graphs.pt'),
                pg  = join(self.processed_dir, f'{dataset_prefix}.product_graphs.pt'),
                rm  = join(self.processed_dir, f'{dataset_prefix}.reactants_maps.pt'),
                )

        print("Loading data into memory...")

        self.df = pd.read_csv(csv_path)
        self.labels = torch.tensor(self.df['G_act'].values)
        indices = self.df['rxn_id'].to_list()
        self.indices = indices
        self.nreactions = len(self.labels)

        if process == True:
            print("processing by request...")
            self.process()
        else:
            if self.atom_mapping:
                if exists(self.paths.r0g) and exists(self.paths.r1g) and exists(self.paths.pg) and exists(self.paths.rm):
                    self.reactant_0_graphs = torch.load(self.paths.r0g)
                    self.reactant_1_graphs = torch.load(self.paths.r1g)
                    self.product_graphs    = torch.load(self.paths.pg)
                    self.reactants_maps    = torch.load(self.paths.rm)
                    print(f"Coords and graphs successfully read from {self.processed_dir}")
                else:
                    print("processed data not found, processing data...")
                    self.process()
            else:
                if exists(self.paths.r0g) and exists(self.paths.r1g) and exists(self.paths.pg):
                    self.reactant_0_graphs = torch.load(self.paths.r0g)
                    self.reactant_1_graphs = torch.load(self.paths.r1g)
                    self.product_graphs    = torch.load(self.paths.pg)
                    print(f"Coords and graphs successfully read from {self.processed_dir}")
                else:
                    print("processed data not found, processing data...")
                    self.process()

        self.standardize_labels()


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        r_0_graph = self.reactant_0_graphs[idx]
        r_1_graph = self.reactant_1_graphs[idx]
        p_graph = self.product_graphs[idx]
        label = self.labels[idx]
        if self.atom_mapping:
            r_map = self.reactants_maps[idx]
            return label, idx, r_0_graph, r_1_graph, p_graph, r_map
        else:
            return label, idx, r_0_graph, r_1_graph, p_graph


    def check_alt_files(self, list_files):
        files = []
        if len(list_files) < 3:
            return list_files
        for file in list_files:
            if "_alt" in file:
                dup_file_label = file.split("_alt.xyz")[0]
        for file in list_files:
            if dup_file_label in file:
                if "_alt" in file:
                    files.append(file)
            else:
                files.append(file)
        return files


    def process(self):
        print(f"Processing xyz files and saving coords to {self.processed_dir}")
        if not exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            print(f"Creating processed directory {self.processed_dir}")

        self.reactant_0_graphs = []
        self.reactant_1_graphs = []
        self.reactants_maps = []
        self.product_graphs = []

        print(f"Processing csv file and saving graphs to {self.processed_dir}")
        for i, idx in enumerate(tqdm(self.indices, desc="making graphs")):

            entry = self.df[self.df['rxn_id'] == idx]
            switch = entry['switch_reactants'].item()
            rfiles, mapfiles = self.get_r_files(idx, switch=switch)
            r0atoms, r0coords = reader(rfiles[0])
            r1atoms, r1coords = reader(rfiles[1])
            pfile = glob(f'{self.files_dir}/{idx}/p*.xyz')[0]
            patoms, pcoords = reader(pfile)

            rxnsmi = entry['rxn_smiles'].item()
            rsmis, psmi = rxnsmi.split('>>')
            r0smi, r1smi = rsmis.split('.')

            if self.noH:
                # atom mapping from SMILES
                r0graph, r0atoms, r0coords, r0map = self.make_graph_noH(r0smi, r0atoms, r0coords, idx)
                r1graph, r1atoms, r1coords, r1map = self.make_graph_noH(r1smi, r1atoms, r1coords, idx)
                pgraph, patoms, pcoords, pmap = self.make_graph_noH(psmi, patoms, pcoords, idx)
                rmap = np.hstack((r0map, r1map))
                assert np.all(sorted(rmap)==sorted(pmap)), f'atoms missing from mapping {idx}'
                assert np.all(sorted(pmap)==np.arange(len(patoms))), f'atoms missing from mapping {idx}'
                p2rmap = np.hstack([np.where(pmap==j)[0] for j in rmap])
                assert np.all(rmap == pmap[p2rmap])
            else:
                # atom mapping from files
                r0graph = self.make_graph(r0smi, r0atoms, r0coords, idx)
                r1graph = self.make_graph(r1smi, r1atoms, r1coords, idx)
                pgraph = self.make_graph(psmi, patoms, pcoords, idx)
                r0map = np.loadtxt(mapfiles[0], dtype=int)
                r1map = np.loadtxt(mapfiles[1], dtype=int)
                assert len(r0atoms)==len(r0map), 'different number of atoms in the xyz and mapping files'
                assert len(r1atoms)==len(r1map), 'different number of atoms in the xyz and mapping files'
                p2rmap = np.hstack((r0map, r1map))

            assert np.all(np.hstack((r0atoms, r1atoms)) == patoms[p2rmap]), 'mapping leads to atom-type mismatch'

            self.reactant_0_graphs.append(r0graph)
            self.reactant_1_graphs.append(r1graph)
            self.reactants_maps.append(p2rmap)
            self.product_graphs.append(pgraph)

        torch.save(self.reactant_0_graphs, self.paths.r0g)
        torch.save(self.reactant_1_graphs, self.paths.r1g)
        torch.save(self.reactants_maps, self.paths.rm)
        torch.save(self.product_graphs, self.paths.pg)
        print(f"Saved graphs to {self.paths.r0g}, {self.paths.r1g} and {self.paths.pg}")


    def make_graph_noH(self, smi, atoms, coords, idx, check=True):
        mol = Chem.MolFromSmiles(smi)
        mol = canon_mol(mol)
        mol = Chem.RemoveAllHs(mol)
        assert mol is not None, f"mol obj {idx} is None from smi {smi}"
        ats = [at.GetSymbol() for at in mol.GetAtoms()]
#        print()
#        print([at.GetAtomMapNum() for at in mol.GetAtoms()])
#        print(ats)

        mol2 = Chem.MolFromSmiles(smi)
        mapping = np.array([at.GetAtomMapNum() for at in mol2.GetAtoms()])
        G1 = self.make_nx_graph_from_mol(mol)
        G2 = self.make_nx_graph_from_mol(mol2)
        GM = iso.GraphMatcher(G1, G2, node_match=iso.categorical_node_match('q', None))
        assert GM.is_isomorphic(), f"smiles and xyz graphs are not isomorphic in {idx}"
        match = next(GM.match())
        src, dst = np.array(sorted(match.items(), key=lambda match: match[0])).T
        assert np.all(src==np.arange(G1.number_of_nodes()))
        mapping = mapping[dst]

        noH_idx = np.where(atoms!='H')
        atoms = atoms[noH_idx]
#        print(atoms)
        coords = coords[noH_idx]

        assert len(ats) == len(atoms), f"nats don't match in idx {idx}"
        if check:
            assert np.all(ats == atoms), "atomtypes don't match"
        return get_graph(mol, atoms, coords, idx), atoms, coords, mapping-1


    def make_graph(self, smi, atoms, coords, idx, check=True):
        mol = Chem.MolFromSmiles(smi)
        mol = canon_mol(mol)
        assert mol is not None, f"mol obj {idx} is None from smi {smi}"
        ats = [at.GetSymbol() for at in mol.GetAtoms()]
        assert len(ats) == len(atoms), f"nats don't match in idx {idx}"
        if check:
            assert np.all(ats == atoms), "atomtypes don't match"
        return get_graph(mol, atoms, coords, idx)


    def get_r_files(self, idx, switch=False):
        reactants = sorted(glob(f'{self.files_dir}/{idx}/r*.xyz'), reverse=True)
        reactants = self.check_alt_files(reactants)
        assert len(reactants) == 2, f"Inconsistent length of {len(reactants)}"
        # inverse labelling in their xyz files
        r0_label = reactants[0].split("/")[-1][:2]
        assert r0_label == 'r1', 'not r0'
        r1_label = reactants[1].split("/")[-1][:2]
        assert r1_label == 'r0', 'not r1'
        mapfiles = [f'{self.map_dir}/R1_{idx}.dat', f'{self.map_dir}/R0_{idx}.dat']
        if switch:
            reactants = reactants[::-1]
            mapfiles = mapfiles[::-1]
        return reactants, mapfiles


    def standardize_labels(self):
        mean = torch.mean(self.labels)
        std = torch.std(self.labels)
        self.std = std
        self.labels = (self.labels - mean)/std


    def make_nx_graph_from_mol(self, mol):
        bonds = np.array(sorted(sorted((i.GetBeginAtomIdx(), i.GetEndAtomIdx())) for i in mol.GetBonds()))
        atoms = np.array([at.GetSymbol() for at in mol.GetAtoms()])
        G = networkx.Graph()
        G.add_nodes_from([(i, {'q': q}) for i, q in enumerate(atoms)])
        G.add_edges_from(bonds)
        return G
