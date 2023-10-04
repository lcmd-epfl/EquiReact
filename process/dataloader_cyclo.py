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
    def __init__(self, csv_path='data/cyclo/cyclo.csv',
                 map_dir='data/cyclo/matches/',
                 processed_dir='data/cyclo/processed/', process=True,
                 xtb=False,
                 noH=False, rxnmapper=False, atom_mapping=False):

        self.version = 5  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.max_number_of_reactants = 2
        self.max_number_of_products = 1

        self.processed_dir = processed_dir + '/'
        self.map_dir = map_dir
        self.atom_mapping = atom_mapping
        self.noH = noH
        self.xtb = xtb
        if rxnmapper:
            self.column = 'rxn_smiles_rxnmapper'
        else:
            self.column = 'rxn_smiles_mapped'
        if rxnmapper and not noH:
            raise RuntimeError
        if xtb:
            self.files_dir='data/cyclo/xyz-xtb/'
        else:
            self.files_dir='data/cyclo/xyz/'

        dataset_prefix = os.path.splitext(os.path.basename(csv_path))[0]+'.'+self.column
        if noH:
            dataset_prefix += '.noH'
        if xtb:
            dataset_prefix += '.xtb'
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
        if xtb:
            bad_idx = np.loadtxt('data/cyclo/bad-xtb.dat', dtype=int)
            for idx in bad_idx:
                self.df.drop(self.df[self.df['rxn_id']==idx].index, axis=0, inplace=True)
        self.labels = torch.tensor(self.df['G_act'].values)
        indices = self.df['rxn_id'].to_list()
        self.indices = indices
        self.nreactions = len(self.labels)

        if process == True:
            print("processing by request...")
            self.process()
        else:
            if exists(self.paths.r0g) and exists(self.paths.r1g) and exists(self.paths.pg) and exists(self.paths.rm):
                self.reactant_0_graphs = torch.load(self.paths.r0g)
                self.reactant_1_graphs = torch.load(self.paths.r1g)
                self.product_graphs    = torch.load(self.paths.pg)
                self.reactants_maps    = torch.load(self.paths.rm)
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
        for idx in tqdm(self.indices, desc="making graphs"):

            entry = self.df[self.df['rxn_id'] == idx]
            switch = entry['switch_reactants'].item()
            rfiles, mapfiles = self.get_r_files(idx, switch=switch)
            r0atoms, r0coords = reader(rfiles[0])
            r1atoms, r1coords = reader(rfiles[1])
            if self.xtb:
                pfile = f'{self.files_dir}/Product_{idx}.xyz'
            else:
                pfile = glob(f'{self.files_dir}/{idx}/p*.xyz')[0]
            patoms, pcoords = reader(pfile)

            # atom mapping is read from files
            # in case of noH/rxnmapper will be reread from smiles
            r0map = np.loadtxt(mapfiles[0], dtype=int)
            r1map = np.loadtxt(mapfiles[1], dtype=int)
            pmap = np.arange(len(patoms))

            rxnsmi = entry[self.column].item()
            rsmis, psmi = rxnsmi.split('>>')
            r0smi, r1smi = rsmis.split('.')

            r0graph, r0atoms, r0map = self.make_graph(r0smi, r0atoms, r0coords, r0map, idx)
            r1graph, r1atoms, r1map = self.make_graph(r1smi, r1atoms, r1coords, r1map, idx)
            pgraph,  patoms,  pmap  = self.make_graph(psmi,  patoms,  pcoords,  pmap,  idx)

            rmap = np.hstack((r0map, r1map))
            p2rmap = np.hstack([np.where(pmap==j)[0] for j in rmap])
            assert np.all(sorted(rmap)==sorted(pmap)), f'atoms missing from mapping {idx}'
            assert np.all(sorted(pmap)==np.arange(len(patoms))), f'atoms missing from mapping {idx}'
            assert np.all(rmap == pmap[p2rmap])
            assert np.all(np.hstack((r0atoms, r1atoms)) == patoms[p2rmap]), f'mapping leads to atom-type mismatch in {idx}'

            self.reactant_0_graphs.append(r0graph)
            self.reactant_1_graphs.append(r1graph)
            self.reactants_maps.append(p2rmap)
            self.product_graphs.append(pgraph)

        torch.save(self.reactant_0_graphs, self.paths.r0g)
        torch.save(self.reactant_1_graphs, self.paths.r1g)
        torch.save(self.reactants_maps, self.paths.rm)
        torch.save(self.product_graphs, self.paths.pg)
        print(f"Saved graphs to {self.paths.r0g}, {self.paths.r1g} and {self.paths.pg}")


    def make_graph(self, smi, atoms, coords, mapping, idx):
        mol = Chem.MolFromSmiles(smi)
        mol = canon_mol(mol)
        assert mol is not None, f"mol obj {idx} is None from smi {smi}"
        atoms, coords, mapping = self.reorder_xyz(mol, atoms, coords, mapping, idx)
        if self.noH:
            # use mapping from SMILES
            mol = Chem.RemoveAllHs(mol)
            mapping = self.reorder_mapping(smi, mol, idx)
            noH_idx = np.where(atoms!='H')
            atoms = atoms[noH_idx]
            coords = coords[noH_idx]
        ats = [at.GetSymbol() for at in mol.GetAtoms()]
        assert len(ats) == len(atoms), f"nats don't match in {idx}"
        assert np.all(ats == atoms), f"atomtypes don't match in {idx}"
        return get_graph(mol, atoms, coords, idx), atoms, mapping


    def reorder_xyz(self, mol, atoms, coords, mapping, idx):
        ats = [at.GetSymbol() for at in mol.GetAtoms()]
        assert len(ats) == len(atoms), f"nats don't match in {idx}"
        G1 = self.make_nx_graph_from_mol(mol)
        G2 = self.make_nx_graph_from_xyz(atoms, coords)
        assert len(G1.edges)==len(G2.edges), f"different number of bonds in {idx}"
        if not (np.all(ats==atoms) and (G1.edges==G2.edges)):
            GM = iso.GraphMatcher(G1, G2, node_match=iso.categorical_node_match('q', None))
            assert GM.is_isomorphic(), f'xyz and smiles are not isomorphic for {idx}'
            match = next(GM.match())
            src, dst = np.array(sorted(match.items(), key=lambda match: match[0])).T
            assert np.all(src==np.arange(G1.number_of_nodes()))
            atoms = atoms[dst]
            coords = coords[dst]
            mapping = mapping[dst]
        assert np.all(ats == atoms), f"atomtypes don't match in {idx}"
        return atoms, coords, mapping


    def reorder_mapping(self, smi, mol, idx):
        mol2 = Chem.MolFromSmiles(smi)
        mapping = np.array([at.GetAtomMapNum() for at in mol2.GetAtoms()])-1
        G1 = self.make_nx_graph_from_mol(mol)
        G2 = self.make_nx_graph_from_mol(mol2)
        GM = iso.GraphMatcher(G1, G2, node_match=iso.categorical_node_match('q', None))
        assert GM.is_isomorphic(), f"smiles and xyz graphs are not isomorphic in {idx}"
        match = next(GM.match())
        src, dst = np.array(sorted(match.items(), key=lambda match: match[0])).T
        assert np.all(src==np.arange(G1.number_of_nodes()))
        mapping = mapping[dst]
        return mapping


    def get_r_files(self, idx, switch=False):
        if self.xtb:
            reactants = [f'{self.files_dir}/Reactant_{idx}_{ir}.xyz' for ir in (0,1)]
        else:
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
            if not self.xtb:
                reactants = reactants[::-1]
            mapfiles = mapfiles[::-1]
        return reactants, mapfiles


    def standardize_labels(self):
        mean = torch.mean(self.labels)
        std = torch.std(self.labels)
        self.std = std
        self.labels = (self.labels - mean)/std


    def make_nx_graph(self, atoms, bonds):
        G = networkx.Graph()
        G.add_nodes_from([(i, {'q': q}) for i, q in enumerate(atoms)])
        G.add_edges_from(bonds)
        return G

    def make_nx_graph_from_mol(self, mol):
        bonds = np.array(sorted(sorted((i.GetBeginAtomIdx(), i.GetEndAtomIdx())) for i in mol.GetBonds()))
        atoms = np.array([at.GetSymbol() for at in mol.GetAtoms()])
        return self.make_nx_graph(atoms, bonds)

    def make_nx_graph_from_xyz(self, atoms, coords):
        bonds = self.get_xyz_bonds(atoms, coords)
        return self.make_nx_graph(atoms, bonds)

    def get_xyz_bonds(self, atoms, coords):
        rad = {'H': 0.455, 'C':  0.910, 'N': 0.845, 'O': 0.780, 'F': 0.650, 'Cl': 1.300, 'Br': 1.495}
        xyz_bonds = []
        for i1, (a1, r1) in enumerate(zip(atoms, coords)):
            for i2, (a2, r2) in enumerate(list(zip(atoms, coords))[i1+1:], start=i1+1):
                rmax = (rad[a1]+rad[a2])
                if np.linalg.norm(r1-r2) < rmax:
                    xyz_bonds.append((i1, i2))
        return np.array(xyz_bonds)
