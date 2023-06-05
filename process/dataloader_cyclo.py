import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from process.create_graph import reader, get_graph, canon_mol


class Cyclo23TS(Dataset):
    def __init__(self, files_dir='data/cyclo/xyz/', csv_path='data/cyclo/mod_dataset.csv',
                 map_dir='data/cyclo/matches/',
                 radius=20, max_neighbor=24, processed_dir='data/cyclo/processed/', process=True,
                 atom_mapping=False):

        self.max_number_of_reactants = 2
        self.max_number_of_products = 1

        self.max_neighbor = max_neighbor
        self.radius = radius
        self.files_dir = files_dir + '/'
        self.processed_dir = processed_dir + '/'
        self.map_dir = map_dir
        self.atom_mapping = atom_mapping

        print("Loading data into memory...")

        self.df = pd.read_csv(csv_path)
        labels = torch.tensor(self.df['G_act'].values)

        mean = torch.mean(labels)
        std = torch.std(labels)
        self.std = std
        #TODO to normalise in train/test/val split
        self.labels = (labels - mean)/std

        indices = self.df['rxn_id'].to_list()
        self.indices = indices

        if not self.atom_mapping:
            if (not os.path.exists(os.path.join(self.processed_dir, 'reactant_0_graphs.pt')) or
                    not os.path.exists(os.path.join(self.processed_dir, 'reactant_1_graphs.pt')) or
                    not os.path.exists(os.path.join(self.processed_dir, 'product_graphs.pt'))):
                print("processed data not found, processing data...")
                self.process()

            elif process == True:
                print("processing by request...")
                self.process()

            else:
                self.reactant_0_graphs = torch.load(os.path.join(self.processed_dir, 'reactant_0_graphs.pt'))
                self.reactant_1_graphs = torch.load(os.path.join(self.processed_dir, 'reactant_1_graphs.pt'))
                self.product_graphs = torch.load(os.path.join(self.processed_dir, 'product_graphs.pt'))
                print(f"Coords and graphs successfully read from {self.processed_dir}")
        else:
            if (not os.path.exists(os.path.join(self.processed_dir, 'reactant_0_graphs.pt')) or
                    not os.path.exists(os.path.join(self.processed_dir, 'reactant_1_graphs.pt')) or
                    not os.path.exists(os.path.join(self.processed_dir, 'product_graphs.pt')) or
                    not os.path.exists(os.path.join(self.processed_dir, 'reactant_0_maps.pt')) or
                    not os.path.exists(os.path.join(self.processed_dir, 'reactant_1_maps.pt'))):
                print("processed data not found, processing data...")
                self.process()

            elif process == True:
                print("processing by request...")
                self.process()

            else:
                self.reactant_0_graphs = torch.load(os.path.join(self.processed_dir, 'reactant_0_graphs.pt'))
                self.reactant_1_graphs = torch.load(os.path.join(self.processed_dir, 'reactant_1_graphs.pt'))
                self.reactant_0_maps = torch.load(self.processed_dir + 'reactant_0_maps.pt')
                self.reactant_1_maps = torch.load(self.processed_dir + 'reactant_1_maps.pt')
                self.product_graphs = torch.load(os.path.join(self.processed_dir, 'product_graphs.pt'))
                print(f"Coords and graphs successfully read from {self.processed_dir}")


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        r_0_graph = self.reactant_0_graphs[idx]
        r_1_graph = self.reactant_1_graphs[idx]
        p_graph = self.product_graphs[idx]
        label = self.labels[idx]
        if self.atom_mapping:
            r_0_map = self.reactant_0_maps[idx]
            r_1_map = self.reactant_1_maps[idx]
            return label, idx, r_0_graph, r_1_graph, p_graph, np.hstack((r_0_map, r_1_map))
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
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            print(f"Creating processed directory {self.processed_dir}")

        # two reactants
        r0coords = []
        r0atoms = []
        r0maps = []

        r1coords = []
        r1atoms = []
        r1maps = []

        # one product
        pcoords = []
        patoms = []

        for idx in tqdm(self.indices, desc='reading xyz files'):
            rxn_dir = self.files_dir + str(idx) + '/'

            reactants = self.get_r_files(rxn_dir)

            r_0_atomtypes, r_0_coords = reader(reactants[0])
            r0atoms.append(r_0_atomtypes)
            r0coords.append(r_0_coords)
            r0maps.append(np.loadtxt(f'{self.map_dir}/R1_{idx}.dat', dtype=int))
            assert len(r0atoms[-1])==len(r0maps[-1]), 'different number of atoms in the xyz and mapping files'

            r_1_atomtypes, r_1_coords = reader(reactants[1])
            r1atoms.append(r_1_atomtypes)
            r1coords.append(r_1_coords)
            r1maps.append(np.loadtxt(f'{self.map_dir}/R0_{idx}.dat', dtype=int))
            assert len(r1atoms[-1])==len(r1maps[-1]), 'different number of atoms in the xyz and mapping files'

            p_file = glob(rxn_dir + "p*.xyz")[0]
            p_atomtypes, p_coords = reader(p_file)
            patoms.append(p_atomtypes)
            pcoords.append(p_coords)

            assert np.all(np.hstack((r0atoms[-1], r1atoms[-1])) == patoms[-1][np.hstack((r0maps[-1], r1maps[-1]))]), 'mapping leads to atom-type mismatch'

        assert len(pcoords)  == len(r0coords), 'not as many reactants 0 as products'
        assert len(r1coords) == len(r0coords), 'not as many reactants 0 as reactants 1'
        assert len(patoms)   == len(pcoords),  'not as many atomtypes as coords'
        assert len(r0atoms)  == len(r0coords), 'not as many atomtypes as coords'
        assert len(r1atoms)  == len(r1coords), 'not as many atomtypes as coords'
        assert len(r0atoms)  == len(r0maps),   'not as many atomtypes as mappings'
        assert len(r1atoms)  == len(r1maps),   'not as many atomtypes as mappings'

        print(f"Processing csv file and saving graphs to {self.processed_dir}")
        self.reactant_0_graphs = []
        self.reactant_1_graphs = []
        self.reactant_0_maps = []
        self.reactant_1_maps = []
        self.product_graphs = []
        for i, idx in enumerate(tqdm(self.indices, desc="making graphs")):
            # IMPORTANT : in db they are inconsistent about what is r0 and what is r1.
            # current soln is to check both. not ideal.
            rxnsmi = self.df[self.df['rxn_id'] == idx]['rxn_smiles'].item()
            rsmis, psmi = rxnsmi.split('>>')
            rsmi_0, rsmi_1 = rsmis.split('.')
            try:
                r_graph_0 = self.make_graph(rsmi_0, r0atoms[i], r0coords[i], self.labels[i], idx)
                r_graph_1 = self.make_graph(rsmi_1, r1atoms[i], r1coords[i], self.labels[i], idx)
                r_map_0 = r0maps[i]
                r_map_1 = r1maps[i]
            except:  # switch r0/r1 in atoms & coordinates & maps
                r_graph_0 = self.make_graph(rsmi_0, r1atoms[i], r1coords[i], self.labels[i], idx)
                r_graph_1 = self.make_graph(rsmi_1, r0atoms[i], r0coords[i], self.labels[i], idx)
                r_map_0 = r1maps[i]
                r_map_1 = r0maps[i]
            p_graph = self.make_graph(psmi, patoms[i], pcoords[i], self.labels[i], idx)

            self.reactant_0_graphs.append(r_graph_0)
            self.reactant_1_graphs.append(r_graph_1)
            self.reactant_0_maps.append(r_map_0)
            self.reactant_1_maps.append(r_map_1)
            self.product_graphs.append(p_graph)

        assert len(self.reactant_0_graphs) == len(self.reactant_1_graphs), 'number of reactants dont match'
        assert len(self.reactant_1_graphs) == len(self.product_graphs), 'number of reactants and products dont match'

        r0graphsavename = self.processed_dir + 'reactant_0_graphs.pt'
        r1graphsavename = self.processed_dir + 'reactant_1_graphs.pt'
        r0mapsavename = self.processed_dir + 'reactant_0_maps.pt'
        r1mapsavename = self.processed_dir + 'reactant_1_maps.pt'
        pgraphsavename  = self.processed_dir + 'product_graphs.pt'

        torch.save(self.reactant_0_graphs, r0graphsavename)
        torch.save(self.reactant_1_graphs, r1graphsavename)
        torch.save(self.reactant_0_maps, r0mapsavename)
        torch.save(self.reactant_1_maps, r1mapsavename)
        torch.save(self.product_graphs, pgraphsavename)
        print(f"Saved graphs to {r0graphsavename}, {r1graphsavename} and {pgraphsavename}")


    def make_graph(self, smi, atoms, coords, label, idx, check=True):
        mol = Chem.MolFromSmiles(smi)
        mol = canon_mol(mol)
        assert mol is not None, f"mol obj {idx} is None from smi {smi}"
        ats = [at.GetSymbol() for at in mol.GetAtoms()]
        assert len(ats) == len(atoms), f"nats don't match in idx {idx}"
        if check:
            assert np.all(ats == atoms), "atomtypes don't match"
        return get_graph(mol, atoms, coords, label, radius=self.radius, max_neighbor=self.max_neighbor)


    def get_r_files(self, rxn_dir):
        reactants = sorted(glob(rxn_dir +"r*.xyz"), reverse=True)
        reactants = self.check_alt_files(reactants)
        assert len(reactants) == 2, f"Inconsistent length of {len(reactants)}"
        # inverse labelling in their xyz files
        r0_label = reactants[0].split("/")[-1][:2]
        assert r0_label == 'r1', 'not r0'
        r1_label = reactants[1].split("/")[-1][:2]
        assert r1_label == 'r0', 'not r1'
        return reactants
