from torch.utils.data import Dataset
import torch
import pandas as pd
from process.create_graph import reader, get_graph, count_ats
import os
from glob import glob
from copy import deepcopy
from tqdm import tqdm
import rdkit
from rdkit import Chem


class Cyclo23TS(Dataset):
    def __init__(self, files_dir='data/cyclo/xyz/', csv_path='data/cyclo/full_dataset.csv',
                 radius=20, max_neighbor=24, device='cpu', processed_dir='data/cyclo/processed/', process=True):
        self.device = device
        self.max_neighbor = max_neighbor
        self.radius = radius

        if files_dir[-1] != '/':
            files_dir += '/'
        if processed_dir[-1] != '/':
            processed_dir += '/'

        self.files_dir = files_dir
        self.processed_dir = processed_dir

        print("Loading data into memory...")

        df = pd.read_csv(csv_path)
        labels = torch.tensor(df['G_act'].values)

        self.df = df

        mean = torch.mean(labels)
        std = torch.std(labels)
        self.std = std
        #TODO to normalise in train/test/val split
        self.labels = (labels - mean)/std

        indices = df['rxn_id'].to_list()
        self.indices = indices

        if (not os.path.exists(os.path.join(self.processed_dir, 'reactant_0_graphs.pt')) and
                not os.path.exists(os.path.join(self.processed_dir, 'reactant_1_graphs.pt')) and
                not os.path.exists(os.path.join(self.processed_dir, 'product_graphs.pt')) and
                not os.path.exists(os.path.join(self.processed_dir, 'atomtypes_coords.pt'))):
            print("processed data not found, processing data...")
            self.process()

        elif process == True:
            print("processing by request...")
            self.process()

        else:
            self.reactant_0_graphs = torch.load(os.path.join(self.processed_dir, 'reactant_0_graphs.pt'))
            self.reactant_1_graphs = torch.load(os.path.join(self.processed_dir, 'reactant_1_graphs.pt'))
            self.product_graphs = torch.load(os.path.join(self.processed_dir, 'product_graphs.pt'))
            atomtypes_coords = torch.load(os.path.join(self.processed_dir, 'atomtypes_coords.pt'))
            self.reactant_0_coords = atomtypes_coords['reactant_0_coords']
            self.reactant_0_atomtypes = atomtypes_coords['reactant_0_atomtypes']
            self.reactant_1_coords = atomtypes_coords['reactant_1_coords']
            self.reactant_1_atomtypes = atomtypes_coords['reactant_1_atomtypes']
            self.product_coords = atomtypes_coords['product_coords']
            self.product_atomtypes = atomtypes_coords['product_atomtypes']
            print(f"Coords and graphs successfully read from {self.processed_dir}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r_0_atomtypes = self.reactant_0_atomtypes[idx]
        r_0_coords = self.reactant_0_coords[idx]
        r_0_graph = self.reactant_0_graphs[idx]

        r_1_atomtypes = self.reactant_1_atomtypes[idx]
        r_1_coords = self.reactant_1_coords[idx]
        r_1_graph = self.reactant_1_graphs[idx]

        p_atomtypes = self.product_atomtypes[idx]
        p_coords = self.product_coords[idx]
        p_graph = self.product_graphs[idx]

        label = self.labels[idx]
        return r_0_graph.to(self.device), r_0_atomtypes, r_0_coords, r_1_graph.to(self.device), r_1_atomtypes, r_1_coords, p_graph.to(self.device), p_atomtypes, p_coords, label.to(self.device), idx

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

    def clear_atom_map(self, mol):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return mol

    def process(self):
        print(f"Processing xyz files and saving coords to {self.processed_dir}")
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            print(f"Creating processed directory {self.processed_dir}")

        # two reactants
        reactant_0_coords_list = []
        reactant_0_atomtypes_list = []

        reactant_1_coords_list = []
        reactant_1_atomtypes_list = []

        # one product
        product_coords_list = []
        product_atomtypes_list = []

        for idx in tqdm(self.indices, desc='reading xyz files'):
            rxn_dir = self.files_dir + str(idx) + '/'

            # multiple reactants
            reactants = sorted(glob(rxn_dir +"r*.xyz"), reverse=True)
            reactants = self.check_alt_files(reactants)
            assert len(reactants) == 2, f"Inconsistent length of {len(reactants)}"

            # inverse labelling in their xyz files
            r0_label = reactants[0].split("/")[-1][:2]
            assert r0_label == 'r1', 'not r0'
            r1_label = reactants[1].split("/")[-1][:2]
            assert r1_label == 'r0', 'not r1'

            r_0_atomtypes, r_0_coords = reader(reactants[0])
            r_1_atomtypes, r_1_coords = reader(reactants[1])

            reactant_0_atomtypes_list.append(r_0_atomtypes)
            reactant_0_coords_list.append(r_0_coords)
            reactant_1_atomtypes_list.append(r_1_atomtypes)
            reactant_1_coords_list.append(r_1_coords)

            p_file = glob(rxn_dir + "p*.xyz")[0]
            p_atomtypes, p_coords = reader(p_file)
            product_atomtypes_list.append(p_atomtypes)
            product_coords_list.append(p_coords)

        assert len(product_coords_list) == len(reactant_0_coords_list), 'not as many reactants 0 as products'
        assert len(reactant_1_coords_list) == len(reactant_0_coords_list), 'not as many reactants 0 as reactants 1'

        assert len(product_atomtypes_list) == len(product_coords_list), 'not as many atomtypes as coords'
        assert len(reactant_0_atomtypes_list) == len(reactant_0_coords_list), 'not as many atomtypes as coords'
        assert len(reactant_1_atomtypes_list) == len(reactant_1_coords_list), 'not as many atomtypes as coords'

        self.reactant_0_coords = reactant_0_coords_list
        self.reactant_1_coords = reactant_1_coords_list
        self.reactant_0_atomtypes = reactant_0_atomtypes_list
        self.reactant_1_atomtypes = reactant_1_atomtypes_list
        self.product_coords = product_coords_list
        self.product_atomtypes = product_atomtypes_list

        savename = self.processed_dir + 'atomtypes_coords.pt'

        torch.save({'reactant_0_coords': reactant_0_coords_list,
                    'reactant_1_coords': reactant_1_coords_list,
                    'product_coords': product_coords_list,
                    'reactant_0_atomtypes': reactant_0_atomtypes_list,
                    'reactant_1_atomtypes': reactant_1_atomtypes_list,
                    'product_atomtypes': product_atomtypes_list,
                    'indices': self.indices},
                   savename)
        print(f"Atomtypes coords saved to {savename}")

        # now graphs
        print(f"Processing csv file and saving graphs to {self.processed_dir}")
        reactant_0_graphs_list = []
        reactant_1_graphs_list = []
        product_graphs_list = []
        for i, idx in enumerate(tqdm(self.indices, desc="making graphs")):
            # IMPORTANT : in db they are inconsistent about what is r0 and what is r1.
            # current soln is to check both. not ideal.
            rxnsmi = self.df[self.df['rxn_id'] == idx]['rxn_smiles'].item()
            rsmis, psmi = rxnsmi.split('>>')
            rsmi_0, rsmi_1 = rsmis.split('.')

            # REACTANT 0
            rmol_0 = Chem.MolFromSmiles(rsmi_0)
            assert rmol_0 is not None, f"rmol obj {idx} is None from smi {rsmi_0}"
            # add Hs
            nats = count_ats(rmol_0)
            rmol_0 = Chem.AddHs(rmol_0)
            nats = count_ats(rmol_0)
            rcoords_0 = reactant_0_coords_list[i]
            if nats != len(rcoords_0):
                # NATS dont match probably because r0 is r1
                rcoords_0 = reactant_1_coords_list[i]
            assert nats == len(rcoords_0), "nats don't match for either 0 or 1"
            r_graph_0 = get_graph(rmol_0, rcoords_0, self.labels[i],
                                radius=self.radius, max_neighbor=self.max_neighbor)
            reactant_0_graphs_list.append(r_graph_0)

            # REACTANT 1
            rmol_1 = Chem.MolFromSmiles(rsmi_1)
            assert rmol_1 is not None, f"rmol obj {idx} is None from smi {rsmi_1}"
            # add Hs
            rmol_1 = Chem.AddHs(rmol_1)
            nats = count_ats(rmol_1)
            rcoords_1 = reactant_1_coords_list[i]
            if nats != len(rcoords_1):
                # NATS dont match probably because r0 is r1
                rcoords_1 = reactant_0_coords_list[i]
            assert nats == len(rcoords_1), "nats don't match for either 0 or 1"
            r_graph_1 = get_graph(rmol_1, rcoords_1, self.labels[i],
                                  radius=self.radius, max_neighbor=self.max_neighbor)
            reactant_1_graphs_list.append(r_graph_1)

            # PRODUCT
            pmol = Chem.MolFromSmiles(psmi)
            assert pmol is not None, f"pmol obj {idx} is None from smi {psmi}"
            # add Hs
            pmol = Chem.AddHs(pmol)
            nats = count_ats(pmol)
            pcoords = product_coords_list[i]
            assert nats == len(pcoords), f"nats don't match in idx {idx}"
            p_graph = get_graph(pmol, pcoords, self.labels[i],
                                  radius=self.radius, max_neighbor=self.max_neighbor)
            product_graphs_list.append(p_graph)

        assert len(reactant_0_graphs_list) == len(reactant_1_graphs_list), 'number of reactants dont match'
        assert len(reactant_1_graphs_list) == len(product_graphs_list), 'number of reactants and products dont match'

        self.reactant_0_graphs = reactant_0_graphs_list
        self.reactant_1_graphs = reactant_1_graphs_list
        self.product_graphs = product_graphs_list

        r0graphsavename = self.processed_dir + 'reactant_0_graphs.pt'
        r1graphsavename = self.processed_dir + 'reactant_1_graphs.pt'

        pgraphsavename = self.processed_dir + 'product_graphs.pt'

        torch.save(reactant_0_graphs_list, r0graphsavename)
        torch.save(reactant_1_graphs_list, r1graphsavename)

        torch.save(product_graphs_list, pgraphsavename)
        print(f"Saved graphs to {r0graphsavename}, {r1graphsavename} and {pgraphsavename}")

