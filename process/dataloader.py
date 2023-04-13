from torch.utils.data import Dataset
import torch
import pandas as pd
from process.create_graph import reader, get_graph, count_ats
import os
import glob
from copy import deepcopy
from tqdm import tqdm
import rdkit
from rdkit import Chem


class GDB7RXN(Dataset):
    def __init__(self, files_dir='data/xyz/', csv_path='data/ccsdtf12_dz.csv',
                 radius=20, max_neighbor=24, device='cpu', processed_dir='data/processed/', process=True):
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

        data = pd.read_csv(csv_path)
        labels = torch.tensor(data['dHrxn298'].values)

        self.df = data

        mean = torch.mean(labels)
        std = torch.std(labels)
        #TODO to normalise in train/test/val split
        self.labels = (labels - mean)/std

        indices = data['idx'].to_list()
        self.indices = indices
        # pad to 6 figs with zeros
        padded_indices = []
        for idx in indices:
            idx = str(idx)
            len_idx = len(idx)
            pad = 6 - len_idx
            padded_idx = pad * '0' + idx
            assert len(padded_idx) == 6, 'filename padding incorrect'
            padded_indices.append(padded_idx)

        self.padded_indices = padded_indices

        if (not os.path.exists(os.path.join(self.processed_dir, 'reactant_graphs.pt')) and
                not os.path.exists(os.path.join(self.processed_dir, 'product_graphs.pt')) and
                not os.path.exists(os.path.join(self.processed_dir, 'atomtypes_coords.pt'))):
            print("processed data not found, processing data...")
            self.process()

        elif process == True:
            print("processing by request...")
            self.process()

        else:
            self.reactant_graphs = torch.load(os.path.join(self.processed_dir, 'reactant_graphs.pt'))
            self.product_graphs = torch.load(os.path.join(self.processed_dir, 'product_graphs.pt'))
            atomtypes_coords = torch.load(os.path.join(self.processed_dir, 'atomtypes_coords.pt'))
            self.reactant_coords = atomtypes_coords['reactant_coords']
            self.reactant_atomtypes = atomtypes_coords['reactant_atomtypes']
            self.product_coords = atomtypes_coords['product_coords']
            self.product_atomtypes = atomtypes_coords['product_atomtypes']
            print(f"Coords and graphs successfully read from {self.processed_dir}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # maybe several graphs in prods
        r_atomtypes = self.reactant_atomtypes[idx]
        p_atomtypes = self.product_atomtypes[idx]
        r_coords = self.reactant_coords[idx]
        p_coords = self.product_coords[idx]
        r_graph = self.reactant_graphs[idx]
        p_graph = self.product_graphs[idx]
        label = self.labels[idx]
        return r_graph, r_atomtypes, r_coords, p_graph, p_atomtypes, p_coords, label, idx

    def process(self):
        print(f"Processing xyz files and saving coords to {self.processed_dir}")
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            print(f"Creating processed directory {self.processed_dir}")

        reactant_coords_list = []
        reactant_atomtypes_list = []
        product_coords_list = []
        product_atomtypes_list = []
        for idx in tqdm(self.padded_indices, desc='reading xyz files'):
            # single reactant, but treat as several since there are several products
            r_file = self.files_dir + idx + "/r" + idx + ".xyz"
            r_atomtypes, r_coords = reader(r_file)
            reactant_coords_list.append([r_coords])
            reactant_atomtypes_list.append([r_atomtypes])

            # sometimes several products
            p_single_file = self.files_dir + idx + "/p" + idx + ".xyz"
            if os.path.exists(p_single_file):
                p_file = p_single_file
                p_atomtypes, p_coords = reader(p_file)
                product_coords_list.append([p_coords])
                product_atomtypes_list.append([p_atomtypes])
            else:
                p_files = sorted(glob.glob(self.files_dir + idx + "/p" + idx + "_*.xyz"))
                p_coords_list = []
                p_atomtypes_list = []
                for p_file in p_files:
                    p_atomtypes, p_coords = reader(p_file)
                    p_coords_list.append(p_coords)
                    p_atomtypes_list.append(p_atomtypes)
                product_coords_list.append(p_coords_list)
                product_atomtypes_list.append(p_atomtypes_list)

        assert len(product_coords_list) == len(reactant_coords_list), 'not as many reactants as products'
        assert len(product_atomtypes_list) == len(product_coords_list), 'not as many atomtypes as coords'
        assert len(reactant_atomtypes_list) == len(reactant_coords_list), 'not as many atomtypes as coords'

        self.reactant_coords = reactant_coords_list
        self.reactant_atomtypes = reactant_atomtypes_list
        self.product_coords = product_coords_list
        self.product_atomtypes = product_atomtypes_list

        savename = self.processed_dir + 'atomtypes_coords.pt'

        torch.save({'reactant_coords': reactant_coords_list,
                    'product_coords': product_coords_list,
                    'reactant_atomtypes': reactant_atomtypes_list,
                    'product_atomtypes': product_atomtypes_list,
                    'padded_indices': self.padded_indices},
                   savename)
        print(f"Atomtypes coords saved to {savename}")

        # now graphs
        print(f"Processing csv file and saving graphs to {self.processed_dir}")
        reactant_graphs_list = []
        product_graphs_list = []
        for i, idx in enumerate(tqdm(self.indices, desc="making graphs")):
            # get_graph expects mol obj, coords, y
            rsmi = self.df[self.df['idx'] == idx]['rsmi'].item()
            rmol = Chem.MolFromSmiles(rsmi)
            assert rmol is not None, f"rmol obj {idx} is None from smi {rsmi}"
            # add Hs
            rmol = Chem.AddHs(rmol)
            nats = count_ats(rmol)
            rcoords = reactant_coords_list[i][0]
            assert nats == len(rcoords), "nats don't match"
            r_graph = get_graph(rmol, rcoords, self.labels[i],
                                radius=self.radius, max_neighbor=self.max_neighbor)
            reactant_graphs_list.append([r_graph])

            # need to separate smiles by .
            psmi = self.df[self.df['idx'] == idx]['psmi'].item()
            if '.' in psmi:
                psmis = psmi.split(".")
            else:
                psmis = [psmi]
            p_graphs = []
            for p_idx, psmi in enumerate(psmis):
                pmol = Chem.MolFromSmiles(psmi)
                assert pmol is not None, f"pmol obj {idx} is None from smi {psmi}"
                pmol = Chem.AddHs(pmol)
                nats = count_ats(pmol)
                pcoords = product_coords_list[i][p_idx]
                assert nats == len(pcoords), "nats don't match"
                p_graph = get_graph(pmol, pcoords, self.labels[i],
                                    radius=self.radius, max_neighbor=self.max_neighbor)
                p_graphs.append(p_graph)
            product_graphs_list.append(p_graphs)

        assert len(reactant_graphs_list) == len(product_graphs_list), 'not as many reactant as product graphs'

        self.reactant_graphs = reactant_graphs_list
        self.product_graphs = product_graphs_list

        rgraphsavename = self.processed_dir + 'reactant_graphs.pt'
        pgraphsavename = self.processed_dir + 'product_graphs.pt'

        torch.save(reactant_graphs_list, rgraphsavename)
        torch.save(product_graphs_list, pgraphsavename)
        print(f"Saved graphs to {rgraphsavename} and {pgraphsavename}")

