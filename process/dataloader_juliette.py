import os
from os.path import exists, join
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from process.create_graph import get_graph, reader


class Juliette(Dataset):

    def __init__(self, process=True,
                 processed_dir='data/juliette/processed/',
                 noH=True, atom_mapping=False,
                 geometry='sub', reaction='cmd', geometry_p='sub',
                 target_column=None
                 ):

        self.version = 0  # INCREASE IF CHANGE THE DATA / DATALOADER / GRAPHS / ETC
        self.max_number_of_reactants = 1
        self.max_number_of_products = 1
        self.processed_dir = processed_dir + '/'
        self.atom_mapping = atom_mapping
        self.noH = noH
        if reaction.lower() != 'bde':
            if not target_column:
                target_column = 'fw_td_kcalmol'#BDE_kcalmol
            else:
                target_column = target_column
        elif reaction.lower() == 'bde':
            target_column = 'BDE_kcalmol'
        print(f'Target column set to: {target_column}')
        geometries = ['sub', 'int', 'int_keep_Pd_sub', 'int_changePdtoHe', 'int_changePdtoKr', 'sub_opt', "sub_Xebond00", "sub_Xebond10", "sub_Xebond20", "sub_Xebond30", "sub_Xe00", "sub_Xe10", "sub_Xe20", "sub_H00", "sub_He00", "sub_C00", "sub_Kr00"]
        if geometry not in geometries:
            raise NotImplementedError
        self.geometry = geometry
        self.reaction = reaction
        self.geometry_p = geometry_p

        if self.geometry == 'sub' and not noH:
            raise NotImplementedError

        if reaction.lower() == 'cmd':
            csv_path='data/juliette/CMD_TS_smiles_ok.csv'
            reaction = 'CMD'
        elif reaction.lower() == 'irb':
            csv_path='data/juliette/IrB_TS_smiles_kcalmolto20.csv'
            reaction = 'IrB'
        elif reaction.lower() == 'bde':
            csv_path = 'data/juliette/DFT-BDEs_sp2_sample2000_cleaned.csv'
            if self.geometry == "sub":
                self.files_dir_r = f'data/juliette/DFT-BDE/substrate/'
                self.files_dir_p = f'data/juliette/DFT-BDE/substrate_minusH/'
            elif self.geometry == "sub_Xebond00":
                self.files_dir_r = f'data/juliette/DFT-BDE/structures_tagged_atomXe_bond_dist0.0/'
                self.files_dir_p = f'data/juliette/DFT-BDE/substrate_minusH/'  
            elif self.geometry == "sub_Xebond10":
                self.files_dir_r = f'data/juliette/DFT-BDE/structures_tagged_atomXe_bond_dist1.0/'
                self.files_dir_p = f'data/juliette/DFT-BDE/substrate_minusH/'
            elif self.geometry == "sub_Xebond20":
                self.files_dir_r = f'data/juliette/DFT-BDE/structures_tagged_atomXe_bond_dist2.0/'
                self.files_dir_p = f'data/juliette/DFT-BDE/substrate_minusH/'
            elif self.geometry == "sub_Xebond30":
                self.files_dir_r = f'data/juliette/DFT-BDE/structures_tagged_atomXe_bond_dist3.0/'
                self.files_dir_p = f'data/juliette/DFT-BDE/substrate_minusH/'
            elif self.geometry == "sub_Xe00":
                self.files_dir_r = f'data/juliette/DFT-BDE/structures_tagged_atomXe_dist0.0/'
                self.files_dir_p = f'data/juliette/DFT-BDE/substrate_minusH/'
            elif self.geometry == "sub_Xe10":
                self.files_dir_r = f'data/juliette/DFT-BDE/structures_tagged_atomXe_dist1.0/'
                self.files_dir_p = f'data/juliette/DFT-BDE/substrate_minusH/'
            elif self.geometry == "sub_Xe20":
                self.files_dir_r = f'data/juliette/DFT-BDE/structures_tagged_atomXe_dist2.0/'
                self.files_dir_p = f'data/juliette/DFT-BDE/substrate_minusH/'     
            elif self.geometry == "sub_H00":
                self.files_dir_r = f'data/juliette/DFT-BDE/structures_tagged_atomH_dist0.0/'
                self.files_dir_p = f'data/juliette/DFT-BDE/substrate_minusH/'
            elif self.geometry == "sub_He00":
                self.files_dir_r = f'data/juliette/DFT-BDE/structures_tagged_atomHe_dist0.0/'
                self.files_dir_p = f'data/juliette/DFT-BDE/substrate_minusH/'
            elif self.geometry == "sub_C00":
                self.files_dir_r = f'data/juliette/DFT-BDE/structures_tagged_atomC_dist0.0/'
                self.files_dir_p = f'data/juliette/DFT-BDE/substrate_minusH/'
            elif self.geometry == "sub_Kr00":
                self.files_dir_r = f'data/juliette/DFT-BDE/structures_tagged_atomKr_dist0.0/'
                self.files_dir_p = f'data/juliette/DFT-BDE/substrate_minusH/'
        

        if reaction.lower() != 'bde':
            if self.geometry == 'int':
                self.files_dir_r = f'data/juliette/TS{reaction}/Int1/'
                self.files_dir_p = f'data/juliette/TS{reaction}/Int2/'
            elif self.geometry == 'sub':
                self.files_dir_r = f'data/juliette/TS{reaction}/substrate/'
                self.files_dir_p = f'data/juliette/TS{reaction}/substrate_minusH/'
            elif self.geometry == 'int_keep_Pd_sub':
                self.files_dir_r = f'data/juliette/TS{reaction}/Int1_keep_Pd_substrate/'
                self.files_dir_p = f'data/juliette/TS{reaction}/Int2_keep_Pd_substrate/'
            elif self.geometry == 'int_changePdtoHe':
                self.files_dir_r = f'data/juliette/TS{reaction}/Int1_changePdtoHe/'
                self.files_dir_p = f'data/juliette/TS{reaction}/Int2_changePdtoHe/'
            elif self.geometry == 'int_changePdtoKr':
                self.files_dir_r = f'data/juliette/TS{reaction}/Int1_changePdtoKr/'
                self.files_dir_p = f'data/juliette/TS{reaction}/Int2_changePdtoKr/'
            elif self.geometry == 'sub_opt':
                self.files_dir_r = f'data/juliette/TS{reaction}/substrate_opt/'
                self.files_dir_p = f'data/juliette/TS{reaction}/substrate_opt_minusH/'

        dataset_prefix = os.path.splitext(os.path.basename(csv_path))[0]
        dataset_prefix += f'.{geometry}'

        if noH:
            dataset_prefix += '.noH'
        self.paths = SimpleNamespace(
                rg = join(self.processed_dir, f'{dataset_prefix}.v{self.version}.reactants_graphs.pt'),
                pg = join(self.processed_dir, f'{dataset_prefix}.v{self.version}.products_graphs.pt'),
                mp = join(self.processed_dir, f'{dataset_prefix}.v{self.version}.p2r_mapping.pt'),
                )

        print("Loading data into memory...")
        print(f'{dataset_prefix=}')

        self.df = pd.read_csv(csv_path)
        self.nreactions = len(self.df)
        if reaction.lower() == 'bde' and self.geometry != 'sub':
            self.indices = self.df[['Int2_Name', 'Int2_Name']].to_numpy() 
        else:
            self.indices = self.df[['Int1_Name', 'Int2_Name']].to_numpy()

        self.labels = torch.tensor(self.df[target_column].values)
        #self.smiles = self.df[column]

        if process == True:
            print("Processing by request...")
            self.process()
        else:
            if exists(self.paths.rg) and exists(self.paths.pg) and exists(self.paths.mp):
                self.reactants_graphs = torch.load(self.paths.rg, weights_only=False)
                self.products_graphs = torch.load(self.paths.pg, weights_only=False)
                self.p2r_maps        = torch.load(self.paths.mp, weights_only=False)
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
            return label, idx, r, p, self.p2r_maps[idx]
        else:
            return label, idx, r, p


    def process(self):

        print(f"Processing xyz files and saving coords to {self.processed_dir}")
        if not exists(self.processed_dir):
            os.mkdir(self.processed_dir)
            print(f"Creating processed directory {self.processed_dir}")

        self.products_graphs = []
        self.reactants_graphs = []
        self.p2r_maps = []

        for i, idx in enumerate(tqdm(self.indices, desc="making graphs")):

            r_atomtypes, r_coords = reader(f'{self.files_dir_r}/{idx[0]}.xyz')
            p_atomtypes, p_coords = reader(f'{self.files_dir_p}/{idx[1]}.xyz')

            if self.geometry == 'int':
                assert len(r_atomtypes) == len(p_atomtypes), f'{idx}'
            else:
                assert len(r_atomtypes) == len(p_atomtypes)+1, f'{idx}'
            assert len(r_coords) == len(r_atomtypes), f'{idx}'
            assert len(p_coords) == len(p_atomtypes), f'{idx}'

            rgraph, ratoms, rmap = self.make_graph(r_atomtypes, r_coords, i)
            pgraph, patoms, pmap = self.make_graph(p_atomtypes, p_coords, i)

            self.reactants_graphs.append(rgraph)
            self.products_graphs.append(pgraph)

            assert np.all(ratoms == patoms)
            assert np.all(sorted(rmap)==np.arange(len(rmap))), f'atoms missing from mapping {idx}'
            assert np.all(sorted(rmap)==sorted(pmap)), f'atoms missing from mapping {idx}'
            p2rmap = np.hstack([np.where(pmap==j)[0] for j in rmap])
            assert np.all(rmap == pmap[p2rmap])
            assert np.all(ratoms == patoms[p2rmap])
            self.p2r_maps.append(p2rmap)

        torch.save(self.reactants_graphs, self.paths.rg)
        torch.save(self.products_graphs, self.paths.pg)
        torch.save(self.p2r_maps, self.paths.mp)
        print(f"Saved graphs to {self.paths.rg} and {self.paths.pg}")


    def make_graph(self, atoms, coords, ireact):

        if self.noH:
            noH_idx = np.where(atoms!='H')
            new_atoms = atoms[noH_idx]
            new_coords = coords[noH_idx]
        else:
            new_atoms = atoms
            new_coords = coords

        graph = get_graph(None, new_atoms, new_coords, ireact, features='torchchem_v1')
        atom_map = np.arange(graph.num_nodes)
        return graph, new_atoms, atom_map


    def standardize_labels(self):
        mean = torch.mean(self.labels)
        std = torch.std(self.labels)
        self.std = std
        self.labels = (self.labels - mean)/std
