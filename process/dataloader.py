import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
from process.create_graph import reader, get_graph, canon_mol, atom_featurizer
from glob import glob
from tqdm import tqdm
from rdkit import Chem
import os
import numpy as np

class Cyclo23TS(Dataset):
    def __init__(self, files_dir='data/cyclo/xyz/', csv_path='data/cyclo/mod_dataset.csv',
                 radius=20, max_neighbor=24, processed_dir='data/cyclo/processed/', process=True):

        self.max_number_of_reactants = 2
        self.max_number_of_products = 1

        self.max_neighbor = max_neighbor
        self.radius = radius
        self.files_dir = files_dir + '/'
        self.processed_dir = processed_dir + '/'

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

        if (not os.path.exists(os.path.join(self.processed_dir, 'reactant_0_graphs.pt')) or
                not os.path.exists(os.path.join(self.processed_dir, 'reactant_1_graphs.pt')) or
                not os.path.exists(os.path.join(self.processed_dir, 'product_graphs.pt')) or
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
            print(f"Coords and graphs successfully read from {self.processed_dir}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r_0_graph = self.reactant_0_graphs[idx]
        r_1_graph = self.reactant_1_graphs[idx]
        p_graph = self.product_graphs[idx]
        label = self.labels[idx]
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
            rmol_0 = canon_mol(rmol_0)
            assert rmol_0 is not None, f"rmol obj {idx} is None from smi {rsmi_0}"
            ats = []
            for at in rmol_0.GetAtoms():
                ats.append(at.GetSymbol())
            nats = len(ats)
            rcoords_0 = reactant_0_coords_list[i]
            ratoms_0 = reactant_0_atomtypes_list[i]

            if nats != len(rcoords_0) or np.all(ats == ratoms_0) == False:
                # NATS dont match probably because r0 is r1
                rcoords_0 = reactant_1_coords_list[i]
                ratoms_0 = reactant_1_atomtypes_list[i]

            assert nats == len(rcoords_0), "nats don't match for either 0 or 1"
            assert np.all(ats == ratoms_0), "atomtypes don't match for either 0 or 1"
            r_graph_0 = get_graph(rmol_0, ratoms_0, rcoords_0, self.labels[i],
                                  radius=self.radius, max_neighbor=self.max_neighbor)
            reactant_0_graphs_list.append(r_graph_0)

            # REACTANT 1
            rmol_1 = Chem.MolFromSmiles(rsmi_1)
            rmol_1 = canon_mol(rmol_1)
            assert rmol_1 is not None, f"rmol obj {idx} is None from smi {rsmi_1}"
            ats = []
            for at in rmol_1.GetAtoms():
                ats.append(at.GetSymbol())
            nats = len(ats)
            rcoords_1 = reactant_1_coords_list[i]
            ratoms_1 = reactant_1_atomtypes_list[i]
            # Need to check nats AND atomtype match
            if nats != len(rcoords_1) or np.all(ats == ratoms_1) == False:
                # NATS dont match probably because r0 is r1
                rcoords_1 = reactant_0_coords_list[i]
                ratoms_1 = reactant_0_atomtypes_list[i]

            assert nats == len(rcoords_1), "nats don't match for either 0 or 1"
            assert np.all(ats == ratoms_1), "atomtypes don't match for either 0 or 1"
            r_graph_1 = get_graph(rmol_1, ratoms_1, rcoords_1, self.labels[i],
                                  radius=self.radius, max_neighbor=self.max_neighbor)
            reactant_1_graphs_list.append(r_graph_1)

            # PRODUCT
            pmol = Chem.MolFromSmiles(psmi)
            pmol = canon_mol(pmol)
            assert pmol is not None, f"pmol obj {idx} is None from smi {psmi}"
            ats = []
            for at in pmol.GetAtoms():
                ats.append(at.GetSymbol())
            nats = len(ats)
            pcoords = product_coords_list[i]
            patoms = product_atomtypes_list[i]

            assert nats == len(pcoords), f"nats don't match in idx {idx}"
            assert np.all(ats == patoms), "atomtypes don't match"
            p_graph = get_graph(pmol, patoms, pcoords, self.labels[i],
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
        # TODO #######################################################
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
            atomtypes, coords = reader(r_file)
            reactant_atomtypes_list.append(atomtypes)
            reactant_coords_list.append(coords)
            # multiple products
            p_files = sorted(glob(rxn_dir +"p*.xyz"))
            products_atomtypes_list.append([])
            products_coords_list.append([])
            assert len(p_files) <= self.max_number_of_products, 'more products than the maximum number of products'
            for p_file in p_files:
                atomtypes, coords = reader(p_file)
                products_atomtypes_list[-1].append(atomtypes)
                products_coords_list[-1].append(coords)

        assert len(reactant_coords_list)    == len(products_coords_list), 'not as many products as reactants'
        assert len(products_atomtypes_list) == len(products_coords_list), 'not as many atomtypes as coords for products'
        assert len(reactant_atomtypes_list) == len(reactant_coords_list), 'not as many atomtypes as coords for reactants'


        # Graphs
        def make_graph(smi, atoms, coords):
            mol = Chem.MolFromSmiles(smi)

            try:
                mol = Chem.AddHs(mol)
                assert mol is not None, f"mol obj {idx} is None from smi {smi}"
                ats = [at.GetSymbol() for at in mol.GetAtoms()]
                assert len(ats) == len(atoms), f"nats don't match in idx {idx}"
                print('atoms from smi', ats)
                print('atoms from xyz', atoms)
                assert np.all(ats == atoms), "atomtypes don't match"
            except:
                print('standard smiles failed, canonicalising...')
                mol = canon_mol(mol)
                assert mol is not None, f"mol obj {idx} is None from smi {smi}"
                ats = [at.GetSymbol() for at in mol.GetAtoms()]
                assert len(ats) == len(atoms), f"nats don't match in idx {idx}"
                print('atoms from smi', ats)
                print('atoms from xyz', atoms)
                assert np.all(ats == atoms), "atomtypes don't match"
            return get_graph(mol, atoms, coords, self.labels[i],
                             radius=self.radius, max_neighbor=self.max_neighbor)

        print(f"Processing csv file and saving graphs to {self.processed_dir}")

        products_graphs_list = []
        reactant_graphs_list = []

        num_node_feat = atom_featurizer(Chem.MolFromSmiles('C')).shape[-1]
        empty = Data(x=torch.zeros((0, num_node_feat)), edge_index=torch.zeros((2,0)),
                     edge_attr=torch.zeros(0), y=torch.tensor(0.0), pos=torch.zeros((0,3)))

        for i, idx in enumerate(tqdm(self.indices, desc="making graphs")):
            print('idx', idx)
            rxnsmi = self.df[self.df['idx'] == idx]['rxn_smiles'].item()
            rsmi, psmis = rxnsmi.split('>>')

            # reactant
            reactant_graphs_list.append(make_graph(rsmi, reactant_atomtypes_list[i], reactant_coords_list[i]))

            # products
            psmis = psmis.split('.')
            nprod = len(products_coords_list[i])
            assert len(psmis) == nprod, 'number of products doesnt match'
            pgraphs = [make_graph(*args) for args in zip(psmis, products_atomtypes_list[i], products_coords_list[i])]
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
