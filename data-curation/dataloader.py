#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
from glob import glob
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
import os
import sys
import numpy as np
import argparse
sys.path.insert(0, '../')
from process.create_graph import reader, get_graph, canon_mol, atom_featurizer

import networkx as nx
import networkx.algorithms.isomorphism as iso



class GDB722TS(Dataset):

    def __init__(self, files_dir='../data/gdb7-22-ts/xyz/', csv_path='../data/gdb7-22-ts/ccsdtf12_dz.csv',
                 radius=20, max_neighbor=24):

        self.max_number_of_reactants = 1
        self.max_number_of_products = 3

        self.max_neighbor = max_neighbor
        self.radius = radius
        self.files_dir = files_dir + '/'

        print("Loading data into memory...")

        self.df = pd.read_csv(csv_path)
        self.indices = self.df['idx'].to_list()
        self.process()


    def process(self):

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
        print(f"Processing csv file")

        num_node_feat = atom_featurizer(Chem.MolFromSmiles('C')).shape[-1]
        empty = Data(x=torch.zeros((0, num_node_feat)), edge_index=torch.zeros((2,0)),
                     edge_attr=torch.zeros(0), y=torch.tensor(0.0), pos=torch.zeros((0,3)))

        products_graphs_list = []
        reactant_graphs_list = []
        for i, idx in enumerate(tqdm(self.indices, desc="making graphs")):
            rxnsmi = self.df[self.df['idx'] == idx]['rxn_smiles'].item()
            rsmi, psmis = rxnsmi.split('>>')
            # reactant
            reactant_graphs_list.append(self.make_graph(rsmi, reactant_atomtypes_list[i], reactant_coords_list[i], f'r{idx:06d}'))
            # products
            psmis = psmis.split('.')
            nprod = len(products_coords_list[i])
            assert len(psmis) == nprod, 'number of products doesnt match'
            pgraphs = [self.make_graph(*args, f'p{idx:06d}_{j}') for j, args in enumerate(zip(psmis, products_atomtypes_list[i], products_coords_list[i]))]
            padding = [empty] * (self.max_number_of_products-nprod)
            products_graphs_list.append(pgraphs + padding)

        assert len(reactant_graphs_list) == len(products_graphs_list), 'not as many products as reactants'

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
        G = nx.Graph()
        G.add_nodes_from([(i, {'q': q}) for i, q in enumerate(atoms)])
        G.add_edges_from(bonds)
        return G

    def make_graph(self, smi, atoms, coords, idx):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"mol obj {idx} is None from smi {smi}"
        mol = Chem.AddHs(mol)
        assert len(atoms)==mol.GetNumAtoms(), f"nats don't match in idx {idx}"

        rdkit_bonds = np.array(sorted(sorted((i.GetBeginAtomIdx(), i.GetEndAtomIdx())) for i in mol.GetBonds()))
        rdkit_atoms = np.array([at.GetSymbol() for at in mol.GetAtoms()])

        xyz_bonds = self.get_xyz_bonds(len(rdkit_bonds), atoms, coords)
        if xyz_bonds is None:
            #TODO
            #Draw.MolToImage(mol).save(f'{idx}.png')
            print('bond number', idx)
            return None
        assert len(rdkit_bonds)==len(xyz_bonds), "different number of bonds"

        if np.all(rdkit_atoms==atoms) and np.all(rdkit_bonds==xyz_bonds):
            # Don't search for a match because the first one doesn't have to be the shortest one
            new_atoms = atoms
            new_coords = coords
        else:
            G1 = self.make_nx_graph(rdkit_atoms, rdkit_bonds)
            G2 = self.make_nx_graph(atoms, xyz_bonds)
            GM = iso.GraphMatcher(G1, G2, node_match=iso.categorical_node_match('q', None))
            if not GM.is_isomorphic():
                #TODO
                #Draw.MolToImage(mol).save(f'{idx}.png')
                print('isomorphism', idx)
                return None
            match = next(GM.match())
            src, dst = np.array(sorted(match.items(), key=lambda match: match[0])).T
            assert np.all(src==np.arange(G1.number_of_nodes()))

            new_atoms = atoms[dst]
            new_coords = coords[dst]
            new_xyz_bonds = self.get_xyz_bonds(len(rdkit_bonds), new_atoms, new_coords)
            assert np.all(rdkit_atoms==new_atoms), "different atoms"
            assert np.all(rdkit_bonds==new_xyz_bonds), "different bonds"

        return get_graph(mol, new_atoms, new_coords, idx, radius=self.radius, max_neighbor=self.max_neighbor)



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--radius'           ,  type=float ,  default=5.0      ,  help='max radius of graph')
    args = p.parse_args()
    data = GDB722TS(radius=args.radius)
