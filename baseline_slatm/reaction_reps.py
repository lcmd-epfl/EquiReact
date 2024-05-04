import os
from glob import glob
from tqdm import tqdm
from itertools import chain
import numpy as np
import pandas as pd
import qml


def check_alt_files(list_files):
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


def read_xyz(xyz, bohr=False):
    mol0 = qml.Compound(xyz)
    mol = qml.Compound()
    mol.atomtypes       = np.copy(mol0.atomtypes)
    mol.nuclear_charges = np.copy(mol0.nuclear_charges)
    mol.coordinates     = mol0.coordinates * 0.529177 if bohr else np.copy(mol0.coordinates)
    return mol


class QML:
    def __init__(self):

        def get_cyclo_reactants_xyz(idx, xtb):
            if xtb:
                return [f'../data/cyclo/xyz-xtb/Reactant_{idx}_{reactant_id}.xyz' for reactant_id in (0, 1)]
            else:
                reactants = glob(f'../data/cyclo/xyz/{idx}/r*.xyz')
                reactants = check_alt_files(reactants)
                assert len(reactants)==2
                return reactants

        def get_cyclo_products_xyz(idx, xtb):
            if xtb:
                return [f'../data/cyclo/xyz-xtb/Product_{idx}.xyz']
            else:
                products = glob(f'../data/cyclo/xyz/{idx}/p*.xyz')
                assert len(products)==1
                return products


        self.get_cyclo_data = self.get_data_template(csv_path="../data/cyclo/cyclo.csv",
                                                     bad_idx_path=('../data/cyclo/bad-xtb.dat', 'rxn_id'),
                                                     target_column='G_act',
                                                     bohr=lambda _: False,
                                                     get_indices=lambda df: df['rxn_id'].to_list(),
                                                     get_reactants_xyz = get_cyclo_reactants_xyz,
                                                     get_products_xyz = get_cyclo_products_xyz)


        self.get_proparg_data = self.get_data_template(csv_path="../data/proparg/data.csv",
                                                       target_column='Eafw',
                                                       bohr=lambda _: False,
                                                       get_indices=lambda df: [''.join(x) for x in zip(df['mol'].to_list(), df['enan'].to_list())],
                                                       get_reactants_xyz=lambda idx, xtb: [f'../data/proparg/{"xyz-xtb" if xtb else "xyz"}/{idx}.r.xyz'],
                                                       get_products_xyz =lambda idx, xtb: [f'../data/proparg/{"xyz-xtb" if xtb else "xyz"}/{idx}.p.xyz'])


        def get_gdb_reactants_xyz(idx, xtb):
            if xtb:
                return [f'../data/gdb7-22-ts/xyz-xtb/{idx}/Reactant_{idx}_0_opt.xyz']
            else:
                return [f'../data/gdb7-22-ts/xyz/{idx:06}/r{idx:06}.xyz']

        def get_gdb_products_xyz(idx, xtb):
            if xtb:
                return sorted(glob(f'../data/gdb7-22-ts/xyz-xtb/{idx}/Product_{idx}_*_opt.xyz'))
            else:
                return sorted(glob(f'../data/gdb7-22-ts/xyz/{idx:06}/p*.xyz'))

        self.get_GDB7_ccsd_data = self.get_data_template(csv_path="../data/gdb7-22-ts/ccsdtf12_dz_cleaned.csv",
                                                         bad_idx_path=('../data/gdb7-22-ts/bad-xtb.dat', 'idx'),
                                                         target_column='dE0',
                                                         bohr=lambda xtb: not xtb,
                                                         get_indices=lambda df: df['idx'].tolist(),
                                                         get_reactants_xyz = get_gdb_reactants_xyz,
                                                         get_products_xyz = get_gdb_products_xyz)

        return


    def get_data_template(self, csv_path, target_column, get_indices, get_reactants_xyz, get_products_xyz, bohr, bad_idx_path=None):
        def get_data(xtb=False, xtb_subset=False):
            df = pd.read_csv(csv_path, index_col=0)
            if (xtb or xtb_subset) and bad_idx_path:
                bad_idx = np.loadtxt(bad_idx_path[0], dtype=int)
                for idx in bad_idx:
                    df.drop(df[df[bad_idx_path[1]]==idx].index, axis=0, inplace=True)

            self.barriers = df[target_column].to_numpy()
            indices = get_indices(df)
            print(f'{len(indices)} dataset size')

            reactants_files = [get_reactants_xyz(idx, xtb) for idx in indices]
            products_files  = [get_products_xyz(idx, xtb)  for idx in indices]
            self.mols_reactants = [[read_xyz(x, bohr=bohr(xtb)) for x in y] for y in reactants_files]
            self.mols_products  = [[read_xyz(x, bohr=bohr(xtb)) for x in y] for y in products_files]
            self.ncharges = [mol.nuclear_charges for mol in chain.from_iterable(self.mols_reactants+self.mols_products)]
            return
        return get_data


    def get_SLATM(self, no_h_atoms=False, no_h_total=False):
        def get_slatm_for_nestd_list(reaction_components, desc):
            xs_sum = []
            for mols in tqdm(reaction_components, desc=desc):
                xs = []
                for mol in mols:
                    h_mask = (mol.nuclear_charges==1)
                    if no_h_atoms:
                        x = np.array(qml.representations.generate_slatm(mol.coordinates, mol.nuclear_charges, mbtypes, local=True))
                        x = x[~h_mask].sum(axis=0)
                    elif no_h_total:
                        x = qml.representations.generate_slatm(mol.coordinates[~h_mask], mol.nuclear_charges[~h_mask], mbtypes, local=False)
                    else:
                        x = qml.representations.generate_slatm(mol.coordinates, mol.nuclear_charges, mbtypes, local=False)
                    xs.append(x)

                xs_sum.append(sum(np.array(xs)))
            return np.array(xs_sum)

        mbtypes = qml.representations.get_slatm_mbtypes(self.ncharges)
        slatm_reactants = get_slatm_for_nestd_list(self.mols_reactants, 'reactants')
        slatm_products  = get_slatm_for_nestd_list(self.mols_products, 'products')
        slatm_diff = slatm_products - slatm_reactants
        return slatm_diff
