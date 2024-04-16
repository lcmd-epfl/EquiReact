#!/usr/bin/env python3

import glob
from collections import defaultdict
import numpy as np


def round_with_std(mae, std):
    if std > 1:
        std_round = str(round(std, 1))
    else:
        # works only is std < 1
        std_1digit = int(std // 10**np.floor(np.log10(std)))
        std_1digit_pos = f'{std:.10f}'.split('.')[1].index(str(std_1digit))
        if std_1digit > 2:
            std_round = f'{std:.{std_1digit_pos+1}f}'
        else:
            std_round = f'{std:.{std_1digit_pos+2}f}'
    n_after_point = len(std_round.split('.')[1])
    mae_round = f'{mae:.{n_after_point}f}'
    return f'$ {mae_round} \\pm {std_round} $'


def get_error(val, use_rmse, latex=True):
    if val is None:
        if latex:
            return '---'
        else:
            return None, None
    if use_rmse:
        val = val[2:4]
    else:
        val = val[0:2]
    if latex:
        return round_with_std(*val)
    else:
        return val


def load_slatm():
    d = defaultdict(lambda: None)
    for f in glob.glob('baseline_slatm/slatm_10_fold_*.npy'):
        key = f.replace('baseline_slatm/slatm_10_fold_', '').replace('.npy', '').replace('_split', '')
        k = key.split('_')
        if len(k)==2:
            key = '-'.join([k[0], 'dft', k[1]])
        else:
            key = key.replace('_', '-')
        key = key + '-none'
        val = np.load(f)
        val = np.hstack((val.mean(axis=1), val.std(axis=1)))[[0,2,1,3]]
        d[key] = val
    return d


def load_chemprop():
    chemprop = arr2dict(np.loadtxt('baseline_chemprop/res-short.txt', dtype=str, skiprows=1))
    d = defaultdict(lambda: None)
    for key, val in chemprop.items():
        k = key.split('-')
        if k[1]!='scaffold':
            key = '-'.join([k[0], 'random', *k[1:]])
        if k[-1]!='withH':
            key = key + '-noH'
        d[key] = val
    return d


def load_equireact():
    equireact = {}
    for dataset in ['gdb', 'cyclo', 'proparg']:
        equireact.update(arr2dict(np.loadtxt(f'results/results-{dataset}.txt', dtype=str)))
    equireact = {key[:key.find('-ns')].replace('normal', 'none'): val for key, val in equireact.items()}
    return defaultdict(lambda: None, equireact)


def arr2dict(arr):
    return {x[0]: [*map(float, x[1:])] for x in arr}


def print_main_table(geometry='dft', use_H=False, use_rmse=False):
    header=r'''\begin{tabular}{@{}ccccc@{}} \toprule
\makecell{Dataset \\ (property, units)} & \makecell{Atom-mapping\\ regime}
            & \CGR & SLATM$_d$+KRR & \textsc{EquiReact} \\ '''

    splitter_header = {
        'random': r'\multicolumn{5}{@{}c@{}}{\emph{Random splits}}\\ \midrule',
        'scaffold': r'\multicolumn{5}{@{}c@{}}{\emph{Scaffold splits}}\\ \midrule',
        }

    dataset_header = {
        'gdb': r'\multirow{3}{*}{\makecell{\gdb \\ ($\Delta E^\ddag$, kcal/mol)}}',
        'cyclo': r'\\[0.002cm]\multirow{3}{*}{\makecell{\cyclo \\ ($\Delta G^\ddag$, kcal/mol)}}',
        'proparg': r'\\[0.002cm] \multirow{2}{*}{\makecell{\proparg \\ ($\Delta E^\ddag$, kcal/mol)}}'
        }

    footer=r'''\bottomrule
\end{tabular}
'''
    h_key = "withH" if use_H else "noH"
    print(header)
    for splitter in ['random', 'scaffold']:
        print('\midrule')
        print(splitter_header[splitter])
        for dataset in ['gdb', 'cyclo', 'proparg']:
            print(dataset_header[dataset])
            for atom_mapping in ['True', 'RXNMapper', 'None']:
                if dataset=='proparg' and atom_mapping=='RXNMapper':
                    continue

                print(f'& {atom_mapping} & ', end='')

                chemprop_key  = f'{dataset}-{splitter}-{atom_mapping.lower()}-{h_key}'
                slatm_key     = f'{dataset}-{geometry}-{splitter}-{atom_mapping.lower()}'
                equireact_key = f'cv10-{dataset}-{splitter}-{h_key}-{geometry}-{atom_mapping.lower()}'

                print(get_error(chemprop[chemprop_key], use_rmse), end='')
                print(' & ', end='')
                print(get_error(slatm[slatm_key], use_rmse), end='')
                print(' & ', end='')
                print(get_error(equireact[equireact_key], use_rmse), end='')

                print(r' \\')
    print(footer)


def print_main_table_with_inv(geometry='dft', use_H=False, use_rmse=False):
    header=r'''\begin{tabular}{@{}cccccc@{}} \toprule
\makecell{Dataset \\ (property, units)} & \makecell{Atom-mapping\\ regime}
            & \CGR & SLATM$_d$+KRR & \textsc{EquiReact} & \textsc{InReact} \\ '''

    splitter_header = {
        'random': r'\multicolumn{6}{@{}c@{}}{\emph{Random splits}}\\ \midrule',
        'scaffold': r'\multicolumn{6}{@{}c@{}}{\emph{Scaffold splits}}\\ \midrule',
        }

    dataset_header = {
        'gdb': r'\multirow{3}{*}{\makecell{\gdb \\ ($\Delta E^\ddag$, kcal/mol)}}',
        'cyclo': r'\\[0.002cm]\multirow{3}{*}{\makecell{\cyclo \\ ($\Delta G^\ddag$, kcal/mol)}}',
        'proparg': r'\\[0.002cm] \multirow{2}{*}{\makecell{\proparg \\ ($\Delta E^\ddag$, kcal/mol)}}'
        }

    footer=r'''\bottomrule
\end{tabular}
'''
    h_key = "withH" if use_H else "noH"
    print(header)
    for splitter in ['random', 'scaffold']:
        print('\midrule')
        print(splitter_header[splitter])
        for dataset in ['gdb', 'cyclo', 'proparg']:
            print(dataset_header[dataset])
            for atom_mapping in ['True', 'RXNMapper', 'None']:
                if dataset=='proparg' and atom_mapping=='RXNMapper':
                    continue

                print(f'& {atom_mapping} & ', end='')

                chemprop_key  = f'{dataset}-{splitter}-{atom_mapping.lower()}-{h_key}'
                slatm_key     = f'{dataset}-{geometry}-{splitter}-{atom_mapping.lower()}'
                equireact_key = f'cv10-{dataset}-{splitter}-{h_key}-{geometry}-{atom_mapping.lower()}'
                inreact_key   = f'cv10-{dataset}-inv-{splitter}-{h_key}-{geometry}-{atom_mapping.lower()}'

                print(get_error(chemprop[chemprop_key], use_rmse), end='')
                print(' & ', end='')
                print(get_error(slatm[slatm_key], use_rmse), end='')
                print(' & ', end='')
                print(get_error(equireact[equireact_key], use_rmse), end='')
                print(' & ', end='')
                print(get_error(equireact[inreact_key], use_rmse), end='')

                print(r' \\')
    print(footer)


def print_hydrogen_table(geometry='dft', use_H=False, use_rmse=False, splitter='random'):
    pass
    header=r'''\begin{tabular}{@{}cccccccc@{}} \toprule
\multirow{3}{*}{\makecell{Dataset \\ (property, units)}}&
\multirow{3}{*}{H mode}&
\multicolumn{6}{c}{Atom mapping regime} \\ \cmidrule(lr){3-8}'''
    dataset_header = {
        'gdb': r'\midrule \multirow{2}{*}{\makecell{\gdb \\ ($\Delta E^\ddag$, kcal/mol)}}',
        'cyclo': r'\\[0.002cm] \multirow{2}{*}{\makecell{\cyclo \\ ($\Delta G^\ddag$, kcal/mol)}}',
        'proparg': r'\\[0.002cm] \multirow{2}{*}{\makecell{\proparg \\ ($\Delta E^\ddag$, kcal/mol)}}'
        }
    footer=r'''\bottomrule
\end{tabular}'''
    print(header)
    print('&', end='')
    for atom_mapping in ['True', 'RXNMapper', 'None']:
        print(r'& \multicolumn{2}{@{}c@{}}{'+atom_mapping+'} ', end='')
    print(r'\\ \cmidrule(lr){3-4} \cmidrule(lr){5-6}  \cmidrule(lr){7-8}')
    print('&', end='')
    for mode in ['M', 'M', 'S']:
        print(r'& \CGR & \textsc{EquiReact}$_'+mode+'$ ', end='')
    print(r'\\')
    for dataset in ['gdb', 'cyclo', 'proparg']:
        print(dataset_header[dataset])
        for h_label, h_key in zip(['with', 'w/o'], ['withH', 'noH']):
            print(f'& {h_label} ', end='')
            for atom_mapping in ['True', 'RXNMapper', 'None']:
                chemprop_key  = f'{dataset}-{splitter}-{atom_mapping.lower()}-{h_key}'
                equireact_key = f'cv10-{dataset}-{splitter}-{h_key}-{geometry}-{atom_mapping.lower()}'
                print(' & ', end='')
                print(get_error(chemprop[chemprop_key], use_rmse), end='')
                print(' & ', end='')
                print(get_error(equireact[equireact_key], use_rmse), end='')
            print(r'\\')
    print(footer)


def print_xtb_data(use_H=False, use_rmse=False, splitter='random'):
    h_key = "withH" if use_H else "noH"
    for dataset in ['gdb', 'cyclo', 'proparg']:
        print(f'#{dataset}')
        geometries = ['dft', 'xtb'] if dataset=='proparg' else ['sub', 'xtb']
        i=1.0
        for atom_mapping in ['True', 'RXNMapper', 'None']:
            if dataset=='proparg' and atom_mapping=='RXNMapper':
                continue
            for geom_label, geometry in zip(['DFT', 'xTB'], geometries):
                equireact_key = f'cv10-{dataset}-{splitter}-{h_key}-{geometry}-{atom_mapping.lower()}'
                print(i, '\t', geom_label, '\t', equireact_key, '\t', *get_error(equireact[equireact_key], use_rmse=use_rmse, latex=False))
                i+=1.0
            i+=0.5
        for geom_label, geometry in zip(['DFT', 'xTB'], geometries):
            slatm_key = f'{dataset}-{geometry}-{splitter}-{atom_mapping.lower()}'
            print(i, '\t', geom_label, '\t', 'slatm-'+slatm_key, '\t', *get_error(slatm[slatm_key], use_rmse=use_rmse, latex=False))
            i+=1.0
        print()
        print()


def print_attn_table(use_H=False, use_rmse=False, splitter='random', geometry='dft'):
    h_key = "withH" if use_H else "noH"
    header = r'''\begin{tabular}{@{}cccc@{}} \toprule
Dataset (property, units)
& Mapping mode & \textsc{EquiReact}$_X$ & \textsc{EquiReact}$_S$ \\ \midrule'''
    footer=r'''\bottomrule
\end{tabular}'''
    print(header)
    for dataset, prop in zip(['gdb', 'cyclo', 'proparg'], ['E', 'G', 'E']):
        print('\\'+dataset, r'($\Delta '+prop+'^\ddag$, kcal/mol) & None', end='')
        for atom_mapping in ['cross', 'none']:
            equireact_key = f'cv10-{dataset}-{splitter}-{h_key}-{geometry}-{atom_mapping}'
            print('&', get_error(equireact[equireact_key], use_rmse), end='')
        print(r' \\[0.002cm]')
    print(footer)


if __name__=='__main__':
    chemprop = load_chemprop()
    slatm = load_slatm()
    equireact = load_equireact()

    print('% THE MAIN TABLE OF THE MAIN TEXT: DFT, NO HYDROGENS, MAE')
    print_main_table(geometry='dft', use_H=False, use_rmse=False)
    print()

    print('% THE MAIN TABLE SI SUPPLEMENT: DFT, NO HYDROGENS, RMSE')
    print_main_table(geometry='dft', use_H=False, use_rmse=True)
    print()

    print('% HYDROGENS vs NO HYDROGENS SI TABLE: DFT, MAE, RANDOM SPLITS')
    print_hydrogen_table(geometry='dft', use_rmse=False)
    print()

    print('% EQUIREACT_X vs EQUIREACT_S SI TABLE: DFT, MAE, RANDOM, NO H')
    print_attn_table()
    print()

    print('% DFT vs XTB RESULTS FOR GNUPLOT: MAE, RANDOM, NO H')
    print_xtb_data()

    print('% THE MAIN TABLE OF THE MAIN TEXT BUT WITH INVARIANT: DFT, NO HYDROGENS, MAE')
    print_main_table_with_inv(geometry='dft', use_H=False, use_rmse=False)
    print()
