#!/usr/bin/env python3

import os
from PIL import Image
from bokeh.embed import file_html
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, LinearColorMapper
from bokeh.palettes import Spectral10, Turbo256
from bokeh.plotting import figure
from bokeh.resources import CDN
from io import BytesIO
from sklearn.preprocessing import StandardScaler
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
import re
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--repr_path',  help='path to representation', default='gdb.noH.true.123.dat')
    parser.add_argument('--csv_path',   help='path to dataset csv', default='../../data/gdb7-22-ts/gdb.csv')
    parser.add_argument('--error_path', help='path to error per mol', default='../by_reaction/cv10-gdb-inv-random-noH-dft-true-ns64-nv64-d32-l2-vector-diff-both.123.dat')
    parser.add_argument('--class_path', help='file with bond-based classes', default='../gdb-reaction-classes/class_indices.dat')

    parser.add_argument('--cache_dir', help='path to cache', default='./data')

    parser.add_argument('--umap_n',      help='umap neighor number', type=float, default=10)
    parser.add_argument('--umap_d',      help='umap min distance',   type=float, default=0.1)
    parser.add_argument('--pcovr_mix',   help='pcovr pca:regression mixing', type=float, default=0.5)
    parser.add_argument('--pcovr_gamma', help='pcovr rbf kernel gamma (None for linear)', type=float, default=None)
    parser.add_argument('--tsne_perp',   help='tsne perplexity', type=float, default=30.0)
    parser.add_argument('--tsne_ex',     help='tsne early exaggeration', type=float, default=12.0)

    parser.add_argument('--loop', action='store_true', help='loop over parameters')
    parser.add_argument('--how_to_color', type=str, default='targets', help='how to color (targets/errors/rxnclass/bonds)')
    parser.add_argument('--method', type=str, default='umap', help='dimensionality reduction method (umap/pca/tsne/pcovr)')

    parser.add_argument('--dump', action='store_true', help='dump data for gnuplot instead of making html')

    args = parser.parse_args()


    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    df, data = load_data(args)
    if args.loop:
        if args.method=='umap':
            for n in (10, 20, 50, 100, 200):
                for d in (0.1, 0.25, 0.5, 0.8, 0.99):
                    print(f'{n=} {d=}')
                    write_plot(n, d, None, None, None, None, data, df, args)
        elif args.method=='pcovr':
            for mix in (0.25, 0.5, 0.75):
                for gamma in (None, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3):
                    print(f'{mix=} {gamma=}')
                    write_plot(None, None, mix, gamma, None, None, data, df, args)
        if args.method=='tsne':
            for perp in (10, 30, 64, 128, 256):
                for ex in (1, 6, 12, 24, 48):
                    print(f'{perp=} {ex=}')
                    write_plot(None, None, None, None, perp, ex, data, df, args)
    else:
        write_plot(args.umap_n, args.umap_d, args.pcovr_mix, args.pcovr_gamma, args.tsne_perp, args.tsne_ex, data, df, args)


def load_data(args):

    try:
        data = np.loadtxt(args.repr_path, converters={0:lambda s: 0.0, 1:lambda s: 0.0})[:,2:]  # remove first 2 columns

        indices = np.loadtxt(args.repr_path, usecols=1, dtype=int)
        classes = np.loadtxt(args.repr_path, usecols=0, dtype=str)
        train_idx = indices[np.where(classes=='train')]
        val_idx   = indices[np.where(classes=='val')]
        test_idx  = indices[np.where(classes=='test')]

        prediction_idx = np.loadtxt(args.error_path, usecols=0, dtype=int)
        assert np.all(test_idx==prediction_idx)
        prediction_errors = np.loadtxt(args.error_path, usecols=[1,2]).T
        prediction_errors = prediction_errors[1]-prediction_errors[0]
    except:
        data = np.load(args.repr_path)
        indices = np.arange(len(data))
        prediction_errors = np.zeros_like(indices)
        classes = np.zeros_like(indices)
        test_idx = np.arange(len(data))


    data = StandardScaler().fit_transform(data)

    df = pd.read_csv(args.csv_path)
    smiles = df['rxn_smiles'].values[indices]
    targets = df['dE0'].values[indices]
    errors = np.zeros_like(targets)
    errors[test_idx] = prediction_errors

    rxnclass  = df['rmg_family']
    rxnclass_num = rxnclass.replace(np.nan, '')
    rxnclasses_dict = {x:i for i,x in enumerate(sorted(set(rxnclass_num)))}
    rxnclass_num = rxnclass_num.replace(rxnclasses_dict)
    rxnclass = rxnclass.values[indices]
    rxnclass_num = rxnclass_num.values[indices]

    img_path = f'{args.cache_dir}/gdb.img.npy'
    if os.path.isfile(img_path):
        images = np.load(img_path)
    else:
        images = np.array([*map(embeddable_image, smiles)])
        np.save(img_path, images)

    labels = [f'#{i} ({c}) target:{t} error:{e} class:{rc}' for i,c,t,e,rc in zip(indices, classes, targets, errors, rxnclass)]
    if args.how_to_color=='errors':
        colors = errors
    elif args.how_to_color=='targets':
        colors = targets
    elif args.how_to_color=='rxnclass':
        colors = rxnclass_num
    elif args.how_to_color=='bonds':
        bonds = np.loadtxt(args.class_path, dtype=int)[indices]
        unique_bonds, count_bonds = np.unique(bonds, return_counts=True)
        bonds_dict = {x:i for i, x in enumerate(unique_bonds[np.argsort(count_bonds)][::-1])}
        bonds_num = np.squeeze(pd.DataFrame(bonds).replace(bonds_dict).values)
        colors = bonds_num

    radii = np.zeros_like(targets, dtype=int)
    radii[:] = 4
    if args.how_to_color=='errors':
        radii[test_idx] = 16.0
    elif args.how_to_color=='bonds':
        radii[np.where(bonds<0)] = 1.0

    df = pd.DataFrame({'image': images, 'label':labels, 'radii':radii, 'color':colors, 'targets':targets})

    return df, data


def write_plot(n, d, mix, gamma, perp, ex, data, df, args):

    if args.method=='umap':
        emb_path = f'{args.cache_dir}/{os.path.basename(args.repr_path)}.{args.method}.{d=}.{n=}.npy'
        out_path = f'{os.path.basename(args.repr_path)}.{args.method}.{args.how_to_color}.{d=}.{n=}.html'
        dump_path= f'{os.path.basename(args.repr_path)}.{args.method}.{args.how_to_color}.{d=}.{n=}.dat'
    elif args.method=='pcovr':
        emb_path = f'{args.cache_dir}/{os.path.basename(args.repr_path)}.{args.method}.{mix=}.{gamma=}.npy'
        out_path = f'{os.path.basename(args.repr_path)}.{args.method}.{args.how_to_color}.{mix=}.{gamma=}.html'
        dump_path= f'{os.path.basename(args.repr_path)}.{args.method}.{args.how_to_color}.{mix=}.{gamma=}.html'
    elif args.method=='tsne':
        emb_path = f'{args.cache_dir}/{os.path.basename(args.repr_path)}.{args.method}.{perp=}.{ex=}.npy'
        out_path = f'{os.path.basename(args.repr_path)}.{args.method}.{args.how_to_color}.{perp=}.{ex=}.html'
        dump_path= f'{os.path.basename(args.repr_path)}.{args.method}.{args.how_to_color}.{perp=}.{ex=}.dat'
    else:
        emb_path = f'{args.cache_dir}/{os.path.basename(args.repr_path)}.{args.method}.npy'
        out_path = f'{os.path.basename(args.repr_path)}.{args.method}.{args.how_to_color}.html'
        dump_path= f'{os.path.basename(args.repr_path)}.{args.method}.{args.how_to_color}.dat'

    if os.path.isfile(emb_path):
        print(f'loading precomputed embedding from {emb_path}')
        embedding = np.load(emb_path)
    else:
        if args.method=='umap':
            import umap
            reducer = umap.UMAP(n_neighbors=n, min_dist=d)
            embedding = reducer.fit_transform(data)
        elif args.method=='pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embedding = pca.fit_transform(data)
        elif args.method=='tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, perplexity=perp, early_exaggeration=ex)
            embedding = tsne.fit_transform(data)
        elif args.method=='pcovr':
            if args.pcovr_gamma is None:
                from sklearn.linear_model import Ridge
                from skmatter.decomposition import PCovR
                pcovr = PCovR(mixing=mix, regressor=Ridge(alpha=1e-8, fit_intercept=False, tol=1e-12), n_components=2)
            else:
                from sklearn.kernel_ridge import KernelRidge
                from skmatter.decomposition import KernelPCovR
                pcovr = KernelPCovR(mixing=mix, regressor=KernelRidge(alpha=1e-8, kernel="rbf", gamma=gamma),
                                    kernel="rbf", gamma=gamma, n_components=2)
            pcovr.fit(data, StandardScaler().fit_transform(np.array(df['targets']).reshape(-1, 1)))
            embedding = pcovr.transform(data)

        np.save(emb_path, embedding)

    if args.dump:
        np.savetxt(dump_path, np.hstack((embedding, np.array(df.color)[:,None])))

    else:
        df['x'] = embedding[:,0]
        df['y'] = embedding[:,1]
        datasource = ColumnDataSource(df)
        color_mapping = LinearColorMapper(palette=Turbo256)

        plot_figure = figure(
               title=f'{args.method} projection of the GDB dataset',
               width=1000,
               height=1000,
               tools=('pan, wheel_zoom, reset')
           )

        plot_figure.add_tools(HoverTool(tooltips="""
        <div>
            <div>
                <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
            </div>
            <div>
                <span style='font-size: 18px'>@label</span>
            </div>
        </div>
        """))

        plot_figure.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='color', transform=color_mapping),
            line_alpha=0.6,
            fill_alpha=0.6,
            size='radii',
            #size=4
        )

        html = file_html(plot_figure, CDN, f"{args.method} {os.path.basename(args.repr_path)}")

        with open(out_path, "w") as f:
            f.write(html)


def embeddable_image(smi):
    smi = re.sub(':[0-9]+', '', smi).replace('>>', '.')
    mol = Chem.MolFromSmiles(smi)
    img_data = Draw.MolToImage(mol, returnPNG=True, size=(128, 128))
    buffer = BytesIO()
    img_data.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()


if __name__ == '__main__':
    main()
