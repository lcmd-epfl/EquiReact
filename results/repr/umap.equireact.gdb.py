import os
from PIL import Image
from bokeh.embed import file_html
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, LinearColorMapper
from bokeh.palettes import Spectral10
from bokeh.plotting import figure, show, output_notebook
from bokeh.resources import CDN
from io import BytesIO
from sklearn.preprocessing import StandardScaler
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from rdkit import Chem
from rdkit.Chem import Draw
import re


repr_path  = 'gdb.noH.true.123.dat'
csv_path   = '../../data/gdb7-22-ts/ccsdtf12_dz_cleaned.csv'
error_path = '../by_mol/cv10-LP-gdb-ns64-nv64-d48-layers3-vector-diff-node-noH-truemapping.123.dat'
repr_len = 128
how_to_color = 'rxnclass' # errors / targets / rxnclass
img_path = './gdb.img.npy'


def main():
    df, data = load_data()
    if False:
        for n in (10, 20, 50, 100, 200):
            for d in (0.1, 0.25, 0.5, 0.8, 0.99):
                print(f'{n=} {d=}')
                write_plot(n, d, data, df)
    else:
        n = 10
        d = 0.1
        write_plot(n, d, data, df)


def load_data():
    data = np.loadtxt(repr_path, usecols=np.arange(2,2+repr_len))
    data = StandardScaler().fit_transform(data)

    indices = np.loadtxt(repr_path, usecols=1, dtype=int)
    classes = np.loadtxt(repr_path, usecols=0, dtype=str)
    train_idx = indices[np.where(classes=='train')]
    val_idx   = indices[np.where(classes=='val')]
    test_idx  = indices[np.where(classes=='test')]

    prediction_idx = np.loadtxt(error_path, usecols=0, dtype=int)
    assert np.all(test_idx==prediction_idx)
    prediction_errors = np.loadtxt(error_path, usecols=[1,2]).T
    prediction_errors = prediction_errors[1]-prediction_errors[0]

    df = pd.read_csv(csv_path)
    smiles = df['rxn_smiles'].values[indices]
    targets = df['dE0'].values[indices]
    errors = np.zeros_like(targets)
    errors[test_idx] = prediction_errors

    rxnclass  = df['rmg_family']
    rxnclasses_dict = {x:i for i,x in enumerate(set(rxnclass))}
    rxnclass_num = rxnclass.replace(rxnclasses_dict)

    if os.path.isfile(img_path):
        images = np.load(img_path)
    else:
        images = np.array([*map(embeddable_image, smiles)])
        np.save(img_path, images)

    labels = [f'#{i} ({c}) target:{t} error:{e} class:{rc}' for i,c,t,e,rc in zip(indices, classes, targets, errors, rxnclass)]
    if how_to_color=='errors':
        colors = errors
    elif how_to_color=='targets':
        colors = targets
    elif how_to_color=='rxnclass':
        colors = rxnclass_num

    radii = np.zeros_like(targets, dtype=int)
    radii[:] = 4
    if how_to_color=='errors':
        radii[test_idx] = 16.0

    df = pd.DataFrame({'image': images, 'label':labels, 'radii':radii, 'color':colors})

    return df, data


def write_plot(n, d, data, df):

    emb_path = f'{repr_path}.{d=}.{n=}.npy'
    out_path = f'{repr_path}.{how_to_color}.{d=}.{n=}.html'

    if os.path.isfile(emb_path):
        embedding = np.load(emb_path)
    else:
        reducer = umap.UMAP(n_neighbors=n, min_dist=d)
        embedding = reducer.fit_transform(data)
        np.save(emb_path, embedding)

    df['x'] = embedding[:,0]
    df['y'] = embedding[:,1]
    datasource = ColumnDataSource(df)
    color_mapping = LinearColorMapper(palette=Spectral10)

    plot_figure = figure(
           title='UMAP projection of the GDB dataset',
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

    html = file_html(plot_figure, CDN, "{d=} {n=}")

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
