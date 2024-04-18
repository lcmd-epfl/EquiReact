import argparse
from itertools import compress
import pprint
import wandb
from train import train
import os
def train_wrapper():
    with wandb.init(config=None):
        args = wandb.config
        try:
            train(run_dir, logname, project, wandb_name, args, seed0=123,
                  device='cuda', num_epochs=args.num_epochs, checkpoint=None,
                  subset=args.subset, dataset=args.dataset, process=False,
                  radius=args.radius, max_neighbors=args.max_neighbors, sum_mode=args.sum_mode,
                  n_s=args.n_s, n_v=args.n_v, n_conv_layers=args.n_conv_layers, distance_emb_dim=args.distance_emb_dim,
                  graph_mode=args.graph_mode, dropout_p=args.dropout_p, random_baseline=False,
                  combine_mode=args.combine_mode, atom_mapping=args.atom_mapping, CV=1, attention=args.attention,
                  noH=args.noH, two_layers_atom_diff=args.two_layers_atom_diff, rxnmapper=False, reverse=False,
                  xtb=args.xtb, split_complexes=False, sweep=True, lr=args.lr, weight_decay=args.weight_decay,
                  training_fractions=[args.train_frac])
        except Exception as e:
            print(e)
            pass


parser = argparse.ArgumentParser()
g = parser.add_mutually_exclusive_group(required=True)
g.add_argument('-c', '--cyclo', action='store_true', help='use Cyclo-23-TS dataset')
g.add_argument('-g', '--gdb', action='store_true', help='use GDB7-22-TS dataset')
g.add_argument('-p', '--proparg', action='store_true', help='use Proparg-21-TS dataset')
args = parser.parse_args()
dataset = next(compress(('cyclo', 'gdb', 'proparg'), (args.cyclo, args.gdb, args.proparg)))

epochs = {'cyclo': 256, 'gdb': 128, 'proparg': 128}
project = f'nequireact-{dataset}-80'
run_dir = f'sweep_{dataset}'
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
logname = 'sweep.log'

wandb.login()

metric = { 'name': 'val_score_best', 'goal': 'minimize' }
sweep_config = { 'method': 'bayes', 'metric': metric }

parameters_dict = {
    'combine_mode': {
        'values': ['mlp', 'diff', 'mean', 'sum']
        },
    'distance_emb_dim': {
        'values': [16, 32, 48, 64]
        },
    'dropout_p': {
        'values': [0.0, 0.05, 0.1]
        },
    'graph_mode': {
        'values': ['energy', 'vector']
        },
    'max_neighbors': {
        'values': [10, 25, 50]
        },
    'n_conv_layers': {
        'values': [2, 3]
        },
    'n_s': {
          'values': [16, 32, 48, 64]
        },
    'n_v': {
          'values': [16, 32, 48, 64]
        },
    'radius': {
          'values' : [2.5, 5.0, 10.0]
        },
    'sum_mode': {
          'values' : ['node', 'both']
        },
    'lr':  {
        'values' : [0.00005, 0.0001, 0.0005, 0.001]
        },
    'weight_decay' : {
        'values' : [1e-5, 1e-4, 1e-3, 0]
    },
    }

parameters_dict.update({ 'subset': { 'value': None} })
parameters_dict.update({ 'attention': { 'value': None} })
parameters_dict.update({ 'atom_mapping': { 'value': False} })
parameters_dict.update({ 'dataset': { 'value': dataset} })
parameters_dict.update({ 'num_epochs': { 'value': epochs[dataset]} })
parameters_dict.update({ 'train_frac': { 'value': 0.8} })
parameters_dict.update({ 'noH': { 'value': True} })
parameters_dict.update({ 'two_layers_atom_diff': { 'value': False} })
parameters_dict.update({ 'xtb': { 'value': False} })

sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)

wandb_name = 'test'
sweep_id = wandb.sweep(sweep_config, project=project)
print(sweep_id)
wandb.agent(sweep_id, train_wrapper, count=64, project=project)
