import pprint
import wandb
from train import train

def train_wrapper():
    with wandb.init(config=None):
        args = wandb.config
        train(run_dir, logname, project, wandb_name, args, seed=123,
              device='cuda', num_epochs=args.num_epochs, checkpoint=None,
              subset=args.subset, dataset=args.dataset, process=False,
              radius=args.radius, max_neighbors=args.max_neighbors, sum_mode=args.sum_mode,
              n_s=args.n_s, n_v=args.n_v, n_conv_layers=args.n_conv_layers, distance_emb_dim=args.distance_emb_dim,
              graph_mode=args.graph_mode, dropout_p=args.dropout_p, random_baseline=False,
              combine_mode=args.combine_mode, atom_mapping=args.atom_mapping, CV=1, attention=args.attention,
              noH=args.noH, two_layers_atom_diff=None, rxnmapper=False, reverse=False,
              xtb=False, split_complexes=False, sweep=True)

wandb.login()

metric = { 'name': 'val_loss', 'goal': 'minimize' }
sweep_config = { 'method': 'random', 'metric': metric }

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
        'values': [2, 3, 4]
        },
    'n_s': {
          'values': [16, 32, 48, 64]
        },
    'n_v': {
          'values': [16, 32, 48, 64]
        },
    'noH': {
          'values' : [True, False]
        },
    'radius': {
          'values' : [2.5, 5.0, 10.0]
        },
    'sum_mode': {
          'values' : ['node', 'edge', 'both']
        },
    }

parameters_dict.update({ 'num_epochs': { 'value': 16} })
parameters_dict.update({ 'dataset': { 'value': 'gdb'} })
parameters_dict.update({ 'subset': { 'value': 100} })
parameters_dict.update({ 'attention': { 'value': None} })
parameters_dict.update({ 'atom_mapping': { 'value': False} })
sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)

project = 'nequireact-gdb-sweep'



run_dir = 'sweep_test/'
logname = 'sweep_test.log'
wandb_name = 'test'
sweep_id = wandb.sweep(sweep_config, project=project)

print(sweep_id)

wandb.agent(sweep_id, train_wrapper, count=5)

#    'two_layers_atom_diff': {
#          'values' : [True, False]
#        }
#    'attention': {
#        'values': [none, 'self', 'cross', 'masked']
#        },
#    'atom_mapping': {
#        'values': [True, False]
#        },
