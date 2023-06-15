import os
import sys
import argparse
from ast import literal_eval
import traceback
from datetime import datetime
from getpass import getuser  # os.getlogin() won't work on a cluster

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
import wandb

# turn on for debugging for C code like Segmentation Faults
import faulthandler
faulthandler.enable()

from trainer.metrics import MAE
from trainer.react_trainer import ReactTrainer
from models.equireact import EquiReact
from process.dataloader_cyclo import Cyclo23TS
from process.dataloader_gdb import GDB722TS
from process.collate import CustomCollator


class Logger(object):
    def __init__(self, logpath, syspart=sys.stdout):
        self.terminal = syspart
        self.log = open(logpath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def parse_arguments(arglist=sys.argv[1:]):
    p = argparse.ArgumentParser()

    g_run = p.add_argument_group('external run parameters')
    g_run.add_argument('--experiment_name'    , type=str           , default=''       ,  help='name that will be added to the runs folder output')
    g_run.add_argument('--wandb_name'         , type=str           , default=None     ,  help='name of wandb run')
    g_run.add_argument('--device'             , type=str           , default='cuda'   ,  help='cuda or cpu')
    g_run.add_argument('--logdir'             , type=str           , default='logs'   ,  help='log dir')
    g_run.add_argument('--checkpoint'         , type=str           , default=None     ,  help='path the checkpoint file to continue training')
    g_run.add_argument('--CV'                 , type=int           , default=5        ,  help='cross validate')
    g_run.add_argument('--num_epochs'         , type=int           , default=2500     ,  help='number of times to iterate through all samples')
    g_run.add_argument('--seed'               , type=int           , default=123      ,  help='initial seed values')
    g_run.add_argument('--verbose'            , action='store_true', default=False    ,  help='Print dims throughout the training process')
    g_run.add_argument('--process'            , action='store_true', default=False    ,  help='(re-)process data by force (if data is already there, default is to not reprocess)?')

    g_hyper = p.add_argument_group('hyperparameters')
    g_hyper.add_argument('--subset'               , type=int           , default=None     ,  help='size of a subset to use instead of the full set (tr+te+va)')
    g_hyper.add_argument('--max_neighbors'        , type=int           , default=20       ,  help='max number of neighbors')
    g_hyper.add_argument('--n_s'                  , type=int           , default=48       ,  help='dimension of node features')
    g_hyper.add_argument('--n_v'                  , type=int           , default=48       ,  help='dimension of extra (p/d) features')
    g_hyper.add_argument('--n_conv_layers'        , type=int           , default=2        ,  help='number of conv layers')
    g_hyper.add_argument('--distance_emb_dim'     , type=int           , default=16       ,  help='how many gaussian funcs to use')
    g_hyper.add_argument('--radius'               , type=float         , default=5.0      ,  help='max radius of graph')
    g_hyper.add_argument('--dropout_p'            , type=float         , default=0.05     ,  help='dropout probability')
    g_hyper.add_argument('--attention'            , type=str           , default=None     ,  help='use attention')
    g_hyper.add_argument('--sum_mode'             , type=str           , default='node'   ,  help='sum node (node, edge, or both)')
    g_hyper.add_argument('--graph_mode'           , type=str           , default='energy' ,  help='prediction mode, energy, or vector')
    g_hyper.add_argument('--dataset'              , type=str           , default='cyclo'  ,  help='cyclo or gdb')
    g_hyper.add_argument('--combine_mode'         , type=str           , default='mean'   ,  help='combine mode diff, sum, or mean')
    g_hyper.add_argument('--atom_mapping'         , action='store_true', default=False    ,  help='use atom mapping')
    g_hyper.add_argument('--random_baseline'      , action='store_true', default=False    ,  help='random baseline (no graph conv)')
    g_hyper.add_argument('--two_layers_atom_diff' , action='store_true', default=False    ,  help='if use two linear layers in non-linear atom diff')

    args = p.parse_args(arglist)

    arg_groups={}
    for group in p._action_groups:
        group_dict={a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    return args, arg_groups


def train(run_dir, run_name, project, wandb_name, hyper_dict,
          #setup args
          device='cuda', seed=123, eval_on_test=True,
          #dataset args
          subset=None, tr_frac = 0.9, te_frac = 0.05, process=False, CV=0,
          dataset='cyclo',
          #sampling / dataloader args
          batch_size=8, num_workers=0, pin_memory=False, # pin memory is not working
          #graph args
          radius=10, max_neighbors=20, sum_mode='node', n_s=16, n_v=16, n_conv_layers=2, distance_emb_dim=32,
          graph_mode='energy', dropout_p=0.1,
          #trainer args
          val_per_batch=True, checkpoint=False, num_epochs=1000000, eval_per_epochs=0, patience=150,
          minimum_epochs=0, models_to_save=[], clip_grad=100, log_iterations=100,
          # adam opt params
          lr=0.0001, weight_decay=0.0001,
          # lr scheduler params
          lr_scheduler=ReduceLROnPlateau, factor=0.6, min_lr=8.0e-6, mode='max', lr_scheduler_patience=60,
          lr_verbose=True,
          verbose=False,
          random_baseline=False,
          combine_mode='diff',
          atom_mapping=False,
          attention=None,
          two_layers_atom_diff=False,
          ):

    device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    print(f"Running on device {device}")

    if dataset=='cyclo':
        data = Cyclo23TS(radius=radius, process=process, atom_mapping=atom_mapping)
    elif dataset=='gdb':
        data = GDB722TS(radius=radius, process=process, atom_mapping=atom_mapping)
    else:
        raise NotImplementedError(f'Cannot load the {dataset} dataset.')
    labels = data.labels
    std = data.std
    print(f"Data stdev {std:.4f}")
    print()

    maes = []

    for i in range(CV):
        print(f"CV iter {i+1}/{CV}")

        hyper_dict['CV iter'] = i
        hyper_dict['seed'] = seed
        wandb.init(project=project,
                   name = wandb_name if CV==1 else f'{wandb_name}.cv{i}',
                   config = hyper_dict,
                   group = None if CV==1 else wandb_name)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        indices = np.arange(len(data))
        np.random.shuffle(indices)

        if subset:
            indices = indices[:subset]
        tr_size = round(tr_frac*len(indices))
        te_size = round(te_frac*len(indices))
        va_size = len(indices)-tr_size-te_size
        tr_indices, te_indices, val_indices = np.split(indices, [tr_size, tr_size+te_size])
        print(f'total / train / test / val: {len(indices)} {len(tr_indices)} {len(te_indices)} {len(val_indices)}')
        train_data = Subset(data, tr_indices)
        val_data = Subset(data, val_indices)
        test_data = Subset(data, te_indices)

        # train sample
        label, idx, r0graph = train_data[0][:3]
        input_node_feats_dim = r0graph.x.shape[1]
        input_edge_feats_dim = 1
        if verbose:
            print(f"{r0graph=}")
            print(f"{input_node_feats_dim=}")
            print(f"{input_edge_feats_dim=}")

        model = EquiReact(node_fdim=input_node_feats_dim, edge_fdim=1, verbose=verbose, device=device,
                          max_radius=radius, max_neighbors=max_neighbors, sum_mode=sum_mode, n_s=n_s, n_v=n_v, n_conv_layers=n_conv_layers,
                          distance_emb_dim=distance_emb_dim, graph_mode=graph_mode, dropout_p=dropout_p, random_baseline=random_baseline,
                          combine_mode=combine_mode, atom_mapping=atom_mapping, attention=attention, two_layers_atom_diff=two_layers_atom_diff)
        print('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

        sampler = None
        custom_collate = CustomCollator(device=device, nreact=data.max_number_of_reactants,
                                        nprod=data.max_number_of_products, atom_mapping=atom_mapping)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
                                  pin_memory=pin_memory, num_workers=num_workers)

        val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=custom_collate, pin_memory=pin_memory,
                                num_workers=num_workers)

        trainer = ReactTrainer(model=model, std=std, device=device, metrics={'mae':MAE()},
                               run_dir=run_dir, run_name=run_name,
                               sampler=sampler, val_per_batch=val_per_batch,
                               checkpoint=checkpoint, num_epochs=num_epochs,
                               eval_per_epochs=eval_per_epochs, patience=patience,
                               minimum_epochs=minimum_epochs, models_to_save=models_to_save,
                               clip_grad=clip_grad, log_iterations=log_iterations,
                               scheduler_step_per_batch = False, # CHANGED THIS
                               lr=lr, weight_decay=weight_decay,
                               lr_scheduler=lr_scheduler, factor=factor, min_lr=min_lr, mode=mode,
                               lr_scheduler_patience=lr_scheduler_patience, lr_verbose=lr_verbose)

        val_metrics, _, _ = trainer.train(train_loader, val_loader)

        if eval_on_test:
            test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=custom_collate,
                                     pin_memory=pin_memory, num_workers=num_workers)
            print('Evaluating on test, with test size: ', len(test_data))

            # file dump for each split
            data_split_string = 'test_split_' + str(CV)
            test_metrics, _, _ = trainer.evaluation(test_loader, data_split=data_split_string)
            mae_split = test_metrics['mae'] * std
            maes.append(mae_split)

        seed += 1
        wandb.finish()
        print()

    if eval_on_test:
        mean_mae_splits = np.mean(maes)
        std_mae_splits = np.std(maes)
        print(f"Mean MAE across splits {mean_mae_splits} +- {std_mae_splits}")
    return


if __name__ == '__main__':

    args, arg_groups = parse_arguments()

    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists(args.logdir):
        print(f"creating log dir {args.logdir}")
        os.mkdir(args.logdir)

    if args.checkpoint:
        run_dir = os.path.dirname(args.checkpoint)
    else:
        run_dir = os.path.join(args.logdir, args.experiment_name)
    if not os.path.exists(run_dir):
        print(f"creating run dir {run_dir}")
        os.mkdir(run_dir)

    logname = f'{datetime.now().strftime("%y%m%d-%H%M%S.%f")}-{getuser()}'
    logpath = os.path.join(run_dir, f'{logname}.log')
    print(f"stdout to {logpath}")
    sys.stdout = Logger(logpath=logpath, syspart=sys.stdout)
    sys.stderr = Logger(logpath=logpath, syspart=sys.stderr)

    project = 'nequireact-gdb' if args.dataset=='gdb' else 'nequireact'
    print(f'wandb name {args.wandb_name}' if args.wandb_name else 'no wandb name specified')

    print("\ninput args", args, '\n')

    train(run_dir, logname, project, args.wandb_name, vars(arg_groups['hyperparameters']), seed=args.seed,
          device=args.device, num_epochs=args.num_epochs, checkpoint=args.checkpoint,
          subset=args.subset, dataset=args.dataset, process=args.process,
          verbose=args.verbose, radius=args.radius, max_neighbors=args.max_neighbors, sum_mode=args.sum_mode,
          n_s=args.n_s, n_v=args.n_v, n_conv_layers=args.n_conv_layers, distance_emb_dim=args.distance_emb_dim,
          graph_mode=args.graph_mode, dropout_p=args.dropout_p, random_baseline=args.random_baseline,
          combine_mode=args.combine_mode, atom_mapping=args.atom_mapping, CV=args.CV, attention=args.attention,
          two_layers_atom_diff=args.two_layers_atom_diff)
