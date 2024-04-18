import os
import sys
import argparse
from ast import literal_eval
import traceback
from datetime import datetime
from getpass import getuser  # os.getlogin() won't work on a cluster
import copy
import random

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
from process.dataloader_proparg import Proparg21TS
from process.collate import CustomCollator
from process.splitter import split_dataset


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
    g_run.add_argument('--checkpoint'         , type=str           , default=None     ,  help='path of the checkpoint file to continue training')
    g_run.add_argument('--CV'                 , type=int           , default=1        ,  help='cross validate')
    g_run.add_argument('--num_epochs'         , type=int           , default=2500     ,  help='number of times to iterate through all samples')
    g_run.add_argument('--seed'               , type=int           , default=123      ,  help='initial seed values')
    g_run.add_argument('--verbose'            , action='store_true', default=False    ,  help='Print dims throughout the training process')
    g_run.add_argument('--process'            , action='store_true', default=False    ,  help='(re-)process data by force (if data is already there, default is to not reprocess)?')
    g_run.add_argument('--eval_on_test_split' , action='store_true', default=False    ,  help='print error per test molecule')
    g_run.add_argument('--learning_curve'     , action='store_true', default=False    ,  help='run learning curve (5 tr set sizes)')

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
    g_hyper.add_argument('--dataset'              , type=str           , default='cyclo'  ,  help='cyclo / gdb / proparg')
    g_hyper.add_argument('--combine_mode'         , type=str           , default='mean'   ,  help='combine mode diff, sum, or mean')
    g_hyper.add_argument('--splitter'             , type=str           , default='random' ,  help='what splits to use: random / scaffold / yasc / ydesc')
    g_hyper.add_argument('--atom_mapping'         , action='store_true', default=False    ,  help='use atom mapping')
    g_hyper.add_argument('--rxnmapper'            , action='store_true', default=False    ,  help='take atom mapping from rxnmapper')
    g_hyper.add_argument('--random_baseline'      , action='store_true', default=False    ,  help='random baseline (no graph conv)')
    g_hyper.add_argument('--two_layers_atom_diff' , action='store_true', default=False    ,  help='if use two linear layers in non-linear atom diff')
    g_hyper.add_argument('--noH'                  , action='store_true', default=False    ,  help='if remove H')
    g_hyper.add_argument('--reverse'              , action='store_true', default=False    ,  help='if add reverse reactions')
    g_hyper.add_argument('--split_complexes'      , action='store_true', default=False    ,  help='if split reaction complexes into individual molecules (for future datasets)')
    g_hyper.add_argument('--xtb'                  , action='store_true', default=False    ,  help='if use xtb geometries')
    g_hyper.add_argument('--xtb_subset'           , action='store_true', default=False    ,  help='if use dft geometries but on the xtb subset (for gdb and cyclo)')
    g_hyper.add_argument('--invariant'            , action='store_true', default=False    ,  help='if run "InReact"')
    g_hyper.add_argument('--lr'                   , type=float         , default=0.001    ,  help='learning rate for adam')
    g_hyper.add_argument('--weight_decay'         , type=float         , default=0.0001   ,  help='weight decay for adam')
    g_hyper.add_argument('--train_frac'           , type=float         , default=0.9      ,  help='training fraction to use (val/te will be equally split over rest)')

    args = p.parse_args(arglist)

    arg_groups={}
    for group in p._action_groups:
        group_dict={a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    if args.atom_mapping and args.attention:
        raise RuntimeError
    if args.CV > 1 and args.learning_curve:
        raise RuntimeError

    return args, arg_groups


def train(run_dir, run_name, project, wandb_name, hyper_dict,
          #setup args
          device='cuda', seed=123, eval_on_test=True,
          #dataset args
          subset=None, training_fractions = [0.8], process=False, CV=0,
          dataset='cyclo', splitter='random',
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
          # factor 0.6 okay ?
          # min_lr = lr / 100
          lr_verbose=True,
          verbose=False,
          random_baseline=False,
          combine_mode='diff',
          atom_mapping=False,
          rxnmapper=False,
          attention=None,
          two_layers_atom_diff=False,
          noH=False,
          reverse = False,
          split_complexes=False,
          xtb = False,
          xtb_subset = False,
          sweep = False,
          eval_on_test_split=False,
          print_repr=False,
          invariant=False,
          ):
    device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    print(f"Running on device {device}")

    if dataset=='cyclo':
        data = Cyclo23TS(process=process, atom_mapping=atom_mapping, rxnmapper=rxnmapper, noH=noH, xtb=xtb, xtb_subset=xtb_subset)
    elif dataset=='gdb':
        data = GDB722TS(process=process, atom_mapping=atom_mapping, rxnmapper=rxnmapper, noH=noH, reverse=reverse, xtb=xtb, xtb_subset=xtb_subset)
    elif dataset=='proparg':
        data = Proparg21TS(process=process, atom_mapping=atom_mapping, rxnmapper=rxnmapper, noH=noH, xtb=xtb)
    else:
        raise NotImplementedError(f'Cannot load the {dataset} dataset.')

    labels = data.labels
    std = data.std
    print(f"Data stdev {std:.4f}")
    print()

    for tr_frac in training_fractions:
        maes = []
        rmses = []

        for i in range(CV):
            print(f"CV iter {i+1}/{CV}")

            hyper_dict['CV iter'] = i
            hyper_dict['seed'] = seed
            if not sweep:
                if CV==1:
                    wandb.init(project=project, config=hyper_dict, name=wandb_name, group=None)
                elif len(training_fractions)>1:
                    hyper_dict['train_frac'] = f'{tr_frac}/{max(training_fractions)}'
                    wandb.init(project=project, config=hyper_dict, name=f'{wandb_name}.tr{tr_frac}', group=None)
                else:
                    wandb.init(project=project, config=hyper_dict, name=f'{wandb_name}.cv{i}', group=wandb_name)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            tr_indices, te_indices, val_indices, indices = split_dataset(nreactions=data.nreactions, splitter=splitter,
                                                                         tr_frac=max(training_fractions),
                                                                         dataset=dataset, subset=subset)
            tr_indices = tr_indices[:int(tr_frac*len(indices))]

            if reverse:
                tr_indices = np.hstack((tr_indices, tr_indices+data.nreactions))
                te_indices = np.hstack((te_indices, te_indices+data.nreactions))
                val_indices = np.hstack((val_indices, val_indices+data.nreactions))

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
                              combine_mode=combine_mode, atom_mapping=atom_mapping, attention=attention, two_layers_atom_diff=two_layers_atom_diff,
                              invariant=invariant)
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
                test_metrics, pred, targ = trainer.evaluation(test_loader, data_split=data_split_string, return_pred=True)
                if eval_on_test_split:
                    for x in zip(test_data.indices, np.squeeze(torch.vstack(targ).cpu().numpy()),
                                                    np.squeeze(torch.vstack(pred).cpu().numpy())):
                        print('>>>', *x)

                mae_split = test_metrics['mae'] * std
                rmse_split = np.sqrt(test_metrics['MSELoss'])*std
                maes.append(mae_split)
                rmses.append(rmse_split)
                if wandb.run is not None:
                    wandb.run.summary["test_score"] = mae_split
                    wandb.run.summary["test_rmse"] = rmse_split

                if print_repr:
                    for x_indices, x_loader, x_title in zip((train_data.indices, val_data.indices, test_data.indices),
                                                            (train_loader, val_loader, test_loader),
                                                            ('train', 'val', 'test')):
                        representations = trainer.get_repr(x_loader)
                        for x in zip(x_indices, representations):
                            print(f'>>>{x_title}', x[0], *x[1])

            seed += 1
            if not sweep:
                wandb.finish()
            print()

        if eval_on_test:
            print(f"Mean MAE across splits {np.mean(maes)} +- {np.std(maes)}")
            print(f"Mean RMSE across splits {np.mean(rmses)} +- {np.std(rmses)}")
    return maes, rmses


if __name__ == '__main__':

    args, arg_groups = parse_arguments()

    if args.checkpoint:
        run_dir = os.path.dirname(args.checkpoint)
    else:
        run_dir = os.path.join(args.logdir, args.experiment_name)
    if not os.path.exists(run_dir):
        print(f"creating run dir {run_dir}")
        os.makedirs(run_dir)

    logname = f'{args.wandb_name}-{datetime.now().strftime("%y%m%d-%H%M%S.%f")}-{getuser()}'
    logpath = os.path.join(run_dir, f'{logname}.log')
    print(f"stdout to {logpath}")
    sys.stdout = Logger(logpath=logpath, syspart=sys.stdout)
    sys.stderr = Logger(logpath=logpath, syspart=sys.stderr)

    project = f'nequireact-{args.dataset}-80'
    print(f'wandb name {args.wandb_name}' if args.wandb_name else 'no wandb name specified')

    print("\ninput args", args, '\n')

    if args.learning_curve:
        train_frac = args.train_frac * np.logspace(-4, 0, 5, endpoint=True, base=2)
    else:
        train_frac = [args.train_frac]
    print(train_frac)

    train(run_dir, logname, project, args.wandb_name, vars(arg_groups['hyperparameters']), seed=args.seed,
          device=args.device, num_epochs=args.num_epochs, checkpoint=args.checkpoint,
          subset=args.subset, dataset=args.dataset, process=args.process,
          verbose=args.verbose, radius=args.radius, max_neighbors=args.max_neighbors, sum_mode=args.sum_mode,
          n_s=args.n_s, n_v=args.n_v, n_conv_layers=args.n_conv_layers, distance_emb_dim=args.distance_emb_dim,
          graph_mode=args.graph_mode, dropout_p=args.dropout_p, random_baseline=args.random_baseline,
          combine_mode=args.combine_mode, atom_mapping=args.atom_mapping, CV=args.CV, attention=args.attention,
          noH=args.noH, two_layers_atom_diff=args.two_layers_atom_diff, rxnmapper=args.rxnmapper, reverse=args.reverse,
          xtb=args.xtb, xtb_subset=args.xtb_subset,
          eval_on_test_split=args.eval_on_test_split,
          split_complexes=args.split_complexes, lr=args.lr, weight_decay=args.weight_decay, splitter=args.splitter,
          training_fractions=train_frac,
          invariant=args.invariant)
