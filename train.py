import argparse
import numpy as np
import os
import sys
import traceback
from datetime import datetime
import getpass  # os.getlogin() won't work on a cluster

from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove

from torch.utils.data import DataLoader, Subset
import torch

# turn on for debugging for C code like Segmentation Faults
import faulthandler

faulthandler.enable()

# MY IMPORTS
from trainer.metrics import MAE
from trainer.trainer import Trainer
from trainer.react_trainer import ReactTrainer
from models.equireact import EquiReact
from process.dataloader import Cyclo23TS
from process.collate import custom_collate

import wandb

import sys
from datetime import datetime

from ast import literal_eval


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

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment_name', type=str, default='', help='name that will be added to the runs folder output')
    p.add_argument('--num_epochs', type=str, default='2500', help='number of times to iterate through all samples')
    p.add_argument('--checkpoint', type=str, help='path the checkpoint file to continue training')
    p.add_argument('--device', type=str, help='cuda or cpu')
    p.add_argument('--subset', type=int, default=None, help='size of a subset to use instead of the full set (tr+te+va)')
    p.add_argument('--wandb_name', type=str, default=None, help='name of wandb run')
    p.add_argument('--logdir', type=str, default='logs', help='log dir')
    p.add_argument('--process', type=str, default=False, help='(re-)process data by force (if data is already there, default is to not reprocess)?')
    p.add_argument('--verbose', type=str, default=False, help='Print dims throughout the training process')
    p.add_argument('--radius', type=float, default=10.0, help='max radius of graph')
    p.add_argument('--max_neighbors', type=int, default=20, help='max number of neighbors')
    p.add_argument('--sum_mode', type=str, default='node', help='sum node (node, edge, or both)')
    p.add_argument('--n_s', type=int, default=16, help='dimension of node features')
    p.add_argument('--n_v', type=int, default=16, help='dimension of extra (p/d) features')
    p.add_argument('--n_conv_layers', type=int, default=2, help='number of conv layers')
    p.add_argument('--distance_emb_dim', type=int, default=32, help='how many gaussian funcs to use')
    p.add_argument('--dropout_p', type=float, default=0.1, help='dropout probability')

    args = p.parse_args()

    if type(args.verbose) == str:
        args.verbose = literal_eval(args.verbose)
    if type(args.process) == str:
        args.process = literal_eval(args.process)
    if type(args.num_epochs) == str:
        args.num_epochs = int(args.num_epochs)
    if type(args.radius) == str:
        args.radius = float(args.radius)
    if type(args.max_neighbors) == str:
        args.max_neighbors = int(args.max_neighbors)
    if type(args.n_s) == str:
        args.n_s = int(args.n_s)
    if type(args.n_v) == str:
        args.n_v = int(args.n_v)
    if type(args.n_conv_layers) == str:
        args.n_conv_layers = int(args.n_conv_layers)
    if type(args.distance_emb_dim) == str:
        args.distance_emb_dim = int(args.distance_emb_dim)
    if type(args.dropout_p) == str:
        args.dropout_p = float(args.dropout_p)
    return args


def train(run_dir,
          #setup args
          device='cuda:0', seed=123, eval_on_test=True,
          #dataset args
          subset=None, tr_frac = 0.75, te_frac = 0.125, process=False,
          #sampling / dataloader args
          batch_size=8, num_workers=0, pin_memory=False, # pin memory is not working
          #graph args
          radius=10, max_neighbors=20, sum_mode='node', n_s=16, n_v=16, n_conv_layers=2, distance_emb_dim=32,
          dropout_p=0.1,
          #trainer args
          val_per_batch=True, checkpoint=False, num_epochs=1000000, eval_per_epochs=0, patience=150,
          minimum_epochs=0, models_to_save=[], clip_grad=100, log_iterations=100,
          # adam opt params
          lr=0.0001, weight_decay=0.0001,
          # lr scheduler params
          lr_scheduler=ReduceLROnPlateau, factor=0.6, min_lr=8.0e-6, mode='max', lr_scheduler_patience=60,
          lr_verbose=True,
          verbose=False
          ):
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(1)
        np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    print("Running on device", device)

    data = Cyclo23TS(device=device, radius=radius, process=process)
    labels = data.labels
    std = data.std

    print("Data stdev", std)

    # for now arbitrary data split
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    if subset:
        indices = indices[:subset]
    tr_size = round(tr_frac*len(indices))
    te_size = round(te_frac*len(indices))
    va_size = len(indices)-tr_size-te_size
    tr_indices, te_indices, val_indices = np.split(indices, [tr_size, tr_size+te_size])
    print('total / train / test / val:', len(indices), len(tr_indices), len(te_indices), len(val_indices))
    train_data = Subset(data, tr_indices)
    val_data = Subset(data, val_indices)
    test_data = Subset(data, te_indices)

    # train sample
    r_0_graph, r_0_atomtypes, r_0_coords, r_1_graph, r_1_atomtypes, r_1_coords, p_graph, p_atomtypes, p_coords, label, idx = train_data[0]
    print("r_0_graph", r_0_graph)
    input_node_feats_dim = r_0_graph.x.shape[1]
    print(f"input node feats dim {input_node_feats_dim}")
    input_edge_feats_dim = 1
    print(f"input edge feats dim {input_edge_feats_dim}")

    model = EquiReact(node_fdim=input_node_feats_dim, edge_fdim=1, verbose=verbose, device=device,
                      max_radius=radius, max_neighbors=max_neighbors, sum_mode=sum_mode, n_s=n_s, n_v=n_v, n_conv_layers=n_conv_layers,
                      distance_emb_dim=distance_emb_dim, dropout_p=dropout_p)

    print('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    sampler = None
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
                                    pin_memory=pin_memory, num_workers=num_workers)

    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=custom_collate, pin_memory=pin_memory,
                            num_workers=num_workers)

    trainer = ReactTrainer(model=model, std=std, device=device, metrics={'mae':MAE()},
                            run_dir=run_dir, sampler=sampler, val_per_batch=val_per_batch,
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

        test_metrics, _, _ = trainer.evaluation(test_loader, data_split='test')
        return val_metrics, test_metrics, trainer.writer.log_dir

    return val_metrics

if __name__ == '__main__':
    args = parse_arguments()
    start_time = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
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
    # naming is kind of horrible
    logpath = os.path.join(run_dir, f'{datetime.now().strftime("%y%m%d-%H%M%S")}-{getpass.getuser()}.log')
    print('stdout to', logpath)
    sys.stdout = Logger(logpath=logpath, syspart=sys.stdout)
    sys.stderr = Logger(logpath=logpath, syspart=sys.stderr)

    wandb.init(project='nequireact')
    if args.wandb_name:
        wandb.run.name = args.wandb_name
        print('wandb name', args.wandb_name)
    else:
        print('no wandb name specified')
    print('input args', args)
    train(run_dir, device=args.device, num_epochs=args.num_epochs, checkpoint=args.checkpoint, subset=args.subset,
          verbose=args.verbose, radius=args.radius, max_neighbors=args.max_neighbors, sum_mode=args.sum_mode,
          n_s=args.n_s, n_v=args.n_v, n_conv_layers=args.n_conv_layers, distance_emb_dim=args.distance_emb_dim,
          dropout_p=dropout_p)
