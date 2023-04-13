import argparse
import numpy as np
import os
import sys
import traceback
from datetime import datetime
import getpass  # os.getlogin() won't work on a cluster

from commons.logger import Logger
from commons.utils import log, get_CV_splits

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
from process.dataloader import GDB7RXN
from process.collate import custom_collate
from process.samplers import HardSampler

import wandb

def parse_arguments():
    # sparse for now, will be expanded once things are working
    p = argparse.ArgumentParser()
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--num_epochs', type=str, default='2500', help='number of times to iterate through all samples')
    p.add_argument('--checkpoint', type=str, help='path the checkpoint file to continue training')
    p.add_argument('--device', type=str, help='cuda or cpu')
    p.add_argument('--subset', type=int, default=None, help='size of a subset to use instead of the full set (tr+te+va)')
    p.add_argument('--wandb_name', type=str, default=None, help='name of wandb run')

    args = p.parse_args()

    if type(args.num_epochs) == str:
        args.num_epochs = int(args.num_epochs)
    return args


def train(run_dir,
          #setup args
          #TODO implement CV as an option rather than commenting out (do later)
          device='cuda:0', seed=123, eval_on_test=True,
          #dataset args
          subset=None, tr_frac = 0.75, te_frac = 0.125, process=True,
          #sampling / dataloader args
          batch_size=8, num_workers=0, pin_memory=True,
          #graph args
          radius=10,
          #NN args
          n_lays=8,
          #trainer args
          val_per_batch=True, checkpoint=False, num_epochs=1000000, eval_per_epochs=0, patience=150,
          minimum_epochs=0, models_to_save=[], clip_grad=100, log_iterations=100,
          # adam opt params
          lr=0.0001, weight_decay=0.0001,
          # lr scheduler params
          lr_scheduler=ReduceLROnPlateau, factor=0.6, min_lr=8.0e-6, mode='max', lr_scheduler_patience=60,
          lr_verbose=True,
          ):

    device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'cuda' else "cpu")

    data = GDB7RXN(device=device, radius=radius, process=process)

    # for now arbitrary data split
    indices = np.arange(len(data))
    np.random.seed(seed)
    np.random.shuffle(indices)

    if subset:
        indices = indices[:subset]
    tr_size = round(tr_frac*len(indices))
    te_size = round(te_frac*len(indices))
    va_size = len(indices)-tr_size-te_size
    tr_indices, te_indices, val_indices = np.split(indices, [tr_size, tr_size+te_size])
    print('total / train / test / val:', len(indices), len(tr_indices), len(te_indices), len(val_indices))
    train_data = Subset(data, tr_indices)

    # here should normalise data
    val_data = Subset(data, val_indices)
    test_data = Subset(data, te_indices)

    # train sample
    r_graph_0, r_atoms_0, r_coords_0, p_graph_0, p_atoms_0, p_coords_0, label_0, idx_0 = train_data[0]
    input_node_feats_dim = r_graph_0.x.shape[1]
    print(f"input node feats dim {input_node_feats_dim}")
    input_edge_feats_dim = 1
    print(f"input edge feats dim {input_edge_feats_dim}")

    model = EquiReact(node_fdim=input_node_feats_dim, edge_fdim=1)

    print('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))


    sampler = None
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
                                    pin_memory=pin_memory, num_workers=num_workers)

    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=custom_collate, pin_memory=pin_memory,
                            num_workers=num_workers)

    trainer = ReactTrainer(model=model, device=device, metrics={'mae':MAE()},
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
    print('args', args)
    start_time = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists(args.logdir):
        print(f"creating run dir {args.logdir}")
        os.mkdir(args.logdir)

    if args.checkpoint:
        run_dir = os.path.dirname(args.checkpoint)
    else:
        run_dir = os.path.join(args.logdir, args.experiment_name)
    if not os.path.exists(run_dir):
        print(f"creating run dir {run_dir}")
        os.mkdir(run_dir)
    logpath = os.path.join(run_dir, f'{datetime.now().strftime("%y%m%d-%H%M%S")}-{getpass.getuser()}-{os.uname()[1]}.log')
    sys.stdout = Logger(logpath=logpath, syspart=sys.stdout)
    sys.stderr = Logger(logpath=logpath, syspart=sys.stderr)

    wandb.init(project='nequireact')
    if args.wandb_name:
        wandb.run.name = args.wandb_name
        print(args.wandb_name)

    with open(os.path.join('logs', f'{start_time}.log'), "w") as file:
        try:
            #train(run_dir, CV=args.CV, device=args.device, num_epochs=args.num_epochs, checkpoint=args.checkpoint)
            train(run_dir, device=args.device, num_epochs=args.num_epochs, checkpoint=args.checkpoint, subset=args.subset)
        except Exception as e:
            traceback.print_exc(file=file)
            raise
