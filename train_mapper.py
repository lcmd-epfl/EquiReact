import os
import sys
import argparse
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

from trainer.map_trainer import MapTrainer
from models.mapper import AtomMapper
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

def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument('--experiment_name'    , type=str           , default=''       ,  help='name that will be added to the runs folder output')
    p.add_argument('--wandb_name'         , type=str           , default=None     ,  help='name of wandb run')
    p.add_argument('--device'             , type=str           , default='cuda'   ,  help='cuda or cpu')
    p.add_argument('--logdir'             , type=str           , default='logs_map'   ,  help='log dir')
    p.add_argument('--CV'                 , type=int           , default=1        ,  help='cross validate')
    p.add_argument('--num_epochs'         , type=int           , default=10     ,  help='number of times to iterate through all samples')
    p.add_argument('--seed'               , type=int           , default=123      ,  help='initial seed values')
    p.add_argument('--verbose'            , action='store_true', default=True    ,  help='Print dims throughout the training process')
    p.add_argument('--process'            , action='store_true', default=False    ,  help='(re-)process data by force (if data is already there, default is to not reprocess)?')

    args = p.parse_args()

    return args

def train(project, wandb_name,
          #setup args
          device='cuda', seed=123, eval_on_test=True,
          #dataset args
          subset=None, tr_frac = 0.9, te_frac = 0.05, process=False, CV=0,
          #sampling / dataloader args
          batch_size=8, num_workers=0, pin_memory=False, # pin memory is not working
          ):

    device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    print(f"Running on device {device}")

    data = GDB722TS(process=process, atom_mapping=True)

    accs = []

    for i in range(CV):
        print(f"CV iter {i+1}/{CV}")

        wandb.init(project=project,
                   name = wandb_name if CV==1 else f'{wandb_name}.cv{i}',
                   group = None if CV==1 else wandb_name)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        indices = np.arange(data.nreactions)
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

        model = AtomMapper(node_fdim=input_node_feats_dim, edge_fdim=1)
        print('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

        custom_collate = CustomCollator(device=device, nreact=data.max_number_of_reactants,
                                        nprod=data.max_number_of_products, atom_mapping=True)

        dl_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
                                      pin_memory=pin_memory, num_workers=num_workers)

        dl_val = DataLoader(val_data, batch_size=batch_size, collate_fn=custom_collate, pin_memory=pin_memory,
                                    num_workers=num_workers)

        trainer = MapTrainer(nb_epochs=10)

        train_loss_epochs, train_acc_epochs, val_acc_epochs = trainer.train(model, dl_train, dl_val, verbose=True)


        if eval_on_test:
            dl_test = DataLoader(test_data, batch_size=batch_size, collate_fn=custom_collate,
                                         pin_memory=pin_memory, num_workers=num_workers)
            print('Evaluating on test, with test size: ', len(test_data))

            test_acc = trainer.test(model, dl_test, test_verbose=True, return_acc=True)
            accs.append(test_acc)
            print(f'avg acc test {test_acc}')

        seed += 1
        wandb.finish()
        print()

    print(f'mean acc over {CV} folds, {np.mean(accs)} +- {np.std(accs)}')

    return


if __name__ == '__main__':

    args = parse_arguments()

    if not os.path.exists(args.logdir):
        print(f"creating log dir {args.logdir}")
        os.mkdir(args.logdir)


    run_dir = os.path.join(args.logdir, args.experiment_name)
    if not os.path.exists(run_dir):
        print(f"creating run dir {run_dir}")
        os.mkdir(run_dir)

    logname = f'{datetime.now().strftime("%y%m%d-%H%M%S.%f")}-{getuser()}'
    logpath = os.path.join(run_dir, f'{logname}.log')
    print(f"stdout to {logpath}")
    sys.stdout = Logger(logpath=logpath, syspart=sys.stdout)
    sys.stderr = Logger(logpath=logpath, syspart=sys.stderr)

    print(f'wandb name {args.wandb_name}' if args.wandb_name else 'no wandb name specified')

    print("\ninput args", args, '\n')

    project = 'map-gdb-supervised'
    train(project, args.wandb_name,
          #setup args
          device=args.device, seed=args.seed, eval_on_test=True,
          #dataset args
          subset=None, tr_frac = 0.9, te_frac = 0.05, process=False, CV=args.CV,
          batch_size=8, num_workers=0, pin_memory=False, # pin memory is not working
          )
