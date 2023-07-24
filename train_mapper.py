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
def train(
          #setup args
          device='cuda', seed=123, eval_on_test=True,
          #dataset args
          subset=None, tr_frac = 0.9, te_frac = 0.05, process=False, CV=0,
          dataset='cyclo',
          #sampling / dataloader args
          batch_size=8, num_workers=0, pin_memory=False, # pin memory is not working
          #graph args
          radius=10, max_neighbors=20, sum_mode='node', n_s=16, n_v=16, n_conv_layers=2, distance_emb_dim=32,
          graph_mode='energy', dropout_p=0.1
          ):

    print("NO WANDB INTEGRATION FOR NOW.")

    device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'cuda' else "cpu")
    print(f"Running on device {device}")

    data = GDB722TS(process=process, atom_mapping=True)

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

    trainer = MapTrainer(nb_epochs=1)

    train_loss_epochs, train_acc_epochs, val_acc_epochs = trainer.train(model, dl_train, dl_val, verbose=True)

    if eval_on_test:
        dl_test = DataLoader(test_data, batch_size=batch_size, collate_fn=custom_collate,
                                     pin_memory=pin_memory, num_workers=num_workers)
        print('Evaluating on test, with test size: ', len(test_data))

        test_acc = trainer.test(model, dl_test, test_verbose=True, return_acc=True)
        print(f'avg acc test {test_acc}')

    return


if __name__ == '__main__':

    train()
