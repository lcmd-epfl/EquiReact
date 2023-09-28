import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from models.equireact import EquiReact
import numpy as np

class AtomMapper(EquiReact):

    def __init__(self, node_fdim: int, edge_fdim: int):
        super().__init__(node_fdim=node_fdim, edge_fdim=edge_fdim)
        self.loss = CrossEntropyLoss()

        self.rp_attention = nn.MultiheadAttention(self.n_s_full_with_edges, 1)  # query, key, value

    def accuracy(self, y_, y):
        return (y_.flatten() == y.flatten()).sum().item() / y_.flatten().size(0) * 100

    def forward(self, reactants_data, products_data):

        x_react = self.forward_repr_mols(reactants_data)
        x_prod = self.forward_repr_mols(products_data)

        def get_atoms(data):
            at = [g.x[:, [0]].to(torch.int) + 1 for g in data if g.x.shape[0] > 0]
            at = self.split_batch(at, data, merge=False)
            return at

        ratoms = get_atoms(reactants_data)
        patoms = get_atoms(products_data)
        nat_max = max(map(len, ratoms))

        torch_att = []

        for xr, xp, ar, ap in zip(x_react, x_prod, ratoms, patoms):
            xr_p, att_t = self.rp_attention(xr, xp, xp, need_weights=True)
            torch_att.append(F.pad(att_t, (0, nat_max-att_t.shape[1])))

        r2p_torch = torch.vstack(torch_att)
        return r2p_torch

    def training_step(self, batch, batch_idx):
       # print("Train step for batch...")
        # assuming attention DL
        rgraphs, pgraphs, targets, mapping, idx = tuple(batch)
        mappings = np.concatenate(mapping)
      #  print('mappings shape', mappings.shape)
        ohe = np.zeros((mappings.size, mappings.max()+1))
        ohe[np.arange(mappings.size), mappings] = 1
        ohe = torch.tensor(ohe, device=self.device)
     #   print('ohe mappings shape', ohe.shape)

        # need reactants_data and products_data
        pred_att = self(rgraphs, pgraphs)
      #  print(f'pred att dims {pred_att.shape}')
        #print('first 10 vals',pred_att[:10])

        # not preds that goes into loss func
        # need something like class probabilities
        loss = self.loss(pred_att, ohe)
       # print(f'loss {loss}')

        preds = torch.argmax(pred_att, dim=1)
        #print(f'preds dims w argmax {preds.shape}')
        true_mappings = torch.tensor(mappings, device=self.device)
        acc = self.accuracy(preds, true_mappings)
       # print(f'accuracy {acc}')
        return loss, acc

    def validation_step(self, batch, batch_idx):
     #   print("Val step...")
        # assuming attention DL
        rgraphs, pgraphs, targets, mapping, idx = tuple(batch)
        mappings = np.concatenate(mapping)
        # print('mappings shape', mappings.shape)
        ohe = np.zeros((mappings.size, mappings.max() + 1))
        ohe[np.arange(mappings.size), mappings] = 1
        ohe = torch.tensor(ohe, device=self.device)
        # print('ohe mappings shape', ohe.shape)

        # need reactants_data and products_data
        pred_att = self(rgraphs, pgraphs)
        # print(f'pred att dims {pred_att.shape}')

        # not preds that goes into loss func
        # need something like class probabilities
        loss = self.loss(pred_att, ohe)
        #    print(f'loss {loss}')

        preds = torch.argmax(pred_att, dim=1)
        # print(f'preds dims w argmax {preds.shape}')
        true_mappings = torch.tensor(mappings, device=self.device)
        acc = self.accuracy(preds, true_mappings)
        # print(f'accuracy {acc}')
        return loss, acc

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
