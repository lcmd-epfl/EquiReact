import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from models.equireact import EquiReact
import numpy as np

def cross_attention(queries, keys, values, mask):
    """Compute cross attention.
    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    Args:
      queries: NxD float tensor --> queries
      keys: MxD float tensor --> keys
      values: Mxd
      mask: NxM
    Returns:
      attention_x: Nxd float tensor.
    """
    #TODO double check
    a = mask * torch.mm(queries, torch.transpose(keys, 1, 0))
    a_x = torch.softmax(a, dim=1)  # i->j, NxM, a_x.sum(dim=1) = torch.ones(N)
    #attention_x = torch.mm(a_x, values)  # (N,d)
    return a_x

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
        cross_att = []

        for xr, xp, ar, ap in zip(x_react, x_prod, ratoms, patoms):
            mask = (ap != ar.T).to(self.device)  # len(xp) × len(xr) ; True == no attention
      #      print(f'mask shape {mask.shape}')
            xr_p, att_t = self.rp_attention(xr, xp, xp, attn_mask=mask.T, need_weights=True)
       #     print('xr_p shape', xr_p.shape)
            torch_att.append(F.pad(att_t, (0, nat_max-att_t.shape[1])))

            # get Q, K, V
            att_mlp_Q = nn.Sequential(
                nn.Linear(xr.shape[1], xr.shape[1], bias=False),
                nn.LeakyReLU(negative_slope=0.01)
            )

            att_mlp_K = nn.Sequential(
                nn.Linear(xp.shape[1], xp.shape[1], bias=False),
                nn.LeakyReLU(negative_slope=0.01)
            )

            att_mlp_V = nn.Sequential(
                nn.Linear(xp.shape[1], xp.shape[1], bias=False)
            )
            aq = att_mlp_Q(xr)
        #    print('aq shape', aq.shape)
            ak = att_mlp_K(xp)
        #    print('ak shape', ak.shape)
            av = att_mlp_V(xp)
       #     print('av shape', av.shape)
            att = cross_attention(aq,
                                  ak,
                                  av, mask.T)
            att = F.pad(att, (0, nat_max-att.shape[1]))
            #print(att.shape)
         #   print('cross attention computed')
         #   print('att dims', att.shape)
            cross_att.append(att)

        r2p_torch = torch.vstack(torch_att)
        #print('final xr mapped shape', r2p_torch.shape)
        r2p_attention = torch.vstack(cross_att)
        #print('final cross att shape', r2p_attention.shape)
        return r2p_attention

    def training_step(self, batch, batch_idx):
        print("Train step...")
        # assuming attention DL
        rgraphs, pgraphs, targets, mapping, idx = tuple(batch)
        mappings = np.concatenate(mapping)
        #print('mappings shape', mappings.shape)
        ohe = np.zeros((mappings.size, mappings.max()+1))
        ohe[np.arange(mappings.size), mappings] = 1
        ohe = torch.tensor(ohe, device=self.device)
        #print('ohe mappings shape', ohe.shape)

        # need reactants_data and products_data
        pred_att = self(rgraphs, pgraphs)
        #print(f'pred att dims {pred_att.shape}')

        # not preds that goes into loss func
        # need something like class probabilities
        loss = self.loss(pred_att, ohe)

        preds = torch.argmax(pred_att, dim=1)
        #print(f'preds dims w argmax {preds.shape}')
        true_mappings = torch.tensor(mappings, device=self.device)
        acc = self.accuracy(preds, true_mappings)
        print(f'accuracy {acc}')
        return loss, acc

    def validation_step(self, batch, batch_idx):
        print("Val step...")
        # assuming attention DL
        true_maps = batch[:][-1] # last item in DL

        # Compute loss_digit for both input images
        pred_att = self(x)
        #print(f'pred att dims {pred_att.shape}')

        # do we need an argmax here ?
        preds = torch.argmax(pred_att, dim=1)
        #print(f'preds dims w argmax {preds.shape}')
        loss = self.loss(preds, true_maps)

        acc = self.accuracy(preds, y_target)
        print(f'accuracy {acc}')
        return loss, acc

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
