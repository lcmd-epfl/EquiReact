import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from models.equireact import EquiReact

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
    a = mask * torch.mm(queries, torch.transpose(keys, 1, 0)) - 1000. * (1. - mask)
    a_x = torch.softmax(a, dim=1)  # i->j, NxM, a_x.sum(dim=1) = torch.ones(N)
    attention_x = torch.mm(a_x, values)  # (N,d)
    return attention_x

class AtomMapper(EquiReact):

    def __init__(self, node_fdim: int, edge_fdim: int):
        super().__init__(node_fdim=node_fdim, edge_fdim=edge_fdim)
        self.loss = CrossEntropyLoss()

    def accuracy(self, y_, y):
        return (y_.flatten() == y.flatten()).sum().item() / y_.flatten().size(0) * 100

    def forward(self, reactants_data, products_data):

        if self.sum_mode == 'node':
            predictor = self.score_predictor_nodes
        elif self.sum_mode == 'both':
            predictor  = self.score_predictor_nodes_with_edges
        else:
            raise NotImplementedError(f'sum mode "{self.sum_mode}" is not compatible with vector mode')

        x_react = self.forward_repr_mols(reactants_data)
        x_prod  = self.forward_repr_mols(products_data)


        # Hybrid cross/ masked
        def get_atoms(data):
            at = [g.x[:, [0]].to(torch.int) + 1 for g in data if g.x.shape[0] > 0]
            at = self.split_batch(at, data, merge=False)
            return at

        ratoms = get_atoms(reactants_data)
        patoms = get_atoms(products_data)

        r2p_attention = []
        for xr, xp, ar, ap in zip(x_react, x_prod, ratoms, patoms):
            print(f'xr shape {xr.shape}')
            print(f'xp shape {xp.shape}')
            mask = (ap != ar.T).to(self.device)  # len(xp) × len(xr) ; True == no attention
            print(f'mask shape {mask.shape}')
            # rp or pr ?

            # get Q, K, V
            att_mlp_Q = nn.Sequential(
                nn.Linear(xr.shape, xr.shape, bias=False),
                nn.LeakyReLu(negative_slope=0.01)
            )

            att_mlp_K = nn.Sequential(
                nn.Linear(xp.shape, xp.shape, bias=False),
                nn.LeakyReLu(negative_slope=0.01)
            )

            att_mlp_V = nn.Sequential(
                nn.Linear(xp.shape, xp.shape, bias=False)
            )
            att = cross_attention(att_mlp_Q(xr),
                                  att_mlp_K(xp),
                                  att_mlp_V(xp), mask.T)
            r2p_attention.append(att)
        r2p_attention = torch.tensor(r2p_attention, device=self.device)
        print('before padding', r2p_attention.shape)

        nat_list = [i.shape[0] for i in r2p_attention]
        nat_max = max(nat_list)
        for i, nat in enumerate(nat_list):
            r2p_attention[i] = F.pad(r2p_attention[i], (0, nat_max-nat, 0, nat_max-nat))[None,:,:]
        r2p_attention = torch.cat(r2p_attention)
        print('after padding', r2p_attention.shape)
        return r2p_attention

    def training_step(self, batch, batch_idx):
        # assuming attention DL
        rgraphs, pgraphs, targets, mapping, idx = tuple(batch)
       # print('mapping shape', len(mapping))
      #  print('mapping', mapping)

        # need reactants_data and products_data
        pred_att = self(rgraphs, pgraphs)
        print(f'pred att dims {pred_att.shape}')

        # do we need an argmax here ?
        preds = torch.argmax(pred_att, dim=1)
        print(f'preds dims w argmax {preds.shape}')
        loss = self.loss(preds, true_maps)

        acc = self.accuracy(preds, y_target)
        print(f'accuracy {acc}')
        return loss, acc

    def validation_step(self, batch, batch_idx):
        # assuming attention DL
        true_maps = batch[:][-1] # last item in DL

        # Compute loss_digit for both input images
        pred_att = self(x)
        print(f'pred att dims {pred_att.shape}')

        # do we need an argmax here ?
        preds = torch.argmax(pred_att, dim=1)
        print(f'preds dims w argmax {preds.shape}')
        loss = self.loss(preds, true_maps)

        acc = self.accuracy(preds, y_target)
        print(f'accuracy {acc}')
        return loss, acc

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
