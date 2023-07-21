import torch
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

    def __init__(self):

        self.att_mlp_Q = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope)
            nn.LeakyReLu(negative_slope=0.01)
        )

        self.att_mlp_K = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            nn.LeakyReLu(negative_slope=0.01)
        )

        self.att_mlp_V = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False)
        )

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
            mask = (ap != ar.T).to(self.device)  # len(xp) × len(xr) ; True == no attention
            # rp or pr ?
            att = cross_attention(self.att_mlp_Q(xr),
                                  self.att_mlp_K(xp),
                                  self.att_mlp_V(xp), mask.T)
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
        x, y_class, y_target = batch

        # Compute loss_digit for both input images
        d1, d2, out = self(x)
        loss_d1 = self.loss(d1, y_class[:, 0])
        loss_d2 = self.loss(d2, y_class[:, 1])

        if self.target:
            # Compute loss_target with the target network
            preds = torch.argmax(out, dim=1)
            loss_target = self.loss(out, y_target)

            # Equally weight loss_digit from both input images
            loss_digit = (loss_d1 + loss_d2) / 2

            if self.strategy == "random":
                # Alternate the loss (loss_digit / loss_target) to optimize by choosing the loss at random
                decision = random.randint(0, 1)
                if decision:
                    loss = loss_target
                else:
                    loss = loss_digit
            elif self.strategy == "sum":
                # Sum up the two losses (loss_digit / loss_target)
                loss = self.weight_aux * loss_digit + loss_target
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        else:
            # Simulate the target network with the arithmetic operation '<='
            preds = out
            loss = (loss_d1 + loss_d2) / 2

        acc = self.accuracy(preds, y_target)
        return loss, acc

