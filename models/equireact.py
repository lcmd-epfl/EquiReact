import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from torch_scatter import scatter, scatter_mean, scatter_add
from torch_cluster import radius, radius_graph
from torch_geometric.data import Data

class GaussianSmearing(nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        mu = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (mu[1] - mu[0]).item() ** 2
        self.register_buffer('mu', mu)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.mu.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class TensorProductConvLayer(nn.Module):

    def __init__(self, in_irreps, sh_irreps, out_irreps, edge_fdim, residual=True, dropout=0.0,
                 h_dim=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if h_dim is None:
            h_dim = edge_fdim

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc_net = nn.Sequential(
            nn.Linear(edge_fdim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, tp.weight_numel)
        )

    def forward(self, x, edge_index, edge_attr, edge_sh, out_nodes=None, aggr='mean'):
        edge_src, edge_dst = edge_index
        tp_out = self.tp(x[edge_src], edge_sh, self.fc_net(edge_attr))

        out_nodes = out_nodes or x.shape[0]

        out = scatter(src=tp_out, index=edge_dst, dim=0, dim_size=out_nodes, reduce=aggr)
        # assert out.shape[0] == x.shape[0]

        if self.residual:
            padded = F.pad(x, (0, out.shape[-1] - x.shape[-1]))
            out = out + padded

        return out


class EquiReact(nn.Module):

    def __init__(self, node_fdim: int, edge_fdim: int, sh_lmax: int = 2,
                 n_s: int = 16, n_v: int = 16, n_conv_layers: int = 2,
                 # n_s, n_v were 16 originally
                 # maybe radius 5
                 max_radius: float = 10.0, max_neighbors: int = 20,
                 distance_emb_dim: int = 32, dropout_p: float = 0.1,
                 edge_in_score=False, verbose=False, **kwargs
                 ):

        super().__init__(**kwargs)

        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.n_s, self.n_v = n_s, n_v
        self.n_conv_layers = n_conv_layers
        self.edge_in_score = edge_in_score

        self.max_radius = max_radius
        self.max_neighbors = max_neighbors

        self.verbose = verbose

        irrep_seq = [
            f"{n_s}x0e",
            f"{n_s}x0e + {n_v}x1o",
            f"{n_s}x0e + {n_v}x1o + {n_v}x1e",
            f"{n_s}x0e + {n_v}x1o + {n_v}x1e + {n_s}x0o"
        ]

        self.node_embedding = nn.Sequential(
            nn.Linear(node_fdim, n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
            nn.Linear(n_s, n_s)
        )
        self.edge_embedding = nn.Sequential(
            # TODO check: current input dim is ...×distance_emb_dim without edge_fdim
            nn.Linear(distance_emb_dim, n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p) if dropout_p else nn.Identity(),
            nn.Linear(n_s, n_s)
        )

        self.dist_expansion = GaussianSmearing(start=0.0, stop=max_radius, num_gaussians=distance_emb_dim)

        conv_layers = []
        for i in range(n_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]

            parameters = {
                "in_irreps": in_irreps,
                "sh_irreps": self.sh_irreps,
                "out_irreps": out_irreps,
                "edge_fdim": 3 * n_s,
                "h_dim": 3 * n_s,
                "residual": False,
                "dropout": dropout_p
            }

            layer = TensorProductConvLayer(**parameters)
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)

        self.score_predictor_edges = nn.Sequential(
            nn.Linear(4 * self.n_s + distance_emb_dim if n_conv_layers >= 3 else 2 * self.n_s + distance_emb_dim,
                      self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.n_s, self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.n_s, 1)
        )

        self.score_predictor_nodes = nn.Sequential(
            nn.Linear(2 * self.n_s if n_conv_layers >= 3 else self.n_s, 2 * self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * self.n_s, self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.n_s, 1)
        )

    def build_graph(self, data):
        pos = torch.from_numpy(np.vstack(data.pos)).to(torch.float32)   # TODO do something so it's not nparray

        radius_edges = radius_graph(pos, self.max_radius, data.batch)

        src, dst = radius_edges
        edge_vec = pos[dst.long()] - pos[src.long()]
        edge_length_emb = self.dist_expansion(edge_vec.norm(dim=-1))

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return data.x, radius_edges, edge_length_emb, edge_sh

    def forward_molecule(self, data):
        x, edge_index, edge_attr, edge_sh = self.build_graph(data)
        if self.verbose:
            print('dim of x', x.shape)
            print('dim of radius_graph (edges)', edge_index.shape)
            print('dim of edge length emb (gaussians)', edge_attr.shape)
            print('dim of edge sph harmonics', edge_sh.shape)
        src, dst = edge_index
        x = self.node_embedding(x)
        if self.verbose:
            print('dim of x after node embedding', x.shape)
        edge_attr_emb = self.edge_embedding(edge_attr)
        if self.verbose:
            print('dim of radius_graph (edges) after embedding', edge_attr_emb.shape)

        for i in range(self.n_conv_layers):
            if self.verbose:
                print('conv layer', i+1, '/', self.n_conv_layers)
            edge_attr_ = torch.cat([edge_attr_emb, x[dst, :self.n_s], x[src, :self.n_s]], dim=-1)
            x_update = self.conv_layers[i](x, edge_index, edge_attr_, edge_sh)
            if self.verbose:
                print('after update, new xdims', x_update.shape)
            x = F.pad(x, (0, x_update.shape[-1] - x.shape[-1]))

            if self.verbose:
                print('after pad, new xdims', x.shape)
            x = x + x_update

            if self.verbose:
                print('after conv, new x dims', x.shape)

        # remove extra stuff from x
        x = torch.cat([x[:, :self.n_s], x[:, -self.n_s:]], dim=1) if self.n_conv_layers >= 3 else x[:, :self.n_s]
        if self.verbose:
            print('concat x dims', x.shape)

        score_inputs_edges = torch.cat([edge_attr, x[src], x[dst]], dim=-1)
        if self.verbose:
            print('concatenated score_inputs_edges dims', score_inputs_edges.shape)
        score_inputs_nodes = x
        if self.verbose:
            print('score_inputs_nodes dims', score_inputs_nodes.shape)

        scores_nodes = self.score_predictor_nodes(score_inputs_nodes)
        scores_edges = self.score_predictor_edges(score_inputs_edges)

        edge_batch = data.batch[src]

        # want to make sure that we are adding per-atom contributions (and per-bond)?
        if self.edge_in_score:
            score = scatter_add(scores_edges, index=edge_batch, dim=0) + scatter_add(scores_nodes, index=data.batch, dim=0)
        else:
            score = scatter_add(scores_nodes, index=data.batch, dim=0)
        return score

    def forward(self, reactants_data, product_data):
        """
        :param reactants_data: reactant_0, reactant_1
        :param product_data: single product graph
        :return: energy prediction
        """

        batch_size = reactants_data[0].num_graphs

        reactant_energy = torch.zeros((batch_size, 1))
        for i, reactant_graph in enumerate(reactants_data):
            energy = self.forward_molecule(reactant_graph)
            reactant_energy += energy

        product_energy = self.forward_molecule(product_data)

        reaction_energy = product_energy - reactant_energy

        return reaction_energy
