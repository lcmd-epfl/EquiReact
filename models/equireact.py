import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from torch_scatter import scatter, scatter_mean, scatter_add
from torch_cluster import radius, radius_graph
from torch_geometric.data import Data

def get_device(tensor):
    int = tensor.get_device()
    if int == 0:
        return 'cuda'
    elif int == -1:
        return 'cpu'
    else:
        return None

class GaussianSmearing(nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, device='cpu'):
        super().__init__()
        self.device = device
        mu = torch.linspace(start, stop, num_gaussians).to(self.device)
        self.coeff = -0.5 / (mu[1] - mu[0]).item() ** 2
        self.register_buffer('mu', mu)

    def forward(self, dist):
        # right now mixed devices
        dist = dist.to(self.device)
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
                 max_radius: float = 10.0, max_neighbors: int = 20,
                 distance_emb_dim: int = 32, dropout_p: float = 0.1,
                 sum_mode='node', verbose=False, device='cpu', mode='energy',
                 **kwargs
                 ):

        super().__init__(**kwargs)

        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.n_s, self.n_v = n_s, n_v
        self.n_conv_layers = n_conv_layers
        self.sum_mode = sum_mode

        self.max_radius = max_radius
        self.max_neighbors = max_neighbors

        self.verbose = verbose
        self.mode = mode

        self.device = device

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

        self.dist_expansion = GaussianSmearing(start=0.0, stop=max_radius, num_gaussians=distance_emb_dim, device=self.device)

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

        # this can also be messed with
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
        return data.x.to(self.device), radius_edges.to(self.device), edge_length_emb.to(self.device), edge_sh.to(self.device)

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

        # want to make sure that we are adding per-atom contributions (and per-bond)?
        if self.sum_mode == 'both':
            data.batch = data.batch.to(self.device)
            edge_batch = data.batch[src].to(self.device)
            scores_nodes = self.score_predictor_nodes(score_inputs_nodes)
            scores_edges = self.score_predictor_edges(score_inputs_edges)
            score = scatter_add(scores_edges, index=edge_batch, dim=0) + scatter_add(scores_nodes, index=data.batch, dim=0)
        elif self.sum_mode == 'node':
            data.batch = data.batch.to(self.device)
            scores_nodes = self.score_predictor_nodes(score_inputs_nodes)
            score = scatter_add(scores_nodes, index=data.batch, dim=0)
        elif self.sum_mode == 'edge':
            edge_batch = data.batch[src].to(self.device)
            scores_edges = self.score_predictor_edges(score_inputs_edges)
            score = scatter_add(scores_edges, index=edge_batch, dim=0)
        else:
            print('sum mode not defined. default to node')
            data.batch = data.batch.to(self.device)
            edge_batch = data.batch[src].to(self.device)
            scores_nodes = self.score_predictor_nodes(score_inputs_nodes)
            scores_edges = self.score_predictor_edges(score_inputs_edges)
            score = scatter_add(scores_edges, index=edge_batch, dim=0) + scatter_add(scores_nodes, index=data.batch, dim=0)
        return score

    def forward_molecules(self, reactants_data, product_data):
        x_r0, edge_index_r0, edge_attr_r0, edge_sh_r0 = self.build_graph(reactants_data[0])
        x_r1, edge_index_r1, edge_attr_r1, edge_sh_r1 = self.build_graph(reactants_data[1])
        x_p, edge_index_p, edge_attr_p, edge_sh_p = self.build_graph(product_data)

        src_r0, dst_r0 = edge_index_r0
        x_r0 = self.node_embedding(x_r0)
        edge_attr_emb_r0 = self.edge_embedding(edge_attr_r0)

        src_r1, dst_r1 = edge_index_r1
        x_r1 = self.node_embedding(x_r1)
        edge_attr_emb_r1 = self.edge_embedding(edge_attr_r1)

        src_p, dst_p = edge_index_p
        x_p = self.node_embedding(x_p)
        edge_attr_emb_p = self.edge_embedding(edge_attr_p)

        for i in range(self.n_conv_layers):
            edge_attr_r0_ = torch.cat([edge_attr_emb_r0, x_r0[dst_r0, :self.n_s], x_r0[src_r0, :self.n_s]], dim=-1)
            x_r0_update = self.conv_layers[i](x_r0, edge_index_r0, edge_attr_r0_, edge_sh_r0)
            x_r0 = F.pad(x_r0, (0, x_r0_update.shape[-1] - x_r0.shape[-1]))
            x_r0 = x_r0 + x_r0_update

            edge_attr_r1_ = torch.cat([edge_attr_emb_r1, x_r1[dst_r1, :self.n_s], x_r1[src_r1, :self.n_s]], dim=-1)
            x_r1_update = self.conv_layers[i](x_r1, edge_index_r1, edge_attr_r1_, edge_sh_r1)
            x_r1 = F.pad(x_r1, (0, x_r1_update.shape[-1] - x_r1.shape[-1]))
            x_r1 = x_r1 + x_r1_update

            edge_attr_p_ = torch.cat([edge_attr_emb_p, x_p[dst_p, :self.n_s], x_p[src_p, :self.n_s]], dim=-1)
            x_p_update = self.conv_layers[i](x_p, edge_index_p, edge_attr_p_, edge_sh_p)
            x_p = F.pad(x_p, (0, x_p_update.shape[-1] - x_p.shape[-1]))
            x_p = x_p + x_p_update

        x_r0 = torch.cat([x_r0[:, :self.n_s], x_r0[:, -self.n_s:]], dim=1) if self.n_conv_layers >= 3 else x_r0[:, :self.n_s]
        x_r1 = torch.cat([x_r1[:, :self.n_s], x_r1[:, -self.n_s:]], dim=1) if self.n_conv_layers >= 3 else x_r1[:, :self.n_s]
        x_p = torch.cat([x_p[:, :self.n_s], x_p[:, -self.n_s:]], dim=1) if self.n_conv_layers >= 3 else x_p[:, :self.n_s]
        if self.verbose:
            print('x0 dims', x_r0.shape, 'x1 dims', x_r1.shape, 'p dims', x_p.shape)

        x = x_p - (x_r0 + x_r1)
        if self.verbose:
            print('reaction x dims', x.shape)

        # assuming the batching is the same for all of them ?
        data_batch = product_data.batch.to(self.device)
        scores_nodes = self.score_predictor_nodes(x)
        score = scatter_add(scores_nodes, index=data_batch, dim=0)

        return score

    def forward(self, reactants_data, product_data):
        """
        :param reactants_data: reactant_0, reactant_1
        :param product_data: single product graph
        :param mode: 'energy' or 'vector' for energy prediction per molecule or diff-vector prediction
        :return: energy prediction
        """

        if self.mode == 'vector':
            print("Running in vector mode, i.e. using diff vector for prediction")
            reaction_energy = self.forward_molecules(reactants_data, product_data)
        else:
            print("Running in energy mode, i.e. using energy diff for prediction")
            batch_size = reactants_data[0].num_graphs

            reactant_energy = torch.zeros((batch_size, 1), device=self.device)
            for i, reactant_graph in enumerate(reactants_data):
                energy = self.forward_molecule(reactant_graph)
                reactant_energy += energy

            product_energy = self.forward_molecule(product_data)

            reaction_energy = product_energy - reactant_energy

        return reaction_energy
