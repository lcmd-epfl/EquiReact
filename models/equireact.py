import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from torch_scatter import scatter, scatter_mean, scatter_add
from torch_cluster import radius_graph


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
                 sum_mode='node', verbose=False, device='cpu', graph_mode='energy',
                 random_baseline=False, combine_mode='diff', atom_mapping=False,
                 attention=None, two_layers_atom_diff=False,
                 **kwargs):

        super().__init__(**kwargs)

        self.node_fdim = node_fdim
        self.edge_fdim = edge_fdim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.n_s, self.n_v = n_s, n_v
        self.n_conv_layers = n_conv_layers
        self.sum_mode = sum_mode
        self.combine_mode = combine_mode
        self.atom_mapping = atom_mapping
        self.attention = attention
        self.n_s_full = 2 * self.n_s if self.n_conv_layers >= 3 else self.n_s
        self.distance_emb_dim = distance_emb_dim

        self.max_radius = max_radius
        self.max_neighbors = max_neighbors

        self.verbose = verbose
        self.graph_mode = graph_mode

        self.device = device

        self.random_baseline = random_baseline
        if self.random_baseline:
            self.graph_mode = 'node'
            print("random baseline is on, i.e. features will be replaced with random numbers")

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
            nn.Linear(2 * self.n_s_full + distance_emb_dim, self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.n_s, self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.n_s, 1)
        )

        # this can also be messed with
        self.score_predictor_nodes = nn.Sequential(
            nn.Linear(self.n_s_full, 2 * self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * self.n_s, self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.n_s, 1)
        )

        self.score_predictor_nodes_with_edges = nn.Sequential(
            nn.Linear(self.n_s_full + distance_emb_dim, 2 * self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * self.n_s, self.n_s),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.n_s, 1)
        )

        self.energy_mlp = nn.Linear(2, 1)

        if self.sum_mode=='both':
            self.nodes_mlp = nn.Sequential(
                nn.Linear(2*(self.n_s_full + distance_emb_dim), self.n_s_full + distance_emb_dim)
            )
        else:
            self.nodes_mlp = nn.Sequential(
                nn.Linear(2*self.n_s_full, self.n_s_full)
            )

        n_s_full_with_edges = self.n_s_full + distance_emb_dim  if self.sum_mode=='both' else self.n_s_full
        if two_layers_atom_diff:
            self.atom_diff_nonlin = nn.Sequential(
                nn.Linear(n_s_full_with_edges, n_s_full_with_edges),
                nn.ReLU(),
                nn.Linear(n_s_full_with_edges, n_s_full_with_edges),
            )
        else:
            self.atom_diff_nonlin = nn.Sequential(
                nn.ReLU(),
                nn.Linear(n_s_full_with_edges, n_s_full_with_edges),
            )

        self.rp_attention = nn.MultiheadAttention(n_s_full_with_edges, 1)  # query, key, value

        combine_diff = lambda r, p: p-r
        combine_sum  = lambda r, p: r+p
        combine_mean = lambda r, p: (r+p)*0.5
        if self.atom_mapping is True or self.attention is not None or self.graph_mode=='vector':
            combine_mlp  = lambda r, p: self.nodes_mlp(torch.cat((r, p), 1))
        else:
            combine_mlp  = lambda r, p: self.energy_mlp(torch.cat((r, p), 1))
        combine_dict = {'diff': combine_diff, 'difference': combine_diff,
                        'sum' : combine_sum,
                        'mean': combine_mean, 'average': combine_mean, 'avg' : combine_mean,
                        'mlp' : combine_mlp}
        if not self.combine_mode in combine_dict:
            raise NotImplementedError(f'combine mode "{self.combine_mode}" not defined')
        self.combine = combine_dict[self.combine_mode]


    def build_graph(self, data):

        radius_edges = radius_graph(data.pos, self.max_radius, data.batch)

        src, dst = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        edge_length_emb = self.dist_expansion(edge_vec.norm(dim=-1))

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return data.x.to(self.device), radius_edges.to(self.device), edge_length_emb.to(self.device), edge_sh.to(self.device)


    def forward_repr_mol(self, data):

        x, edge_index, edge_attr, edge_sh = self.build_graph(data)
        if self.verbose:
            print('dim of x', x.shape)
            print('dim of radius_graph (edges)', edge_index.shape)
            print('dim of edge length emb (gaussians)', edge_attr.shape)
            print('dim of edge sph harmonics', edge_sh.shape)

        x = self.node_embedding(x)
        edge_attr_emb = self.edge_embedding(edge_attr)
        if self.verbose:
            print('dim of x after node embedding', x.shape)
            print('dim of radius_graph (edges) after embedding', edge_attr_emb.shape)

        src, dst = edge_index
        for i in range(self.n_conv_layers):
            edge_attr_ = torch.cat([edge_attr_emb, x[dst, :self.n_s], x[src, :self.n_s]], dim=-1)
            x_update = self.conv_layers[i](x, edge_index, edge_attr_, edge_sh)
            x = F.pad(x, (0, x_update.shape[-1] - x.shape[-1]))
            x = x + x_update

        x = torch.cat([x[:, :self.n_s], x[:, -self.n_s:]], dim=1) if self.n_conv_layers >= 3 else x[:, :self.n_s]
        return x, edge_index, edge_attr


    def split_batch(self, X_in, data, merge=False):
        batch_size = data[0].num_graphs
        X = []
        for graph, x in zip(data, X_in):
            if graph.x.shape[0]==0:
                continue
            # split into molecules
            sections = [np.count_nonzero(graph.batch==i) for i in range(batch_size)]
            X.append(torch.split(x, sections))
        # regroup so mols from the same reaction are back-to-back
        X_out = [torch.vstack(x) for x in zip(*X)]
        if merge:
            X_out = torch.vstack(X_out)
        return X_out


    def forward_repr_mols(self, data, merge=False):
        X = []
        for graph in data:
            if graph.x.shape[0]==0:
                continue
            x, (src, dst), edge_attr = self.forward_repr_mol(graph)
            if self.sum_mode == 'both':
                xedge = scatter_add(edge_attr, index=src, dim=0)
                xedge = F.pad(xedge, (0, 0, 0, x.shape[0]-xedge.shape[0]))
                x = torch.hstack((x, xedge))
            X.append(x)
        X = self.split_batch(X, data, merge=merge)
        return X


    def forward_molecule(self, data):

        if data.x.shape[0]==0:
            return torch.zeros((data.num_graphs, 1), device=self.device)

        x, (src, dst), edge_attr = self.forward_repr_mol(data)
        data.batch = data.batch.to(self.device)

        if self.random_baseline:
            # reset features to crap of the same dims
            x = torch.rand(x.shape)

        if self.sum_mode == 'both':
            score_inputs_nodes = x
            score_inputs_edges = torch.cat([edge_attr, x[src], x[dst]], dim=-1)

            edge_batch = data.batch[src]
            scores_nodes = self.score_predictor_nodes(score_inputs_nodes)
            scores_edges = self.score_predictor_edges(score_inputs_edges)

            score_node = scatter_add(scores_nodes, index=data.batch, dim=0)
            score_edge = scatter_add(scores_edges, index=edge_batch, dim=0)
            score_edge = F.pad(score_edge, (0, 0, 0, score_node.shape[0]-score_edge.shape[0]))
            score = score_node + score_edge
        elif self.sum_mode == 'node':
            score_inputs_nodes = x
            scores_nodes = self.score_predictor_nodes(score_inputs_nodes)
            score = scatter_add(scores_nodes, index=data.batch, dim=0)
        elif self.sum_mode == 'edge':
            score_inputs_edges = torch.cat([edge_attr, x[src], x[dst]], dim=-1)
            edge_batch = data.batch[src]
            scores_edges = self.score_predictor_edges(score_inputs_edges)
            score = scatter_add(scores_edges, index=edge_batch, dim=0)
        else:
            raise RuntimeError(f'sum mode {self.sum_mode} not defined')

        padsize = data.num_graphs-score.shape[0]
        if padsize>0:
            score = F.pad(score, (0, 0, 0, padsize))
        return score


    def forward_vector_mode(self, reactants_data, products_data, batch_size):

        if self.sum_mode=='node':
            x_size = self.n_s_full
        elif self.sum_mode == 'both':
            x_size = self.n_s_full + self.distance_emb_dim
        else:
            raise NotImplementedError(f'sum mode "{self.sum_mode}" is not compatible with vector mode')
        X_r = torch.zeros((batch_size, x_size), device=self.device)
        X_p = torch.zeros_like(X_r)

        for graphs, X_x in zip([reactants_data, products_data], [X_r, X_p]):
            for graph in graphs:
                if graph.x.shape[0]==0:
                    continue
                x, (src, dst), edge_attr = self.forward_repr_mol(graph)
                if self.sum_mode == 'both':
                    xedge = scatter_add(edge_attr, index=src, dim=0)
                    xedge = F.pad(xedge, (0, 0, 0, x.shape[0]-xedge.shape[0]))
                    x = torch.hstack((x, xedge))
                x = scatter_add(x, index=graph.batch.to(self.device), dim=0)
                x = F.pad(x, (0, 0, 0, batch_size-x.shape[0]))
                X_x += x
        X = self.combine(X_r, X_p)

        if self.verbose:
            print('reaction X dims', X.shape)

        if self.sum_mode == 'node':
            score = self.score_predictor_nodes(X)
        elif self.sum_mode == 'both':
            score = self.score_predictor_nodes_with_edges(X)
        return score


    def forward_energy_mode(self, reactants_data, products_data, batch_size):
        product_energy = torch.zeros((batch_size, 1), device=self.device)
        reactant_energy = torch.zeros((batch_size, 1), device=self.device)
        for graph in reactants_data:
            reactant_energy += self.forward_molecule(graph)
        for graph in products_data:
            product_energy += self.forward_molecule(graph)

        reaction_energy = self.combine(reactant_energy, product_energy)
        return reaction_energy


    def forward_mapped_mode(self, reactants_data, products_data, mapping, return_repr=False):

        if self.sum_mode == 'node':
            predictor = self.score_predictor_nodes
        elif self.sum_mode == 'both':
            predictor  = self.score_predictor_nodes_with_edges
        else:
            raise NotImplementedError(f'sum mode "{self.sum_mode}" is not compatible with vector mode')

        x_react = self.forward_repr_mols(reactants_data)
        x_prod  = self.forward_repr_mols(products_data)

        # mapping overrides attention
        if self.atom_mapping is True:
            x_react_mapped = x_react
            x_prod_mapped = [xp[mp] for xp, mp in zip(x_prod, mapping)]

        elif self.attention is not None:
            if self.attention == 'self':
                x_react_mapped = [self.rp_attention(xr, xr, xr, need_weights=False)[0] for xr in x_react]
                x_prod_mapped  = [self.rp_attention(xp, xp, xp, need_weights=False)[0] for xp in x_prod]
            elif self.attention == 'cross':
                x_react_mapped = [self.rp_attention(xp, xr, xr, need_weights=False)[0] for xp, xr in zip(x_prod, x_react)]
                x_prod_mapped  = [self.rp_attention(xr, xp, xp, need_weights=False)[0] for xp, xr in zip(x_prod, x_react)]
            elif self.attention == 'masked':
                def get_atoms(data):
                    at = [g.x[:,[0]].to(torch.int)+1 for g in data if g.x.shape[0]>0]
                    at = self.split_batch(at, data, merge=False)
                    return at
                ratoms = get_atoms(reactants_data)
                patoms = get_atoms(products_data)
                x_react_mapped = []
                x_prod_mapped = []
                for xr, xp, ar, ap in zip(x_react, x_prod, ratoms, patoms):
                    mask = (ap != ar.T).to(self.device)  # len(xp) × len(xr) ; True == no attention
                    x_react_mapped.append(self.rp_attention(xp, xr, xr, attn_mask=mask, need_weights=False)[0])
                    x_prod_mapped.append(self.rp_attention(xr, xp, xp, attn_mask=mask.T, need_weights=False)[0])
            else:
                raise NotImplementedError(f'attention "{self.attention}" not defined')

        x = self.combine(torch.vstack(x_react_mapped), torch.vstack(x_prod_mapped))

        batch = torch.sort(torch.hstack([g.batch for g in reactants_data])).values.to(self.device)
        if self.graph_mode == 'energy':
            score_atom = predictor(x)
            score = scatter_add(score_atom, index=batch, dim=0)
        elif self.graph_mode == 'vector':
            x = self.atom_diff_nonlin(x)
            x = scatter_add(x, index=batch, dim=0)
            score = predictor(x)

        if return_repr and self.graph_mode == 'vector':
            return score, x
        else:
            return score, None


    def forward(self, reactants_data, products_data, mapping=None, return_repr=False):
        """
        :param reactants_data: reactant graphs
        :param products_data: product graphs
        :param mode: 'energy' or 'vector' for energy prediction per molecule or diff-vector prediction
        :return: energy prediction
        """

        batch_size = reactants_data[0].num_graphs

        if self.atom_mapping is True or self.attention is not None:
            reaction_energy, representations = self.forward_mapped_mode(reactants_data, products_data, mapping, return_repr=return_repr)
        elif self.graph_mode == 'vector':
            reaction_energy = self.forward_vector_mode(reactants_data, products_data, batch_size)
            representations = None
        else:
            reaction_energy = self.forward_energy_mode(reactants_data, products_data, batch_size)
            representations = None

        return reaction_energy, representations
