import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism as iso
import networkx.algorithms.components.connected as con


def read_mol(fname):
    with open(fname, 'r') as f:
        data = f.readlines()
    atoms = map(lambda x: int(x.split()[1]), filter(lambda x: x[:4]=='atom', data))
    bonds = map(lambda x: list(map(lambda y: int(y)-1, x.split()[1:3])), filter(lambda x: x[:4]=='bond', data))
    return list(atoms), list(bonds)


def make_graph(fname):
    atoms, bonds = read_mol(fname)
    G = nx.Graph()
    list(map(lambda i_q: G.add_node(i_q[0], q=i_q[1]), enumerate(atoms)))
    G.add_edges_from(bonds)
    assert G.number_of_nodes() == len(atoms)
    assert G.number_of_edges() == len(bonds)
    return G


def match(G1, G2):
    GM = iso.GraphMatcher(G1, G2, node_match=iso.categorical_node_match('q', None))
    return GM.is_isomorphic(), GM.match()


def check_pair(G1, G2):
    if G1.number_of_nodes() != G2.number_of_nodes():
        return False
    if G1.number_of_edges() != G2.number_of_edges():
        return False

    Q1 = np.array(sorted(node['q'] for node in G1.nodes.values()))
    Q2 = np.array(sorted(node['q'] for node in G2.nodes.values()))
    return np.all(Q1==Q2)


def load_graphs(tsfile, r0file, r1file, verbose=False):

    # Load the TS file
    G = make_graph(tsfile)

    # Split it into two mols
    components = [G.subgraph(c) for c in con.connected_components(G)]
    assert len(components) == 2

    # Load the reactants
    G_r0 = make_graph(r0file)
    G_r1 = make_graph(r1file)
    assert G.number_of_nodes() == G_r0.number_of_nodes() + G_r1.number_of_nodes()

    # Check the reactant order
    permut0 = check_pair(G_r0, components[0]) and check_pair(G_r1, components[1])
    permut1 = check_pair(G_r0, components[1]) and check_pair(G_r1, components[0])
    if verbose:
        print(permut0, permut1)
    assert permut0!=permut1   # one permutation is correct and other isn't

    if permut1:
        components = components[::-1]

    return G_r0, G_r1, components


def get_peturbation(iso, G):
    src, dst = np.array(sorted(iso.items(), key=lambda iso: iso[0])).T
    assert np.all(src==np.arange(G.number_of_nodes()))
    return dst
