import dgl

def custom_collate(batch):
    r_graph, r_atomtypes, r_coords, p_graph, p_atomtypes, p_coords, label, idx = map(list, zip(*batch))

    return dgl.batch(r_graph), r_atomtypes, r_coords, dgl.batch(p_graph), p_atomtypes, p_coords, label, idx
