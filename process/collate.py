from torch_geometric.data import Batch 

def custom_collate(batch):
    r_graph, r_atomtypes, r_coords, p_graph, p_atomtypes, p_coords, label, idx = map(list, zip(*batch))

    return Batch.from_data_list(r_graph), r_atomtypes, r_coords, Batch.from_data_list(p_graph), p_atomtypes, p_coords, label, idx
