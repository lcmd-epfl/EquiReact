from torch_geometric.data import Batch 

def custom_collate(batch):
    r_0_graph, r_0_atomtypes, r_0_coords, r_1_graph, r_1_atomtypes, r_1_coords, p_graph, p_atomtypes, p_coords, label, idx = map(list, zip(*batch))

    return Batch.from_data_list(r_0_graph), r_0_atomtypes, r_0_coords, Batch.from_data_list(r_1_graph), r_1_atomtypes, r_1_coords, Batch.from_data_list(p_graph), p_atomtypes, p_coords, label, idx
