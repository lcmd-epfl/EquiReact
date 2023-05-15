from torch_geometric.data import Batch

def custom_collate(batch):
    r_0_graph, r_1_graph, p_graph, label, idx = map(list, zip(*batch))

    return Batch.from_data_list(r_0_graph), Batch.from_data_list(r_1_graph), Batch.from_data_list(p_graph), label, idx
