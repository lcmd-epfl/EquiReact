import torch
from torch_geometric.data import Batch


class CustomCollator(object):
    def __init__(self, device='cpu', nreact=1, nprod=1, atom_mapping=False):
        self.device = device
        self.nreact = nreact
        self.nprod  = nprod
        self.atom_mapping = atom_mapping

    def __call__(self, batch):
        data = list(map(list, zip(*batch)))
        label, idx = data[:2]
        graphs = data[2:2+self.nreact+self.nprod]
        rgraphs = [Batch.from_data_list(graphs[i]) for i in range(self.nreact)]
        pgraphs = [Batch.from_data_list(graphs[i+self.nreact]) for i in range(self.nprod)]
        targets = torch.tensor(label).float().reshape(-1, 1).to(self.device)
        if self.atom_mapping is False:
            return rgraphs, pgraphs, targets, None, idx
        else:
            mapping = data[-1]
            return rgraphs, pgraphs, targets, mapping, idx
