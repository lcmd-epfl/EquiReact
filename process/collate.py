import torch
from torch_geometric.data import Batch


class CustomCollator(object):
    def __init__(self, device='cpu', nreact=1, nprod=1):
        self.device = device
        self.nreact = nreact
        self.nprod  = nprod

    def __call__(self, batch):
        data = list(map(list, zip(*batch)))
        label, idx = data[:2]
        graphs = data[2:]
        rgraphs = [Batch.from_data_list(graphs[i]) for i in range(self.nreact)]
        pgraphs = [Batch.from_data_list(graphs[i+self.nreact]) for i in range(self.nprod)]
        targets = torch.tensor(label).float().reshape(-1, 1).to(self.device)
        return rgraphs, pgraphs, targets, idx
