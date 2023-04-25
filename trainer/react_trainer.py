import torch
from torch.optim import Adam
from trainer.trainer import Trainer
from trainer.metrics import MAE


class ReactTrainer(Trainer):
    def __init__(self, **kwargs):
        super(ReactTrainer, self).__init__(main_metric='mae',
                                          optim=Adam,
                                          main_metric_goal='min',
                                          **kwargs)
        print(f"In trainer, metrics is {kwargs['metrics']} and std is {kwargs['std']}")

    def forward_pass(self, batch):
        #import code; code.interact(local=locals())
        r1_graph, r1_atomtypes, r1_coords, \
                r2_graph, r2_atomtypes, r2_coords, \
                p_graph, p_atomtypes, p_coords, \
                targets = tuple(batch)
        targets = torch.tensor(targets).float().reshape(-1, 1).to(self.device) # TODO nicer
        y_pred = self.model([r1_graph, r2_graph], p_graph)
        loss = self.loss_func(y_pred, targets)
        return loss, y_pred, targets
