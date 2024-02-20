from torch.optim import Adam
from trainer.trainer import Trainer


class ReactTrainer(Trainer):
    def __init__(self, **kwargs):
        super(ReactTrainer, self).__init__(main_metric='mae',
                                          optim=Adam,
                                          main_metric_goal='min',
                                          **kwargs)
        print(f"In trainer, metrics is {kwargs['metrics']} and std is {kwargs['std']}")

    def forward_pass(self, batch, return_repr=False):
        rgraphs, pgraphs, targets, mapping = tuple(batch)
        y_pred, representations = self.model(rgraphs, pgraphs, mapping=mapping, return_repr=return_repr)
        loss = self.loss_func(y_pred, targets)
        return loss, y_pred, targets, representations
