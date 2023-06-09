import copy
import inspect
import os
import shutil
from typing import Dict, Callable

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import wandb
import pyaml

from models import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
from trainer.lr_schedulers import WarmUpWrapper


def move_to_device(element, device):
    '''
    TODO maybe add other types
    takes arbitrarily nested list and moves everything in it to device if it is a torch tensor
    :param element: arbitrarily nested list
    :param device:
    :return:
    '''
    if isinstance(element, list):
        return [move_to_device(x, device) for x in element]
    else:
        return element.to(device) if isinstance(element,(torch.Tensor)) else element

def list_detach(element):
    '''
    takes arbitrarily nested list and detaches everyting from computation graph
    :param element: arbitrarily nested list
    :return:
    '''
    if isinstance(element, list):
        return [list_detach(x) for x in element]
    else:
        return element.detach()

def concat_if_list(tensor_or_tensors):
    return torch.cat(tensor_or_tensors) if isinstance(tensor_or_tensors, list) else tensor_or_tensors


class Trainer():
    def __init__(self, model, metrics: Dict[str, Callable], main_metric: str, device: torch.device,
                 optim=Adam, main_metric_goal: str = 'min',
                 loss_func=torch.nn.MSELoss(), scheduler_step_per_batch: bool = True, sampler=None,
                 run_dir='', run_name='',
                 checkpoint=None, num_epochs=0, eval_per_epochs=0, patience=0,
                 minimum_epochs=0, models_to_save=[], clip_grad=None, log_iterations=0, lr=0.0001,
                 weight_decay=0.0001, lr_scheduler=None, factor=0, min_lr=0, mode='max', lr_scheduler_patience=0,
                 lr_verbose=True, val_per_batch=True, std=1):

        self.device = device
        self.std = std # stdev of data. to adjust val scores.
        self.model = model.to(self.device)
        self.loss_func = loss_func
        self.metrics = metrics
        self.sampler = sampler
        self.val_per_batch = val_per_batch
        self.main_metric = type(self.loss_func).__name__ if main_metric == 'loss' else main_metric
        self.main_metric_goal = main_metric_goal
        self.scheduler_step_per_batch = scheduler_step_per_batch
        self.checkpoint = checkpoint
        self.num_epochs = num_epochs
        self.eval_per_epochs = eval_per_epochs
        self.patience = patience
        self.minimum_epochs = minimum_epochs
        self.models_to_save = models_to_save
        self.clip_grad = clip_grad
        self.log_iterations = log_iterations
        self.lr = lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.min_lr = min_lr
        self.mode = mode
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_verbose = lr_verbose
        self.optim = optim(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.val_loss_for_wandb = None
        self.val_score_for_wandb = None

        if lr_scheduler:  # Needs "from torch.optim.lr_scheduler import *" to work
            self.lr_scheduler = lr_scheduler(self.optim, mode=mode, factor=factor, patience=lr_scheduler_patience,
                                            min_lr=min_lr, verbose=lr_verbose)
        else:
            self.lr_scheduler = None

        if self.checkpoint:
            check = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(check['model_state_dict'])
            self.optim.load_state_dict(check['optimizer_state_dict'])
            if self.lr_scheduler != None and check['scheduler_state_dict'] != None:
                self.lr_scheduler.load_state_dict(check['scheduler_state_dict'])
            self.start_epoch = check['epoch']
            self.best_val_score = check['best_val_score']
            self.optim_steps = check['optim_steps']
            self.log_dir = os.path.dirname(self.checkpoint)
        else:
            # not sure this is needed
            self.start_epoch = 1
            self.optim_steps = 0
            self.best_val_score = -np.inf if self.main_metric_goal == 'max' else np.inf  # running score to decide whether or not a new model should be saved
            self.log_dir = run_dir
        self.run_name = run_name

        #for i, param_group in enumerate(self.optim.param_groups):
        #    param_group['lr'] = 0.0003
        self.epoch = self.start_epoch
        print(f'Log directory: {self.log_dir}')
        self.hparams = {'checkpoint':checkpoint, 'num epochs':num_epochs,
                        'eval_per_epochs':eval_per_epochs, 'patience':patience,
                        'minimum_epochs':minimum_epochs, 'models_to_save':models_to_save,
                        'clip_grad':clip_grad, 'log_iterations':log_iterations,
                        'lr':lr, 'weight decay':weight_decay, 'lr scheduler':lr_scheduler,
                        'factor':factor, 'min_lr':min_lr, 'mode':mode,
                        'lr_scheduler_patience':lr_scheduler_patience, 'lr_verbose':lr_verbose}
        for key, value in self.hparams.items():
            print(f'{key}: {value}')

    def run_per_epoch_evaluations(self, loader):
        pass

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        epochs_no_improve = 0  # counts every epoch that the validation accuracy did not improve for early stopping
        for epoch in range(self.start_epoch, self.num_epochs + 1):  # loop over the dataset multiple times

            self.epoch = epoch
            self.model.train()
            self.predict(train_loader, optim=self.optim)
            self.model.eval()

            with torch.no_grad():

                metrics, _, _ = self.predict(val_loader)
                # MAE of prediction is * std of data since data is normalised
                val_score = metrics[self.main_metric] * self.std

                if self.lr_scheduler!=None and not self.scheduler_step_per_batch:
                    self.step_schedulers(metrics=val_score)

                if self.eval_per_epochs > 0 and epoch % self.eval_per_epochs == 0:
                    self.run_per_epoch_evaluations(val_loader)

                # val loss is MSE, shouldn't be affected by data normalisation
                val_loss = metrics[type(self.loss_func).__name__]
                wandb.log({"val_loss": val_loss, "val_score": val_score, "epoch": self.epoch})
                print(f'[Epoch {epoch}] {self.main_metric}: {val_score:.6f} val loss: {val_loss:.6f}')
                self.val_loss_for_wandb = val_loss
                self.val_score_for_wandb = val_score

                # save the model with the best main_metric depending on wether we want to maximize or minimize the main metric
                if val_score >= self.best_val_score and self.main_metric_goal == 'max' or val_score <= self.best_val_score and self.main_metric_goal == 'min':
                    epochs_no_improve = 0
                    self.best_val_score = val_score
                    self.save_checkpoint(epoch, checkpoint_name=f'{self.run_name}.best_checkpoint.pt')
                else:
                    epochs_no_improve += 1
                self.save_checkpoint(epoch, checkpoint_name=f'{self.run_name}.last_checkpoint.pt')
                print(f'Epochs with no improvement: [ {epochs_no_improve} ] and the best {self.main_metric} was in {epoch - epochs_no_improve}')
                if epochs_no_improve >= self.patience and epoch >= self.minimum_epochs:  # stopping criterion
                    print(f'Early stopping criterion based on -{self.main_metric}- that should be {self.main_metric_goal}-imized reached after {epoch} epochs. Best model checkpoint was in epoch {epoch - epochs_no_improve}.')
                    break
                if epoch in self.models_to_save:
                    shutil.copyfile(os.path.join(self.log_dir, f'{self.run_name}.best_checkpoint.pt'),
                                    os.path.join(self.log_dir, f'{self.run_name}.best_checkpoint_{epoch}epochs.pt'))
                self.after_epoch()
                #if val_loss > 10000:
                #    raise Exception

        # evaluate on best checkpoint
        checkpoint = torch.load(os.path.join(self.log_dir, f'{self.run_name}.best_checkpoint.pt'), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.evaluation(val_loader, data_split='val_best_checkpoint')

    def forward_pass(self, batch):
        pass

    def process_batch(self, batch, optim):
        loss, predictions, targets = self.forward_pass(batch)

        if optim != None:  # run backpropagation if an optimizer is provided
            loss.backward()
            if self.clip_grad != None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad, norm_type=2)
            self.optim.step()

            self.after_optim_step()
            self.optim.zero_grad()
            self.optim_steps += 1
        return loss, list_detach(predictions), list_detach(targets)

    def predict(self, data_loader: DataLoader, optim: torch.optim.Optimizer = None, return_pred=False):
        total_metrics = {k: 0 for k in
                         list(self.metrics.keys()) + [type(self.loss_func).__name__, 'mean_pred', 'std_pred',
                                                      'mean_targets', 'std_targets']}
        epoch_targets = []
        epoch_predictions = []
        epoch_loss = 0
        for i, batch in enumerate(data_loader):
            *batch, batch_indices = move_to_device(list(batch), self.device)
            loss, predictions, targets = self.process_batch(batch, optim)
            with torch.no_grad():
                if (self.optim_steps % self.log_iterations == 0 or i+1==len(data_loader)) and optim != None:
                    metrics = self.evaluate_metrics(predictions, targets)
                    metrics[type(self.loss_func).__name__] = loss.item()
                    if self.val_score_for_wandb is None:
                        wandb.log({"train loss": loss.item(), "epoch": self.epoch})
                    else:
                        wandb.log({"train loss": loss.item(), "epoch": self.epoch, "val_loss": self.val_loss_for_wandb, "val_score": self.val_score_for_wandb})
                    print(f'[Epoch {self.epoch}; Iter {i+1:5d}/{len(data_loader):5d}] train: loss: {loss.item():.7f}')
                if optim == None and self.val_per_batch:  # during validation or testing when we want to average metrics over all the data in that dataloader
                    metrics = self.evaluate_metrics(predictions, targets, val=True)
                    metrics[type(self.loss_func).__name__] = loss.item()
                    for key, value in metrics.items():
                        total_metrics[key] += value
                if optim == None and not self.val_per_batch or return_pred:
                    epoch_loss += loss.item()
                    epoch_targets.extend(targets if isinstance(targets, list) else [targets])
                    epoch_predictions.extend(predictions if isinstance(predictions, list) else [predictions])
                self.after_batch(predictions, targets, batch_indices)
        if optim == None:
            loader_len = len(data_loader) if len(data_loader) != 0 else 1
            if self.val_per_batch:
                total_metrics = {k: v / loader_len for k, v in total_metrics.items()}
            else:
                total_metrics = self.evaluate_metrics(epoch_predictions, epoch_targets, val=True)
                total_metrics[type(self.loss_func).__name__] = epoch_loss / loader_len
            if return_pred:
                return total_metrics, list_detach(epoch_predictions), list_detach(epoch_targets)
            else:
                return total_metrics, None, None

    def after_batch(self, predictions, targets, batch_indices):
        pass

    def after_epoch(self):
        pass

    def after_optim_step(self):
        # step per batch if that is what we want to do or if we are using a warmup schedule and are still in the warmup period
        we_want = self.scheduler_step_per_batch
        warmup  = isinstance(self.lr_scheduler, WarmUpWrapper) and self.lr_scheduler.total_warmup_steps > self.lr_scheduler._step
        if self.lr_scheduler != None and (we_want or warmup):
            self.step_schedulers()

    def evaluate_metrics(self, predictions, targets, batch=None, val=False) -> Dict[str, float]:
        metrics = {}
        metrics[f'mean_pred'] = torch.mean(concat_if_list(predictions)).item()
        metrics[f'std_pred'] = torch.std(concat_if_list(predictions)).item()
        metrics[f'mean_targets'] = torch.mean(concat_if_list(targets)).item()
        metrics[f'std_targets'] = torch.std(concat_if_list(targets)).item()
        for key, metric in self.metrics.items():
            if not hasattr(metric, 'val_only') or val:
                metrics[key] = metric(predictions, targets).item()
        return metrics

    def evaluation(self, data_loader: DataLoader, data_split: str = '', return_pred=False):
        self.model.eval()
        metrics, predictions, targets = self.predict(data_loader, return_pred=return_pred)

        with open(os.path.join(self.log_dir, f'{self.run_name}.evaluation_{data_split}.txt'), 'w') as f:
            print(f'Statistics on {data_split}')
            for key, value in metrics.items():
                if key == 'mae':
                    value *= self.std
                f.write(f'{key}: {value}\n')
                print(f'{key}: {value}')
        #TODO right now only MAE has the correct units
        return metrics, predictions, targets

    def step_schedulers(self, metrics=None):
        try:
            self.lr_scheduler.step(metrics=metrics)
        except:
            self.lr_scheduler.step()


    def save_checkpoint(self, epoch: int, checkpoint_name: str):
        """
        Saves checkpoint of model in the logdir of the summarywriter in the used rundi
        """
        run_dir = self.log_dir
        self.save_model_state(epoch, checkpoint_name)
        train_args = copy.copy(self.hparams)
        for key in train_args:
            if inspect.isclass(train_args[key]):
                train_args[key] = train_args[key].__name__
        with open(os.path.join(run_dir, f'{self.run_name}.train_arguments.yaml'), 'w') as yaml_path:
            pyaml.dump(train_args, yaml_path)


    def save_model_state(self, epoch: int, checkpoint_name: str):
        torch.save({
            'epoch': epoch,
            'best_val_score': self.best_val_score,
            'optim_steps': self.optim_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': None if self.lr_scheduler == None else self.lr_scheduler.state_dict()
        }, os.path.join(self.log_dir, checkpoint_name))
