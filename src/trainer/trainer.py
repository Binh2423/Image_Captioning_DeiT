import torch
import pandas as pd
import time
import os
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Metric
from torch import Tensor
from typing import Optional,Any
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict
from .history import History
from .batch_results import BatchResults

class Trainer:

    def __init__(self,
        model : Optional[Module] = None,
        criteron : Optional[Module] = None,
        optimizer : Optional[Optimizer] = None,
        scheduler : Optional[LRScheduler] = None,
        metrics : Optional[dict[str, Metric]] = None,
        score_metric : Optional[str] = None,
        device : Optional[torch.device] = None,
        weights_folder : Optional[str] = None,
        save_weights_every : Optional[int] = None,
    ) -> None:
        
        self.model = model
        self.criteron = criteron
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics if metrics is not None else {}
        self.score_metric = score_metric
        self.device = device
        self.weights_folder = weights_folder
        self.save_weights_every = save_weights_every
        self.history = self._init_history()

        self.best_weights = None
        self.best_epoch = None
        self.last_best_score = None

    def _init_history(self) -> History:

        return History(
            splits=['train', 'val'], 
            metrics=['epoch','time','loss'] + (list(self.metrics.keys()) if self.metrics is not None else [])
        )

    def set_model(self, model : Module):
        self.model = model
        return self
    
    def set_criteron(self, criteron : Module):
        self.criteron = criteron
        return self
    
    def set_optimizer(self, optimizer : Optimizer):
        self.optimizer = optimizer
        return self
    
    def set_scheduler(self, scheduler : Optional[LRScheduler]):
        self.scheduler = scheduler
        return self
    
    def set_score_metric(self, name : str):

        if name not in self.metrics.keys():
            raise Exception(f"name must be one of : {','.join(list(self.metrics.keys()))},but found : {name} instead.")
        
        self.score_metric = name

        return self
    
    def set_device(self, device : torch.device):
        self.device = device
        return self
    
    def set_weights_folder(self, weights_folder : str):
        self.weights_folder = weights_folder
        return self
    
    def set_save_weights_every(self, save_weights_every : int):
        self.save_weights_every = save_weights_every
        return self
    
    def add_metric(self, name : str, metric : Metric):
        self.metrics[name] = metric
        self.history = self._init_history()
        return self
    
    def _format(self, dict_ : dict, prefix : str = '') -> str:

        r = []

        for key,value in dict_.items():
            r.append(f"{prefix}{key} = {value}")

        return ','.join(r)
    
    def train_on_batch(self, batch : tuple[Any, Any], results : BatchResults) -> tuple[dict, Tensor]:

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        ### Forward Step
        y_hat = self.model(x)

        ### Calculate the loss
        loss = self.criteron(y_hat, y)

        ### Zero the gradints
        self.optimizer.zero_grad()

        ### Backward
        loss.backward()

        ### Update the weights
        self.optimizer.step()

        ### Update the learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        ### Update the metrics
        step_results = results.update(y_hat, y)

        return step_results,loss
    
    def eval_on_batch(self, batch : tuple[Any, Any], results : BatchResults) -> tuple[dict, Tensor]:

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        ### Forward Step
        y_hat = self.model(x)

        ### Calculate the loss
        loss = self.criteron(y_hat, y)

        ### Update the metrics
        step_results = results.update(y_hat, y)

        return step_results,loss
    
    def train_on_loader(self, loader : DataLoader) -> tuple[dict[str, float],float,float]:

        self.model.train()

        results = BatchResults(self.metrics)
        running_loss = 0.0

        t_object = tqdm(loader)

        tic = time.time()

        for batch in t_object:

            # Call train_on_batch and handle different return types
            batch_output = self.train_on_batch(batch, results)
            
            # Handle different return types from train_on_batch
            if isinstance(batch_output, tuple):
                # Expected: (step_results, loss) or (step_results, loss, ...)
                if len(batch_output) >= 2:
                    step_results, loss = batch_output[0], batch_output[1]
                else:
                    # Single element tuple? Treat as loss
                    step_results = {}
                    loss = batch_output[0]
            elif isinstance(batch_output, dict):
                # Dict with 'loss' and/or 'step_results' keys
                step_results = batch_output.get('step_results', {})
                loss = batch_output.get('loss', batch_output.get('logits', None))
                if loss is None:
                    raise ValueError("train_on_batch returned dict without 'loss' or 'logits' key")
            elif torch.is_tensor(batch_output):
                # Just a tensor (logits) - compute loss ourselves
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                loss = self.criteron(batch_output, y)
                step_results = results.update(batch_output, y)
            else:
                raise TypeError(f"Unexpected return type from train_on_batch: {type(batch_output)}")
            
            # Convert loss to float safely
            if torch.is_tensor(loss):
                loss_value = loss.item()
            elif isinstance(loss, (int, float)):
                loss_value = float(loss)
            else:
                raise TypeError(f"Loss must be a tensor or number, got {type(loss)}")
            
            running_loss += loss_value

            lr = f"{self.scheduler.get_last_lr()[0]}" if self.scheduler is not None else ""
            dsc = self._format(step_results) + f",loss={loss_value},lr={lr}" 
            t_object.set_description(dsc)

        running_loss /= len(loader)

        toc = time.time()

        return results.compute(),running_loss,(toc - tic)
    
    def eval_on_loader(self, loader : DataLoader) -> tuple[dict[str, float],float,float]:

        self.model.eval()

        results = BatchResults(self.metrics)
        running_loss = 0.0

        tic = time.time()

        with torch.inference_mode():

            for batch in tqdm(loader):

                _,loss = self.eval_on_batch(batch, results)
                running_loss += loss.item()

        running_loss /= len(loader)

        toc = time.time()

        return results.compute(),running_loss,(toc - tic)
    
    def train(self, train_dataloader : DataLoader,val_dataloader : DataLoader, epochs : int) -> None:

        for epoch in range(epochs):

            train_results, train_loss, training_time = self.train_on_loader(train_dataloader)
            train_results['loss'] = train_loss

            print()

            val_results, val_loss, validation_time = self.eval_on_loader(val_dataloader)

            val_results['loss'] = val_loss

            msg = f"\n{self._format(train_results,prefix='train_')}"
            msg += f",{self._format(val_results,prefix='val_')}\n"

            print(msg)

            train_results['time'] = training_time
            val_results['time'] = validation_time

            train_results['epoch'] = epoch
            val_results['epoch'] = epoch

            self.history.update(train_results, 'train')
            self.history.update(val_results, 'val')

            if self.scheduler is not None and (self.last_best_score is None or val_results[self.score_metric] > self.last_best_score):

                self.best_epoch = epoch + 1
                self.last_best_score = val_results[self.score_metric]
                self.best_weights = OrderedDict()

                for name,weight in self.model.state_dict().items():
                    self.best_weights[name] = weight.cpu()

            if (self.save_weights_every is not None) and (epoch + 1) % self.save_weights_every == 0:
                path = os.path.join(self.weights_folder, f"{time.time()}_epoch={epoch+1}.pt")
                print(f"\nSaving weights to : {path}\n")
                torch.save(self.model.state_dict(), path)