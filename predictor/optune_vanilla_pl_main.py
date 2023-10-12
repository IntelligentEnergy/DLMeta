'''
Author: Radillus
Date: 2023-06-05 22:50:34
LastEditors: Radillus
LastEditTime: 2023-06-09 14:15:24
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import pickle
import time
from typing import Any

import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from optuna_vanilla_net_build import generate_model
from size_mix_data import SizeMixTrainDataset, SizeMixTestDataset, SizeMixEvalDataset


def net_search(trail:Trial):
    
    loss_beta = trail.suggest_float('loss_beta', 0.1, 0.9)
    
    class PLModel(pl.LightningModule):
        def __init__(self, trail:Trial):
            super().__init__()
            self.model = generate_model(trail)
            self.trail = trail
        
        def configure_optimizers(self) -> Any:
            return torch.optim.AdamW(
                self.parameters(),
                lr=trail.suggest_float('lr', 1e-5, 1e-1, log=True),
                betas=(trail.suggest_float('beta1', 0.5, 0.999), trail.suggest_float('beta2', 0.5, 0.999)),
                weight_decay=trail.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
            )

        def training_step(self, batch:torch.Tensor, batch_idx:int):
            x, y = batch
            y_ = self.model(x)
            diff = torch.abs(y_ - y)
            loss = loss_beta * torch.mean(diff) + (1 - loss_beta) * torch.max(diff)
            self.log('train_loss', loss)
            return loss
        
        def validation_step(self, batch:torch.Tensor, batch_idx:int):
            x, y = batch
            y_ = self.model(x)
            diff = torch.abs(y_ - y)
            loss = loss_beta * torch.mean(diff) + (1 - loss_beta) * torch.max(diff)

            self.log('val_loss', loss)
            return loss
            
        def test_step(self, batch:torch.Tensor, batch_idx:int):
            x,y = batch
            y_ = self.model(x)
            loss = torch.max(torch.abs(y_ - y)).item()
            self.log('test_loss', loss)
            self.trail.report(loss, self.global_step)
            if self.trail.should_prune() and self.global_rank == 0:
                raise optuna.exceptions.TrialPruned()
            return loss
    
    train_dataset = SizeMixTrainDataset()
    eval_dataset = SizeMixEvalDataset()
    test_dataset = SizeMixTestDataset()
    
    train_dataloader = DataLoader(train_dataset)
    eval_dataloader = DataLoader(eval_dataset)
    test_dataloader = DataLoader(test_dataset)
    
    model = PLModel(trail)
    
    trainer = pl.Trainer(
        fast_dev_run=True,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')],
    )
    trainer.fit(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders=eval_dataloader,
    )
    test_loss = trainer.test(
        model=model,
        dataloaders=test_dataloader,
    )
    return test_loss[0]['test_loss']

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.HyperbandPruner())
    study.enqueue_trial({'l0':'source'})
    study.optimize(net_search, n_trials=100)

    with open(f'./data/optuna/study_'+time.strftime("%m-%d-%H-%M", time.localtime()), 'wb') as study_file:
        pickle.dump(study, study_file)
