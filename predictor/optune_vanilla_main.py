'''
Author: Radillus
Date: 2023-06-05 22:50:34
LastEditors: Radillus
LastEditTime: 2023-06-11 10:44:39
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import pickle
import time
import argparse

from tqdm import tqdm
import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn

from predictor.optuna_vanilla_net_build import generate_model
from dataset.size_mix_data import SizeMixTrainDataset, SizeMixTestDataset, SizeMixEvalDataset
from tools.timeout import timeout


IMAGE_SIZE_MAX = 300

parser = argparse.ArgumentParser()
parser.add_argument('--tqdm', type=int, default=1)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda:0')
parser_args = parser.parse_args()

enable_tqdm = parser_args.tqdm
device = torch.device(parser_args.device)
if device.type == 'cuda':
    cudnn.benchmark = True
    cudnn.enabled = True
epoch_max = parser_args.epoch

def max_loss(x, y):
    return torch.max(torch.abs(x - y))

@timeout(10)
def batch_train(batch, model:nn.Module, optimizer:torch.optim.Optimizer, loss_beta:float, grad_clip:float):
    optimizer.zero_grad()
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    y_ = model(x)
    loss = loss_beta * F.l1_loss(y_, y) + (1-loss_beta) * max_loss(y_, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

def net_search(trail:Trial):
    # torch._dynamo.reset()
    loss_beta = trail.suggest_float('loss_beta', 0.0, 1.0)
    grad_clip = trail.suggest_float('grad_clip', 0.1, 100.0, log=True)
    image_size = trail.suggest_int('image_size', 80, IMAGE_SIZE_MAX)
    epochs = trail.suggest_int('epochs', 1, epoch_max)
    # epochs = 1

    model = generate_model(trail).to(device)

    # max_loss = torch.compile(max_loss, mode='reduce-overhead')
    # l1_loss = torch.compile(F.l1_loss, mode='reduce-overhead')
    l1_loss = F.l1_loss
    # model = torch.compile(model)
    # model = torch.compile(model, mode='reduce-overhead')
    # model = torch.compile(model, mode='max-autotune')
    optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=trail.suggest_float('lr', 1e-5, 1, log=True),
                betas=(trail.suggest_float('beta1', 0.5, 0.999), trail.suggest_float('beta2', 0.5, 0.999)),
                weight_decay=trail.suggest_float('weight_decay', 1e-5, 5e-1, log=True),
                )
    
    train_dataset = SizeMixTrainDataset(to_size = image_size)
    eval_dataset = SizeMixEvalDataset(to_size = image_size)
    test_dataset = SizeMixTestDataset(to_size = image_size)
    
    train_dataloader = DataLoader(train_dataset,num_workers=4)
    eval_dataloader = DataLoader(eval_dataset,num_workers=4)
    test_dataloader = DataLoader(test_dataset,num_workers=4)
    
    train_len = len(train_dataset)
    eval_len = len(eval_dataset)
    test_len = len(test_dataset)
    
    for epoch in tqdm(range(epochs),desc='Epoch',total=epochs,leave=False) if enable_tqdm else range(epochs):
        # train
        model.train()
        try:
            for batch in train_dataloader:
                batch_train(batch, model, optimizer, loss_beta, grad_clip)
                break
        except TimeoutError:
            trail.report(torch.inf, epoch)
            raise optuna.exceptions.TrialPruned()
        for batch in tqdm(train_dataloader,desc='Train',total=train_len,leave=False) if enable_tqdm else train_dataloader:
            x, y = batch
            x, y = x.float().to(device), y.float().to(device)
            y_ = model(x)
            if torch.isnan(y_).any() or torch.isinf(y_).any():
                trail.report(torch.inf, epoch)
                raise optuna.exceptions.TrialPruned()
            loss = loss_beta * l1_loss(y, y_) + (1 - loss_beta) * max_loss(y, y_)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # eval
        with torch.no_grad():
            model.eval()
            eval_loss = -torch.inf
            for batch in tqdm(eval_dataloader,desc='Eval',total=eval_len,leave=False) if enable_tqdm else eval_dataloader:
                x, y = batch
                x, y = x.float().to(device), y.float().to(device)
                y_ = model(x)
                if torch.isnan(y_).any() or torch.isinf(y_).any():
                    trail.report(torch.inf, epoch)
                    raise optuna.exceptions.TrialPruned()
                loss = max_loss(y, y_)
                eval_loss = max(eval_loss, loss)
            trail.report(eval_loss, epoch)
            if trail.should_prune():
                raise optuna.exceptions.TrialPruned()

    # # test
    # with torch.no_grad():
    #     model.eval()
    #     test_loss = -torch.inf
    #     for batch in tqdm(test_dataloader,desc='Test',total=test_len,leave=False) if enable_tqdm else test_dataloader:
    #         x, y = batch
    #         x, y = x.float().to(device), y.float().to(device)
    #         y_ = model(x)
    #         loss = max_loss(y, y_)
    #         test_loss = max(test_loss, loss)
    return eval_loss


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.HyperbandPruner())
    study.enqueue_trial({'loss_beta': 0.7761841152216598, 'grad_clip': 0.13675857470892633, 'image_size': 105, 'epochs': 1, 'layer_num': 9, 'l0': 'batch_normalization', 'l1': 'matadd', 'l1_matadd_hs': 302, 'l2': 'matdot', 'l2_matdot_hs': 29, 'l3': 'matadd', 'l3_matadd_hs': 41, 'l4': 'batch_normalization', 'l5': 'tfno', 'l5_tfno_oc': 1, 'l5_tfno_m': 30, 'l5_tfno_hc': 1, 'l5_tfno_lc': 7, 'l5_tfno_pc': 3, 'l5_tfno_nl': 10, 'l5_tfno_r': 0.8740925015093649, 'l5_tfno_s': 'linear', 'l5_tfno_f': 'tucker', 'l5_tfno_i': 'reconstructed', 'l5_af': 'Softsign', 'l6': 'scaling_maxpool', 'l6_maxpool_ks': 5, 'l6_maxpool_st': 3, 'l7': 'convolution', 'l7_conv_oc': 7, 'l7_conv_ks': 26, 'l8': 'source', 'l8_source_oc': 10, 'l8_source_hs': 482, 'last': 'tfno', 'last_tfno_m': 30, 'last_tfno_hc': 4, 'last_tfno_lc': 74, 'last_tfno_pc': 87, 'last_tfno_nl': 5, 'last_tfno_r': 0.13814114000923994, 'last_tfno_s': 'identity', 'last_tfno_f': 'tt', 'last_tfno_i': 'factorized', 'last_af': 'None', 'lr': 0.00040193519484577663, 'beta1': 0.5772246276954849, 'beta2': 0.7159714297834489, 'weight_decay': 2.5314001880163968e-05})
    study.enqueue_trial({'loss_beta': 0.9947342504176155, 'grad_clip': 0.7275699136535639, 'image_size': 293, 'epochs': 1, 'layer_num': 1, 'l0': 'add', 'last': 'convolution', 'last_conv_ks': 1, 'lr': 0.008691473127565522, 'beta1': 0.6041995718761084, 'beta2': 0.7715110529430346, 'weight_decay': 0.45127162842756013})
    study.enqueue_trial( {'loss_beta': 0.7574289550465264, 'grad_clip': 0.1633879974841954, 'image_size': 107, 'epochs': 2, 'layer_num': 9, 'l0': 'scaling_maxpool', 'l0_maxpool_ks': 10, 'l0_maxpool_st': 10, 'l1': 'matmul', 'l1_matmul_hs': 149, 'l2': 'scaling_maxpool', 'l2_maxpool_ks': 3, 'l2_maxpool_st': 3, 'l3': 'batch_normalization', 'l4': 'matmul', 'l4_matmul_hs': 161, 'l5': 'matdot', 'l5_matdot_hs': 444, 'l6': 'tfno', 'l6_tfno_oc': 1, 'l6_tfno_m': 28, 'l6_tfno_hc': 20, 'l6_tfno_lc': 71, 'l6_tfno_pc': 6, 'l6_tfno_nl': 10, 'l6_tfno_r': 0.11633462458159977, 'l6_tfno_s': 'soft-gating', 'l6_tfno_f': 'cp', 'l6_tfno_i': 'reconstructed', 'l6_af': 'GELU', 'l7': 'batch_normalization', 'l8': 'batch_normalization', 'last': 'tfno', 'last_tfno_m': 28, 'last_tfno_hc': 3, 'last_tfno_lc': 79, 'last_tfno_pc': 88, 'last_tfno_nl': 2, 'last_tfno_r': 0.17928475098780228, 'last_tfno_s': 'soft-gating', 'last_tfno_f': 'tucker', 'last_tfno_i': 'factorized', 'last_af': 'Hardtanh', 'lr': 3.797487592886997e-05, 'beta1': 0.59245067918063, 'beta2': 0.6584997495103803, 'weight_decay': 1.8757654040095693e-05})
    study.enqueue_trial({'loss_beta': 0.7317290391236149, 'grad_clip': 0.16004722798884186, 'image_size': 99, 'epochs': 1, 'layer_num': 11, 'l0': 'convolution', 'l0_conv_oc': 8, 'l0_conv_ks': 1, 'l1': 'batch_normalization', 'l2': 'activate', 'l2_af': 'GELU', 'l3': 'matadd', 'l3_matadd_hs': 491, 'l4': 'batch_normalization', 'l5': 'tfno', 'l5_tfno_oc': 2, 'l5_tfno_m': 27, 'l5_tfno_hc': 14, 'l5_tfno_lc': 73, 'l5_tfno_pc': 58, 'l5_tfno_nl': 6, 'l5_tfno_r': 0.5837661200990324, 'l5_tfno_s': 'linear', 'l5_tfno_f': 'tt', 'l5_tfno_i': 'reconstructed', 'l5_af': 'ReLU', 'l6': 'batch_normalization', 'l7': 'convolution', 'l7_conv_oc': 8, 'l7_conv_ks': 11, 'l8': 'matdot', 'l8_matdot_hs': 413, 'l9': 'batch_normalization', 'l10': 'batch_normalization', 'last': 'tfno', 'last_tfno_m': 30, 'last_tfno_hc': 7, 'last_tfno_lc': 86, 'last_tfno_pc': 63, 'last_tfno_nl': 6, 'last_tfno_r': 0.12083149250830966, 'last_tfno_s': 'identity', 'last_tfno_f': 'tt', 'last_tfno_i': 'factorized', 'last_af': 'None', 'lr': 0.002086238676175717, 'beta1': 0.5015745162420369, 'beta2': 0.8015678918008728, 'weight_decay': 6.317914991000814e-05})
    study.optimize(net_search, n_trials=1000, gc_after_trial=True)
    print('best_params',study.best_params)
    print('best_value',study.best_value)
    print('best_trial',study.best_trial)

    with open(f'./data/optuna/study_'+time.strftime("%m-%d-%H-%M", time.localtime()), 'wb') as study_file:
        pickle.dump(study, study_file)
