'''
Author: Radillus
Date: 2023-05-26 14:29:48
LastEditors: Radillus
LastEditTime: 2023-07-13 00:37:25
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import torch.nn as nn



ACTIVATE_FUNCTION_LIST = [
    'Identity',
    'ELU',
    'Hardshrink',
    'Hardsigmoid',
    'Hardtanh',
    'Hardwish',
    'LeakyReLU',
    'LogSigmoid',
    'PReLU',
    'ReLU',
    'RReLU',
    'SELU',
    'CELU',
    'GELU',
    'Sigmoid',
    'SiLU',
    'Softplus',
    'Softshrink',
    'Softsign',
    'Tanh',
    'Tanhshrik',
]
def get_activate_function(activate_function: str, inplace: bool = True):
    match activate_function:
        case 'Identity':
            return nn.Identity()
        case 'ELU':
            return nn.ELU(inplace=inplace)
        case 'Hardshrink':
            return nn.Hardshrink()
        case 'Hardsigmoid':
            return nn.Hardsigmoid(inplace=inplace)
        case 'Hardtanh':
            return nn.Hardtanh(inplace=inplace)
        case 'Hardwish':
            return nn.Hardswish(inplace=inplace)
        case 'LeakyReLU':
            return nn.LeakyReLU(inplace=inplace)
        case 'LogSigmoid':
            return nn.LogSigmoid()
        case 'PReLU':
            return nn.PReLU()
        case 'ReLU':
            return nn.ReLU(inplace=inplace)
        case 'RReLU':
            return nn.RReLU(inplace=inplace)
        case 'SELU':
            return nn.SELU(inplace=inplace)
        case 'CELU':
            return nn.CELU(inplace=inplace)
        case 'GELU':
            return nn.GELU('tanh')
        case 'Sigmoid':
            return nn.Sigmoid()
        case 'SiLU':
            return nn.SiLU(inplace=inplace)
        case 'Softplus':
            return nn.Softplus()
        case 'Softshrink':
            return nn.Softshrink()
        case 'Softsign':
            return nn.Softsign()
        case 'Tanh':
            return nn.Tanh()
        case 'Tanhshrik':
            return nn.Tanhshrink()
        case _:
            raise ValueError(f'Unknown activate function: {activate_function}')