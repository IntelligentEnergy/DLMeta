'''
Author: Radillus
Date: 2023-05-31 17:35:59
LastEditors: Radillus
LastEditTime: 2023-06-24 20:05:54
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import time
import torch


def logging_time_stamp() -> str:
    return time.strftime("[%m-%d %H:%M:%S] ", time.localtime())

def no_autograd(net: torch.nn.Module):
    """Disable autograd for a network."""
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

def caculate_convolution_output_size(
    input_size: int,
    kernel_size: int,
    stride: int=1,
    padding: int=0,
) -> int:
    return int((input_size + 2 * padding - kernel_size) / stride) + 1
