import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import opASL
    
class ASL(Function):
    @staticmethod
    def forward(ctx, input, theta):
        ctx.save_for_backward(input, theta)
        return opASL.forward(input, theta)

    @staticmethod
    def backward(ctx, grad_output):
        input, theta = ctx.saved_tensors
        return opASL.backward(grad_output, theta, input)

class ActiveShift2d(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super(ActiveShift2d, self).__init__()
        self.theta = nn.Parameter(torch.Tensor(in_channel,2))
        self.theta.data.uniform_(-1, 1)

    def forward(self, x):
        return ASL.apply(input, self.theta)