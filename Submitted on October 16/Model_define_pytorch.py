#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
If you have any questions, please contact me with https://github.com/xufana7/AutoEncoder-with-pytorch
Author, Fan xu Aug 2020
"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# This part implement the quantization and dequantization operations.
# The output of the encoder must be the bitstream.
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its four bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2)
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for four time.
        grad_bit = grad_output.repeat_interleave(ctx.constant, dim=1)
        return grad_bit, None
        #b, c = grad_output.shape
        #grad_output = grad_output.unsqueeze(2) / ctx.constant
        #grad_bit = grad_output.expand(b, c, ctx.constant)
        #return torch.reshape(grad_bit, (-1, c * ctx.constant)), None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)



class Encoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits):
        super(Encoder, self).__init__()

        ###(1) 更改特征图数量与层数

        self.conv1 = conv3x3(2, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = conv3x3(32, 128)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = conv3x3(128, 128)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = conv3x3(128, 256)
        self.bn5 = nn.BatchNorm2d(256) 
        self.conv6 = conv3x3(256, 256)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = conv3x3(256, 256)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = conv3x3(256, 128)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = conv3x3(128, 128)
        self.bn9 = nn.BatchNorm2d(128)
        self.conv10 = conv3x3(128, 32)
        self.bn10 = nn.BatchNorm2d(32)
        self.conv11 = conv3x3(32, 2)
        self.bn11 = nn.BatchNorm2d(2)

        self.fc = nn.Linear(1024, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)

    def forward(self, x):

        ###(1) 更改特征图数量与层数
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        out = self.bn3(self.conv3(out))
        out = F.relu(out)
        out = self.bn4(self.conv4(out))
        out = F.relu(out)
        out = self.bn5(self.conv5(out))
        out = F.relu(out)
        out = self.bn6(self.conv6(out))
        out = F.relu(out)
        out = self.bn7(self.conv7(out))
        out = F.relu(out)
        out = self.bn8(self.conv8(out))
        out = F.relu(out)
        out = self.bn9(self.conv9(out))
        out = F.relu(out)
        out = self.bn10(self.conv10(out))
        out = F.relu(out)
        out = self.bn11(self.conv11(out))
        out = F.relu(out)
        
        out = out.view(-1, 1024)
        out = self.fc(out)
        out = self.sig(out)
        out = self.quantize(out)

        return out

class Decoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        self.multiConvs = nn.ModuleList()
        self.fc = nn.Linear(int(feedback_bits / self.B), 1024)
        self.out_cov = conv3x3(2, 2)
        self.sig = nn.Sigmoid()

        for _ in range(3):
            self.multiConvs.append(nn.Sequential(
                conv3x3(2, 16),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                
                conv3x3(16, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                conv3x3(128, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                conv3x3(128, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                conv3x3(128, 256),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                conv3x3(256, 256),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                conv3x3(256, 256),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                conv3x3(256, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                conv3x3(128, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                conv3x3(128, 32),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                
                conv3x3(32, 2),
                nn.BatchNorm2d(2),
                nn.ReLU()))

    def forward(self, x):
        out = self.dequantize(x)
        out = out.view(-1, int(self.feedback_bits / self.B))
        out = self.sig(self.fc(out))
        out = out.view(-1, 2, 16, 32)
        for i in range(3):
            residual = out
            out = self.multiConvs[i](out)
            out = residual + out

        out = self.out_cov(out)
        out = self.sig(out)
        return out


# Note: Do not modify following class and keep it in your submission.
# feedback_bits is 128 by default.
class AutoEncoder(nn.Module):

    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

def NMSE_cuda(x, x_hat):
    x_real = x[:, 0, :, :].view(len(x),-1) - 0.5
    x_imag = x[:, 1, :, :].view(len(x),-1) - 0.5
    x_hat_real = x_hat[:, 0, :, :].view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, 1, :, :].view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real**2 + x_imag**2, axis=1)
    mse = torch.sum((x_real-x_hat_real)**2 + (x_imag-x_hat_imag)**2, axis=1)
    nmse = mse/power
    return nmse
    
class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction
    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse) 
        else:
            nmse = torch.sum(nmse)
        return nmse

def Score(NMSE):
    score = 1 - NMSE
    return score


# dataLoader
class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __getitem__(self, index):
        return self.matdata[index]

    def __len__(self):
        return self.matdata.shape[0]
