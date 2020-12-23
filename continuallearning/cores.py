"""Test module defines the core network architecture used
   for all models. The core network is passed to models.ContinualLearningModel
   child classes as the 'encoder' argument.
   This allows us to separate the network definition from the
   continual learning mechanisms (e.g., output layer expansion)
   and algorithms.

   The core network is defined in the 'Encoder' class.

   The 'Decoder' class is used only by the 'models.AntReg' model class.
"""

from collections import OrderedDict
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d
from torch.nn import Dropout, BatchNorm2d, LeakyReLU
from torch.nn.functional import interpolate
import torch


class Flatten(torch.nn.Module):
    """Custom layer to transition from Conv to Linear."""

    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(torch.nn.Module):
    """Custom layer to transition from Linear to Conv.

       args:
          shape -- output shape.
    """

    def __init__(self, shape):
        super(UnFlatten, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class UpSample(torch.nn.Module):
    def forward(self, x):
        return interpolate(x, scale_factor=2)


def get_fc_block(in_features,
                 out_features,
                 activation,
                 dropout=None):
    block = Sequential(
        OrderedDict([('fc', Linear(in_features, out_features))]))
    if dropout is not None:
        block.add_module('dropout', Dropout(dropout))
    if activation is not None:
        block.add_module('activation', activation)
    return block


def get_conv_block(in_channels,
                   out_channels,
                   kernel_size,
                   activation,
                   pool,
                   batchnorm,
                   dropout=None):
    if kernel_size == 3:
        padding = 1
    elif kernel_size == 5:
        padding = 2
    elif kernel_size == 7:
        padding = 3
    else:
        raise Exception('kernel size can be 3, 5, or 7 only')
    block = Sequential(
        OrderedDict([('conv',
                      Conv2d(
                          in_channels,
                          out_channels,
                          kernel_size,
                          padding=padding))]))
    if batchnorm:
        block.add_module('batchnorm', BatchNorm2d(out_channels))
    if activation is not None:
        block.add_module('activation', activation)
    if dropout is not None:
        block.add_module('dropout', Dropout(dropout))
    if pool:
        block.add_module('pool', MaxPool2d(2))
    return block


def get_deconv_block(in_channels,
                     out_channels,
                     kernel_size,
                     activation,
                     upsample,
                     batchnorm):
    if kernel_size == 3:
        padding = 1
    elif kernel_size == 5:
        padding = 2
    elif kernel_size == 7:
        padding = 3
    else:
        raise Exception('kernel size can be 3, 5, or 7 only')
    block = Sequential(OrderedDict([]))
    if upsample:
        block.add_module('upsample', UpSample())
    block.add_module(
        'conv', Conv2d(
            in_channels, out_channels, kernel_size, padding=padding))
    if batchnorm:
        block.add_module('bn', BatchNorm2d(out_channels))
    if activation is not None:
        block.add_module('activation', activation)
    return block


class Encoder(torch.nn.Module):
    """Class defining core network used in all models.

       args:
          in_shape -- tuple, expected input images shape
             The architecture was tested with in_shape = (3, 32, 32)
             Not tested for other shapes
             Width and height must be multiples of 4
          out_feature -- int, number of units in final layer of encoder
          batchnorm -- bool, whether to use batchnorm in Conv layers
          activation -- expects 'relu' or 'leaky'. Any other value is treated as 'leaky'
          conv_dropout -- [None, float], whether to use dropout in Conv layers
             if None, no dropout; otherwise sets value of dropout probability
          fc_dropout -- same as conv_dropout, but for the Linear layers
          fc1nb -- int, number of units in first Linear layer
          cnb -- int, number of Conv units in each Conv layer
    """

    def __init__(self,
                 in_shape,
                 out_features,
                 batchnorm,
                 activation,
                 conv_dropout=None,
                 fc_dropout=None,
                 fc1nb=1024,
                 cnb=128):
        super(Encoder, self).__init__()
        c, h, w = in_shape
        assert w % 4 == 0
        assert h % 4 == 0
        self.conv = Sequential(
            OrderedDict([
                ('conv1',
                 get_conv_block(
                     c,
                     cnb,
                     3,
                     activation=ReLU() if activation == 'relu' else LeakyReLU(0.1),
                     pool=True,
                     batchnorm=batchnorm,
                     dropout=conv_dropout)),
                ('conv2',
                 get_conv_block(
                     cnb,
                     cnb,
                     3,
                     activation=ReLU() if activation == 'relu' else LeakyReLU(0.1),
                     pool=True,
                     batchnorm=batchnorm,
                     dropout=conv_dropout)),
                ('conv3',
                 get_conv_block(
                     cnb,
                     cnb,
                     3,
                     activation=ReLU() if activation == 'relu' else LeakyReLU(0.1),
                     pool=False,
                     batchnorm=batchnorm,
                     dropout=conv_dropout))]))
        self.flatten = Flatten()
        self.mlp = Sequential(
            OrderedDict([
                ('fc1',
                 get_fc_block(
                     cnb * h * w // 16,
                     fc1nb,
                     activation=ReLU() if activation == 'relu' else LeakyReLU(0.1),
                     dropout=fc_dropout)),
                ('fc2',
                 get_fc_block(
                     fc1nb, out_features,
                     activation=ReLU() if activation == 'relu' else LeakyReLU(0.1),
                     dropout=fc_dropout))]))

    def forward(self, x):
        return self.mlp(self.flatten(self.conv(x)))


class Decoder(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_shape,
                 batchnorm,
                 fc1nb=1024,
                 cnb=128):
        super(Decoder, self).__init__()
        c, h, w = out_shape
        assert w % 4 == 0
        assert h % 4 == 0
        self.mlp = Sequential(
            OrderedDict([('fc1', get_fc_block(in_features, fc1nb, ReLU())),
                         ('fc2', get_fc_block(fc1nb, cnb * h * w // 16,
                                              ReLU()))]))
        self.unflatten = UnFlatten(shape=(cnb, h // 4, w // 4))
        self.conv = Sequential(
            OrderedDict([('conv1',
                          get_deconv_block(
                              cnb,
                              cnb,
                              3,
                              ReLU(),
                              upsample=True,
                              batchnorm=batchnorm)),
                         ('conv2',
                          get_deconv_block(
                              cnb,
                              cnb,
                              3,
                              ReLU(),
                              upsample=True,
                              batchnorm=batchnorm)),
                         ('conv3',
                          get_deconv_block(
                              cnb,
                              c,
                              3,
                              activation=None,
                              upsample=False,
                              batchnorm=batchnorm))]))

    def forward(self, x):
        return self.conv(self.unflatten(self.mlp(x)))
