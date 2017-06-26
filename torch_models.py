"""Build torch models."""
from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import random
from torchvision import datasets, transforms
import numpy as np


class Flatten(nn.Module):
    """A simple flatten module."""

    def __init__(self):
        """Call init."""
        super(Flatten, self).__init__()

    def forward(self, x):
        """forward pass."""
        return x.view(x.size(0), -1)

class CustomModel():

    def __init__(self, build_info, CUDA=True):

        previous_units = 28 * 28
        self.model = nn.Sequential()
        self.model.add_module('flatten', Flatten())
        for i, layer_info in enumerate(build_info['layers']):
            i = str(i)
            self.model.add_module(
                'fc_' + i,
                nn.Linear(previous_units, layer_info['nb_units']['val'])
                )
            self.model.add_module(
                'dropout_' + i,
                nn.Dropout(p=layer_info['dropout_rate']['val'])
                )
            if layer_info['activation']['val'] == 'tanh':
                self.model.add_module(
                    'tanh_'+i,
                    nn.Tanh()
                )
            if layer_info['activation']['val'] == 'relu':
                self.model.add_module(
                    'relu_'+i,
                    nn.ReLU()
                )
            if layer_info['activation']['val'] == 'sigmoid':
                self.model.add_module(
                    'sigm_'+i,
                    nn.Sigmoid()
                )
            if layer_info['activation']['val'] == 'elu':
                self.model.add_module(
                    'elu_'+i,
                    nn.ELU()
                )
            previous_units = layer_info['nb_units']['val']

        self.model.add_module(
            'classification_layer',
            nn.Linear(previous_units, 10)
            )
        self.model.add_module('sofmax', nn.LogSoftmax())
        self.model.cpu()
        
        if build_info['optimizer']['val'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                lr=build_info['weight_decay']['val'],
                                weight_decay=build_info['weight_decay']['val'])

        elif build_info['optimizer']['val'] == 'adadelta':
            optimizer = optim.Adadelta(self.model.parameters(),
                                    lr=build_info['weight_decay']['val'],
                                    weight_decay=build_info['weight_decay']['val'])

        elif build_info['optimizer']['val'] == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(),
                                    lr=build_info['weight_decay']['val'],
                                    weight_decay=build_info['weight_decay']['val'])
        else:
            optimizer = optim.SGD(self.model.parameters(),
                                lr=build_info['weight_decay']['val'],
                                weight_decay=build_info['weight_decay']['val'],
                                momentum=0.9)
        self.optimizer = optimizer
        self.cuda = False
        if CUDA:
            self.model.cuda()
            self.cuda = True
        
    
    def train(self, train_loader, max_batches=100):
        """Train for 1 epoch."""
        self.model.train()

        batch = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            # print(type(loss))
            np_loss = loss.cpu().data.numpy()
            if np.isnan(np_loss):
                print('stopping training - nan loss')
                return -1
            elif loss.cpu().data.numpy()[0] > 100000:
                print('Qutting, loss too high', np_loss)
                return -1

            loss.backward()
            self.optimizer.step()
            batch+=1
            if batch > max_batches:
                break
        return 1


    def test(self, test_loader, CUDA=False):
        """Evaluate a model."""
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target).data[0]
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

        test_loss /= len(test_loader)
        accuarcy = 100. * correct / len(test_loader.dataset)
        return accuarcy
    