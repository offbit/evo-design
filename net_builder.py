"""Build neural networks for Evolution."""
from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import random
from torchvision import datasets, transforms
import numpy as np


# Layer space & net space define the way a model is built and mutated.

LAYER_SPACE = dict()
LAYER_SPACE['nb_units'] = (128, 1024, 'int', 0.2)
LAYER_SPACE['dropout_rate'] = (0.0, 0.7, 'float', 0.2)
LAYER_SPACE['activation'] =\
    (0,  ['linear', 'tanh', 'relu', 'sigmoid', 'elu'], 'list', 0.2)


NET_SPACE = dict()
NET_SPACE['nb_layers'] = (1, 4, 'int', 0.1)
NET_SPACE['lr'] = (0.0005, 0.2, 'float', 0.2)
NET_SPACE['weight_decay'] = (0.00005, 0.002, 'float', 0.2)
NET_SPACE['optimizer'] =\
    (0, ['sgd', 'adam', 'adadelta', 'rmsprop'], 'list', 0.2)


def check_and_assign(val, space):
    """assign a value between the boundaries."""
    val = min(val, space[0])
    val = max(val, space[1])
    return val


def random_value(space):
    """Sample  random value from the given space."""
    val = None
    if space[2] == 'int':
        val = random.randint(space[0], space[1])
    if space[2] == 'list':
        val = random.sample(space[1], 1)[0]
    if space[2] == 'float':
        val = ((space[1] - space[0]) * random.random()) + space[0]
    return {'val': val, 'id': random.randint(0, 2**10)}


def randomize_network(bounded=True):
    """Create a random network."""
    global NET_SPACE, LAYER_SPACE
    net = dict()
    for k in NET_SPACE.keys():
        net[k] = random_value(NET_SPACE[k])
    
    if bounded: 
        net['nb_layers']['val'] = min(net['nb_layers']['val'], 1)
    
    layers = []
    for i in range(net['nb_layers']['val']):
        layer = dict()
        for k in LAYER_SPACE.keys():
            layer[k] = random_value(LAYER_SPACE[k])
        layers.append(layer)
    net['layers'] = layers
    return net


def mutate_net(net):
    """Mutate a network."""
    global NET_SPACE, LAYER_SPACE

    # mutate optimizer
    for k in ['lr', 'weight_decay', 'optimizer']:
        
        if random.random() < NET_SPACE[k][-1]:
            net[k] = random_value(NET_SPACE[k])
            
    # mutate layers
    for layer in net['layers']:
        for k in LAYER_SPACE.keys():
            if random.random() < LAYER_SPACE[k][-1]:
                layer[k] = random_value(LAYER_SPACE[k])
    # mutate number of layers -- RANDOMLY ADD
    if random.random() < NET_SPACE['nb_layers'][-1]:
        if net['nb_layers']['val'] < NET_SPACE['nb_layers'][1]:
            if random.random()< 0.5:
                layer = dict()
                for k in LAYER_SPACE.keys():
                    layer[k] = random_value(LAYER_SPACE[k])
                net['layers'].append(layer)
                # value & id update
                net['nb_layers']['val'] = len(net['layers'])
                net['nb_layers']['id'] +=1
            else:
                if net['nb_layers']['val'] > 1:
                    net['layers'].pop()
                    net['nb_layers']['val'] = len(net['layers'])
                    net['nb_layers']['id'] -=1
    return net


class Flatten(nn.Module):
    """A simple flatten module."""

    def __init__(self):
        """Call init."""
        super(Flatten, self).__init__()

    def forward(self, x):
        """forward pass."""
        return x.view(x.size(0), -1)


def make_model(build_info):
    """make a model according to the build info."""
    previous_units = 28 * 28
    model = nn.Sequential()
    model.add_module('flatten', Flatten())
    for i, layer_info in enumerate(build_info['layers']):
        i = str(i)
        model.add_module(
            'fc_' + i,
            nn.Linear(previous_units, layer_info['nb_units']['val'])
            )
        model.add_module(
            'dropout_' + i,
            nn.Dropout(p=layer_info['dropout_rate']['val'])
            )
        if layer_info['activation']['val'] == 'tanh':
            model.add_module(
                'tanh_'+i,
                nn.Tanh()
            )
        if layer_info['activation']['val'] == 'relu':
            model.add_module(
                'relu_'+i,
                nn.ReLU()
            )
        if layer_info['activation']['val'] == 'sigmoid':
            model.add_module(
                'sigm_'+i,
                nn.Sigmoid()
            )
        if layer_info['activation']['val'] == 'elu':
            model.add_module(
                'elu_'+i,
                nn.ELU()
            )
        previous_units = layer_info['nb_units']['val']

    model.add_module(
        'classification_layer',
        nn.Linear(previous_units, 10)
        )
    model.add_module('sofmax', nn.LogSoftmax())
    model.cpu()
    return model


def train(train_loader, model, optimizer, max_batches=100, CUDA=True):
    """Train for 1 epoch."""
    model.train()
    batch = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if CUDA:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
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
        optimizer.step()
        batch+=1
        if batch > max_batches:
            break
    return 1


def test(test_loader, model, CUDA=True):
    """Evaluate a model."""
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if CUDA:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    # loss function already averages over batch size
    test_loss /= len(test_loader)
    # print('Test set: Average loss:{:.4f},Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    accuarcy = 100. * correct / len(test_loader.dataset)
    return accuarcy


if __name__ == '__main__':
    net_params = randomize_network()
    for k in ['lr', 'optimizer', 'weight_decay']:
        print(k, net_params[k]['val'])
    batch_size = 128
    model = make_model(net_params)
    print(model)
    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    optimizer = optim.Adam(model.parameters(),
                           lr=net_params['weight_decay']['val'],
                           weight_decay=net_params['weight_decay']['val'])

    for epoch in range(1, 5):
        train(train_loader, model, optimizer)
    test(test_loader, model)
