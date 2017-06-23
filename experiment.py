"""Tournament play experiment."""
from __future__ import absolute_import
import net_builder
import gp
import torch.utils as utils
import torch.optim as optim
from torchvision import datasets, transforms
import cPickle
# Use cuda ?
CUDA_ = True

def eval_model(net_params):
    """Model evaluator."""
    model = net_builder.make_model(net_params)
    if CUDA_:
        model.cuda()

    
    batch_size = 64 
    
    # setup our dataloaders
    train_loader = utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)
    
    # select optimizer
    if net_params['optimizer']['val'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=net_params['weight_decay']['val'],
                               weight_decay=net_params['weight_decay']['val'])

    elif net_params['optimizer']['val'] == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(),
                                   lr=net_params['weight_decay']['val'],
                                   weight_decay=net_params['weight_decay']['val'])

    elif net_params['optimizer']['val'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=net_params['weight_decay']['val'],
                                  weight_decay=net_params['weight_decay']['val'])
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=net_params['weight_decay']['val'],
                              weight_decay=net_params['weight_decay']['val'],
                              momentum=0.9)
        
    # train only on 12000 examples
    ret = net_builder.train(train_loader,
                            model, optimizer, 
                            max_batches=12000 // batch_size,
                            CUDA=CUDA_)
    if ret == -1:
        return 0
    accuracy = net_builder.test(test_loader, model, CUDA=CUDA_)
    return accuracy

if __name__=='__main__':
    # setup a tournament!
    nb_evolution_steps = 10
    tournament = \
        gp.TournamentOptimizer(
            population_sz=50,
            init_fn=net_builder.randomize_network,
            mutate_fn=net_builder.mutate_net,
            eval_fn=eval_model,
            nb_workers=1)

    for i in range(nb_evolution_steps):
        print('\nEvolution step:{}'.format(i))
        print('================')
        tournament.step()
        # keep track of the experiment results & corresponding architectures
        name = "tourney_{}".format(i)
        cPickle.dump(tournament.stats, open(name + '.stats','wb'))
        cPickle.dump(tournament.history, open(name +'.pop','wb'))