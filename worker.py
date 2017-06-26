from multiprocessing import Queue, Process
import cv2
import numpy as np
import os
import net_builder

# helper class for scheduling workers
class Scheduler:
    def __init__(self, workerids, use_cuda):
        self._queue = Queue()
        self.workerids = workerids
        self._results = Queue()
        self.use_cuda = use_cuda

        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for wid in self.workerids:
            self._workers.append(CustomWorker(wid, self._queue, self._results, self.use_cuda))


    def start(self, xlist):

        # put all of models into queue
        for model_info in xlist:
            self._queue.put(model_info)

        #add a None into queue to indicate the end of task
        self._queue.put(None)

        #start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        
        print("All workers are done")
        returns = []
        networks = []
        for i in range(len(xlist)):
            score, net = self._results.get()
            returns.append(score)
            networks.append(net)

        return networks, returns

class CustomWorker(Process):
    def __init__(self, workerid, queue, resultq, use_cuda):
        Process.__init__(self, name='ModelProcessor')
        self.workerid = workerid
        self.queue = queue
        self.resultq = resultq
        self.use_cuda = use_cuda
        from torchvision import datasets, transforms
        import torch.utils as utils

        batch_size = 64
        # setup our dataloaders
        self.train_loader = utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=batch_size, shuffle=True)

        self.test_loader = utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=True)


    def run(self):
        import torch_models
        while True:
            net = self.queue.get()
            if net == None:
                self.queue.put(None) # for other workers to consume it.
                break
            # net = net_builder.randomize_network(bounded=False)
            xnet  = torch_models.CustomModel(net, self.use_cuda)
            ret = xnet.train(self.train_loader)
            score = -1
            if ret ==1:
                score = xnet.test(self.test_loader)
                print('worker_{} score:{}'.format(self.workerid, score))
            self.resultq.put((score, net))
            del xnet
