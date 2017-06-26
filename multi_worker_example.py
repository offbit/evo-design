from multiprocessing import Process, Queue, Pool
import os
import argparse
from worker import CustomWorker
# from worker import train_model
import net_builder

class Scheduler:
    def __init__(self, workerids):
        self._queue = Queue()
        self.workerids = workerids
        self._results = Queue()

        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for wid in self.workerids:
            self._workers.append(CustomWorker(wid, self._queue, self._results))


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
        
        
        print "all of workers have been done"
        for i in range(len(xlist)):
            print(i)
            a = self._results.get()
            print(a)

def run(workerids):
    #scan all files under img_path
    xlist = list()
    for i in range(5):
        xlist.append(net_builder.randomize_network())
    #init scheduler
    x = Scheduler(workerids)
    #start processing and wait for complete 
    x.start(xlist)


if __name__ == "__main__":

    # args = parser.parse_args()
    workerids = '0,1'
    workerids = [int(x) for x in workerids.strip().split(',')]

    run(workerids)
