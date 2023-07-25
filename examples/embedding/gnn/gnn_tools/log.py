import multiprocessing
import numpy as np
import time

logfile = open("log.txt", "w")


class SharedTrainingStat():
    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.lock = self.manager.Lock()
        self.total = self.manager.Value("total", 0)
        self.acc = self.manager.Value("acc", 0)
        self.loss = self.manager.Value("loss", 0.0)
        self.count = self.manager.Value("count", 0)
        self.train_total = self.manager.Value("train_total", 0)
        self.train_acc = self.manager.Value("train_acc", 0)
        self.train_loss = self.manager.Value("train_loss", 0.0)
        self.train_count = self.manager.Value("train_count", 0)
        self.time = []

    def update(self, acc, total, loss):
        self.lock.acquire()
        self.total.value += total
        self.acc.value += acc
        self.loss.value += loss
        self.count.value += 1
        self.lock.release()

    def update_train(self, acc, total, loss):
        self.lock.acquire()
        self.train_total.value += total
        self.train_acc.value += acc
        self.train_loss.value += loss
        self.train_count.value += 1
        self.lock.release()

    def print(self, start=""):
        self.lock.acquire()
        if len(self.time) > 3:
            epoch_time = np.array(self.time[1:])-np.array(self.time[:-1])
            print(
                "epoch time: {:.3f}+-{:.3f}".format(np.mean(epoch_time), np.var(epoch_time)))
        self.time.append(time.time())
        print(
            start,
            "test loss: {:.3f} test acc: {:.3f} train loss: {:.3f} train acc: {:.3f}".format(
                self.loss.value / self.count.value,
                self.acc.value / self.total.value,
                self.train_loss.value / self.train_count.value,
                self.train_acc.value / self.train_total.value
            )
        )
        print(
            self.loss.value / self.count.value, self.acc.value / self.total.value,
            self.train_loss.value / self.train_count.value, self.train_acc.value /
            self.train_total.value,
            file=logfile, flush=True
        )
        self.total.value = 0
        self.acc.value = 0
        self.loss.value = 0
        self.count.value = 0
        self.train_total.value = 0
        self.train_acc.value = 0
        self.train_loss.value = 0
        self.train_count.value = 0
        self.lock.release()
