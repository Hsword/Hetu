import argparse
import yaml
import os
import signal
import multiprocessing
import hetu as ht

_procs = []


def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    for proc in _procs:
        proc.kill()
    exit(0)


def launch(target, args):
    file_path = args.config
    settings = yaml.load(open(file_path).read(), Loader=yaml.FullLoader)
    for k, v in settings['shared'].items():
        os.environ[k] = str(v)
    args.num_local_worker = int(settings["launch"]["worker"])
    for i in range(args.num_local_worker):
        proc = multiprocessing.Process(
            target=start_worker, args=[target, args])
        _procs.append(proc)
    for i in range(int(settings["launch"]["server"])):
        proc = multiprocessing.Process(target=start_server)
        _procs.append(proc)
    if settings["launch"]["scheduler"] != 0:
        proc = multiprocessing.Process(target=start_sched)
        _procs.append(proc)
    signal.signal(signal.SIGINT, signal_handler)
    for proc in _procs:
        proc.start()
    for proc in _procs:
        proc.join()


def start_sched():
    os.environ["DMLC_ROLE"] = "scheduler"
    ht.scheduler_init()
    ht.scheduler_finish()


def start_server():
    os.environ["DMLC_ROLE"] = "server"
    ht.server_init()
    ht.server_finish()


def start_worker(target, args):
    os.environ["DMLC_ROLE"] = "worker"
    ht.worker_init()
    target(args)
    ht.worker_finish()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("--sched", action="store_true")
    args = parser.parse_args()
    file_path = args.config
    settings = yaml.load(open(file_path).read(), Loader=yaml.FullLoader)
    for k, v in settings['shared'].items():
        os.environ[k] = str(v)
    if args.sched:
        _procs.append(multiprocessing.Process(target=start_sched))
    for i in range(args.n):
        _procs.append(multiprocessing.Process(target=start_server))
    for proc in _procs:
        proc.start()
    for proc in _procs:
        proc.join()

__all__ = [
    'launch'
]
