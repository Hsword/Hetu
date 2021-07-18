import os
import os.path as osp
import signal
import yaml
import multiprocessing

import libc_graphmix as _C
import hetu as ht
from graphmix.shard import Shard

default_graph_root_port = 27770


def start_graph_server(shard, server_init):
    os.environ['GRAPHMIX_ROLE'] = "server"
    _C.init()
    shard.load_graph_shard(_C.rank())
    server = _C.start_server()
    server.init_meta(shard.meta)
    server.init_data(shard.f_feat, shard.i_feat, shard.edges)
    del shard
    print("GraphMix Server {} : data initialized at {}:{}".format(
        _C.rank(), _C.ip(), _C.port()))
    _C.barrier_all()
    server_init(server)
    _C.finalize()


def start_server():
    os.environ["DMLC_ROLE"] = "server"
    ht.server_init()
    ht.server_finish()

# two scheduler in one process


def start_scheduler():
    os.environ['GRAPHMIX_ROLE'] = "scheduler"
    os.environ['DMLC_ROLE'] = "scheduler"
    _C.init()
    ht.scheduler_init()
    ht.scheduler_finish()
    _C.finalize()


def start_worker(func, args):
    os.environ['GRAPHMIX_ROLE'] = "worker"
    os.environ['DMLC_ROLE'] = "worker"
    _C.init()
    ht.worker_init()
    args.local_rank = _C.rank() % args.num_local_worker
    _C.barrier_all()
    func(args)
    ht.worker_finish()
    _C.finalize()


def start_worker_standalone(func, args, local_rank):
    args.local_rank = local_rank
    func(args)


def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    for proc in process_list:
        proc.kill()
    exit(0)


process_list = []


def launch_graphmix_and_hetu_ps(target, args, server_init, hybrid_config=None):
    # open setting file
    file_path = osp.abspath(osp.expanduser(osp.normpath(args.config)))
    with open(file_path) as setting_file:
        settings = yaml.load(setting_file.read(), Loader=yaml.FullLoader)

    # write environment variables
    for key, value in settings["shared"].items():
        os.environ[str(key)] = str(value)

    # the graph data path is relative to the setting file path
    graph_data_path = osp.abspath(osp.expanduser(osp.normpath(args.path)))
    print("GraphMix launcher : Using Graph Data from ", graph_data_path)

    # load graph and set the server number equal to the number of graph parts
    shard = Shard(graph_data_path)
    os.environ['GRAPHMIX_NUM_SERVER'] = str(shard.meta["num_part"])
    os.environ['GRAPHMIX_NUM_WORKER'] = os.environ['DMLC_NUM_WORKER']
    os.environ['GRAPHMIX_ROOT_URI'] = os.environ['DMLC_PS_ROOT_URI']
    os.environ['GRAPHMIX_ROOT_PORT'] = str(default_graph_root_port)
    if 'DMLC_INTERFACE' in os.environ.keys():
        os.environ['GRAPHMIX_INTERFACE'] = os.environ['DMLC_INTERFACE']

    # get local job number
    args.num_local_worker = int(settings["launch"]["worker"])
    args.num_local_graph_server = int(settings["launch"]["graph_server"])
    args.num_local_server = int(settings["launch"]["server"])
    args.scheduler = settings["launch"]["scheduler"]
    assert args.num_local_graph_server <= shard.meta["num_part"]
    assert args.num_local_worker <= int(os.environ['DMLC_NUM_WORKER'])
    assert args.num_local_server <= int(os.environ['DMLC_NUM_SERVER'])
    if hybrid_config == "worker":
        args.num_local_server = 0
        args.num_local_graph_server = 0
        args.scheduler = False
        args.num_local_worker = 1
    elif hybrid_config == "server":
        args.num_local_worker = 0

    # launch workers
    for i in range(args.num_local_worker):
        proc = multiprocessing.Process(
            target=start_worker, args=[target, args])
        process_list.append(proc)
    # launch graph servers
    for i in range(args.num_local_graph_server):
        proc = multiprocessing.Process(
            target=start_graph_server, args=[shard, server_init])
        process_list.append(proc)
    # launch ps servers
    for i in range(args.num_local_server):
        proc = multiprocessing.Process(target=start_server, args=[])
        process_list.append(proc)
    # launch scheduler
    if args.scheduler:
        proc = multiprocessing.Process(target=start_scheduler)
        process_list.append(proc)
    # wait until all process finish
    for proc in process_list:
        proc.start()
    signal.signal(signal.SIGINT, signal_handler)
    for proc in process_list:
        proc.join()
