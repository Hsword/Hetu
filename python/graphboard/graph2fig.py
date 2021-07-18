from __future__ import absolute_import

from graphviz import Digraph
import subprocess
import os
import signal

pid = None


def show(executor, port=9997):
    print("Generating graph figure")
    dot = Digraph()
    dot.format = 'png'
    for node in executor.topo_order:
        dot.node(str(node.id), node.name)
        print(node.desc)
        if node.inputs:
            for n in node.inputs:
                dot.edge(str(n.id), str(node.id))
    print(dot.source)
    dot.render('python/graphboard/output')
    print("Starting server..")
    cmd = 'cd python/graphboard; python -m SimpleHTTPServer '+str(port)
    pro = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    global pid
    pid = pro.pid


def close():
    global pid
    os.killpg(pid, signal.SIGTERM)
