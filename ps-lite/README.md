# PS-lite Module [adapted from https://github.com/dmlc/ps-lite]

---

## Use Guide

PS-lite module is a a light-weighted C++ parameter server with ctypes python interface. It provides a list of PS functions that are useful in distributed training.

To use PS,we will have three roles: worker, server and scheduler. Worker are training process. Servers are where the parameters are stored. The scheduler setup and tear down the connection. There will be multiple servers and workers and only one scheduler.

Currently, We only implemented python interface for hetu. Since it contains some symbol from hetu, we can only use ps functions after we import hetu. Here is a quick example on how we use ps-lite with hetu.

```python
# worker.py
import hetu
import numpy as np
import ctypes
# create arrays
tgt_array = hetu.ndarray.empty([128])
name = 0 # A number specifies a parameter, should be the same among all workers
param_type = 0 # 0 for dense parameter
# PS initialize
hetu.worker_init()
# PS functions here
comm = hetu.get_worker_communicate()
# InitTensor(node_name, param_type, length, width, init_type, init_param_a, init_param_b, seed, opt_type, opt_args, num_opt_args)
# This function is synchronous.
comm.InitTensor(name, param_type, 128, 1, 0, 5.0, 1.0, 123, 0, (ctypes.c_float * 1)(0.1), 1)
comm.Pull(name, tgt_array.handle)
comm.Wait(name)
print(tgt_array.asnumpy())
# PS finialize
hetu.worker_finish()
```
We will also have server code and scheduler code
```python
# server.py
import hetu
hetu.server_init()
hetu.server_finish()
```

```python
# scheduler.py
import hetu
hetu.scheduler_init()
hetu.scheduler_finish()
```

To run the sricpts, we should use environment variables to specify which ip address and port to use. Note that it is recommended to use a yaml or json file to store these environment variables.

```shell
export DMLC_PS_ROOT_URI=127.0.0.1 DMLC_PS_ROOT_PORT=4080 DMLC_NUM_WORKER=1 DMLC_NUM_SERVER=1 DMLC_PS_VAN_TYPE=p3
DMLC_ROLE=scheduler python3 scheduler.py &
DMLC_ROLE=server SERVER_ID=0 DMLC_PS_SERVER_URI=127.0.0.1 DMLC_PS_SERVER_PORT=4081 python3 server.py &
DMLC_ROLE=worker WORKER_ID=0 DMLC_PS_WORKER_URI=127.0.0.1 DMLC_PS_WORKER_PORT=4082 python3 worker.py
```

## PS functions

We provide a list of useful parameter server functions for training.

It also has the ability to easily extend to new ps functions. There will be several steps to go.

1. Create a enum in psf/PSFunc.h and write a struct to define the ps function.

   ```C++
   template<> struct PSFData<kMyFunction> {
     using Request = tuple<
       unsigned long,
       SArray<float>
     >;
     using Response = tuple<>;
     static void _callback(const Response &response) {/* callback here */}
   };
   ```

   here we can use scalar types like int,float... or arrays as function parameters. Note that arrays are shared and scalars are copied.

2. Implement server handler in server/PSFHandler.h

3. use a kvworker.Request to launch yout ps function and kvworker.Wait to wait till callback ends, see more example in PSAgent.h. We can also write python binding to expose the ps function to python layer.

