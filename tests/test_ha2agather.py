from ctypes import *
from hetu import ndarray
from hetu.stream import *
from hetu.context import DeviceGroup
import numpy as np
import hetu as ht
import hetu.communicator.mpi_nccl_comm as fuck

if __name__ == "__main__":
    t = ht.wrapped_mpi_nccl_init()        
    send_arr = np.ones(16)*t.localRank.value        
    recv_arr = np.ones(16*8)*t.localRank.value            
    print("before: send_arr = "+str(send_arr)+" recv_arr = "+str(recv_arr))                
    send_arr = ndarray.array(send_arr, ctx=ndarray.gpu(t.device_id.value))    
    recv_arr = ndarray.array(recv_arr, ctx=ndarray.gpu(t.device_id.value))
    t.dlarrayHA2AGather(send_arr, recv_arr,fuck.ncclDataType_t.ncclFloat32, t.localRank.value, 8)
    print("after:  send_arr = "+str(send_arr.asnumpy())+" recv_arr = "+str(recv_arr.asnumpy()))

