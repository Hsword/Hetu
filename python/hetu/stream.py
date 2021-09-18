from __future__ import absolute_import

from ._base import _LIB, check_call
import ctypes
from . import ndarray


class DLStream(ctypes.Structure):
    _fields_ = [("device_id", ctypes.c_int),
                ("handle", ctypes.c_void_p)]


DLStreamHandle = ctypes.POINTER(DLStream)


class Stream(ctypes.Structure):
    __slots__ = ["handle"]

    def __init__(self, handle):
        self.handle = handle

    def __del__(self):
        check_call(_LIB.DLStreamDestroy(self.handle))

    def sync(self):
        check_call(_LIB.DLStreamSync(self.handle))


def create_stream_handle(ctx):
    assert ndarray.is_gpu_ctx(ctx)
    handle = DLStreamHandle()
    check_call(_LIB.DLStreamCreate(ctx.device_id, ctypes.byref(handle)))
    return Stream(handle)


class DLEvent(ctypes.Structure):
    _fields_ = [("device_id", ctypes.c_int),
                ("handle", ctypes.c_void_p)]


DLEventHandle = ctypes.POINTER(DLEvent)


class Event(ctypes.Structure):
    __slots__ = ["handle"]

    def __init__(self, handle):
        self.handle = handle

    def __del__(self):
        check_call(_LIB.DLEventDestroy(self.handle))

    def sync(self):
        check_call(_LIB.DLEventSync(self.handle))

    def record(self, stream_handle):
        check_call(_LIB.DLEventRecord(stream_handle.handle, self.handle))

    def time_since(self, event):
        result = ctypes.c_float()
        check_call(_LIB.DLEventElapsedTime(
            event.handle, self.handle, ctypes.byref(result)))
        return result.value


def create_event_handle(ctx):
    assert ndarray.is_gpu_ctx(ctx)
    handle = DLEventHandle()
    check_call(_LIB.DLEventCreate(ctx.device_id, ctypes.byref(handle)))
    return Event(handle)


class PSEvent(object):
    __slots__ = ["comm", "nid", "need_wait"]

    def __init__(self, comm, nid):
        self.comm = comm
        self.nid = nid
        self.need_wait = False

    def update(self):
        self.need_wait = True

    def sync(self):
        if self.need_wait:
            self.comm.Wait(self.nid)
        self.need_wait = False


class CSEvent(PSEvent):
    __slots__ = ["tss"]

    def __init__(self, comm, nid):
        super().__init__(comm, nid)
        self.tss = []

    def update_ts(self, ts):
        self.tss.append(ts)

    def sync(self):
        super().sync()
        if self.tss != []:
            for ts in self.tss:
                ts.wait()
        self.tss = []
