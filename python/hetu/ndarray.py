from __future__ import absolute_import

from ._base import _LIB, check_call, c_array
import ctypes
import numpy as np
import scipy.sparse
import socket


class DLContext(ctypes.Structure):
    """DL context strucure."""
    _fields_ = [("device_id", ctypes.c_int),
                ("device_type", ctypes.c_int)]

    MASK2STR = {
        1: 'cpu',
        2: 'gpu',
    }

    def __init__(self, device_id, device_type, hostname='localhost'):
        super(DLContext, self).__init__()
        self.device_id = device_id
        self.device_type = device_type
        if hostname in ('localhost', socket.gethostname()):
            self.hostname = 'localhost'
            self.local = True
        else:
            self.hostname = hostname
            self.local = False

    def __repr__(self):
        if not hasattr(self, 'local') or self.local:
            return "%s(%d)" % (
                DLContext.MASK2STR[self.device_type], self.device_id)
        else:
            return "%s:%s(%d)" % (
                self.hostname, DLContext.MASK2STR[self.device_type], self.device_id)

    def __hash__(self):
        if not hasattr(self, 'local') or self.local:
            return hash((self.device_type, self.device_id))
        else:
            return hash((self.hostname, self.device_type, self.device_id))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return hash(self) != hash(other)


class DLArray(ctypes.Structure):
    """DLArray in C API"""
    _fields_ = [("data", ctypes.c_void_p),
                ("ctx", DLContext),
                ("ndim", ctypes.c_int),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("stride", ctypes.POINTER(ctypes.c_int64))]


DLArrayHandle = ctypes.POINTER(DLArray)


def cpu(dev_id=0):
    """Construct a CPU device
    Parameters
    ----------
    dev_id : int, optional
        The integer device id
    """
    return DLContext(dev_id, 1)


def gpu(dev_id=0):
    """Construct a GPU device
    Parameters
    ----------
    dev_id : int, optional
        The integer device id
    """
    return DLContext(dev_id, 2)


def rcpu(hostname, dev_id=0):
    """Construct a remote CPU device
    Parameters
    ----------
    hostname: str
        The hostname of device
    dev_id : int, optional
        The integer device id
    """
    return DLContext(dev_id, 1, hostname=hostname)


def rgpu(hostname, dev_id=0):
    """Construct a remote GPU device
    Parameters
    ----------
    hostname: str
        The hostname of device
    dev_id : int, optional
        The integer device id
    """
    return DLContext(dev_id, 2, hostname=hostname)


def is_gpu_ctx(ctx):
    """Return if context is GPU context.
    Parameters
    ----------
    ctx : DLContext
        The query context
    """
    return ctx and ctx.device_type == 2


def shape_to_stride(shape):
    """Return the stride.
    Parameters
    ----------
    shape : tuple(int)
        The shape tuple
    """
    ndim = len(shape)
    stride = [1] * ndim
    for i in range(ndim-1, 0, -1):
        stride[i-1] = stride[i] * shape[i]
    return tuple(stride)


class NDArray(object):
    """Lightweight NDArray class of DL runtime.
    Strictly this is only an Array Container(a buffer object)
    No arthimetic operations are defined.
    """
    __slots__ = ["handle", "no_free"]

    def __init__(self, handle):
        """Initialize the function with handle
        Parameters
        ----------
        handle : DLArrayHandle
            the handle to the underlying C++ DLArray
        """
        self.handle = handle
        self.no_free = False

    def __del__(self):
        if self.no_free:
            return
        check_call(_LIB.DLArrayFree(self.handle))

    @property
    def shape(self):
        """Shape of this array"""
        return tuple(self.handle.contents.shape[i]
                     for i in range(self.handle.contents.ndim))

    @property
    def stride(self):
        """Stride of this array"""
        return tuple(self.handle.contents.stride[i]
                     for i in range(self.handle.contents.ndim))

    @property
    def lazy(self):
        """Whether this array is lazy"""
        return not self.stride == shape_to_stride(self.shape)

    @property
    def ctx(self):
        """context of this array"""
        return self.handle.contents.ctx

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if (not isinstance(in_slice, slice) or
                in_slice.start is not None
                or in_slice.stop is not None):
            raise ValueError('Array only support set from numpy array')
        if isinstance(value, NDArray):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self._sync_copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def _sync_copyfrom(self, source_array, data_type=np.float32):
        """Peform an synchronize copy from the array.
        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.
        """
        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=data_type)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported'
                                % str(type(source_array)))
        source_array = np.ascontiguousarray(source_array, dtype=data_type)
        if source_array.shape != self.shape:
            raise ValueError('array shape do not match the shape of NDArray')
        source_arr, shape, stride = NDArray._numpyasarray(source_array)
        check_call(_LIB.DLArrayCopyFromTo(
            ctypes.byref(source_arr), self.handle, None))
        # de-allocate shape until now
        _ = shape
        _ = stride

    def _async_copyfrom(self, source_array, stream_handle, event_handle=None):
        """Peform an asynchronize copy from the array.
        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.
        """
        check_call(_LIB.DLArrayCopyFromTo(
            source_array.handle, self.handle, stream_handle.handle))
        if not event_handle is None:
            check_call(_LIB.DLEventRecord(
                stream_handle.handle, event_handle.handle))

    def async_h2d(self, source_array, stream_handle, event_handle=None):
        if isinstance(source_array, np.ndarray):
            source_array = array(source_array, cpu(0))
        assert self.handle.contents.ctx.device_type == 2
        assert source_array.handle.contents.ctx.device_type == 1
        assert stream_handle
        self._async_copyfrom(source_array, stream_handle, event_handle)

    def async_d2h(self, source_array, stream_handle, event_handle=None):
        assert self.handle.contents.ctx.device_type == 1
        assert source_array.handle.contents.ctx.device_type == 2
        assert stream_handle
        self._async_copyfrom(source_array, stream_handle, event_handle)

    @staticmethod
    def _numpyasarray(np_data):
        """Return a DLArray representation of a numpy array."""
        data = np_data
        assert data.flags['C_CONTIGUOUS']
        arr = DLArray()
        shape = c_array(ctypes.c_int64, data.shape)
        stride = c_array(ctypes.c_int64, shape_to_stride(data.shape))
        arr.data = data.ctypes.data_as(ctypes.c_void_p)
        arr.shape = shape
        arr.stride = stride
        arr.ndim = data.ndim
        # CPU device
        arr.ctx = cpu(0)
        return arr, shape, stride

    def asnumpy(self):
        """Convert this array to numpy array
        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        self.wrapped_lazy_callback()
        np_arr = np.empty(self.shape, dtype=np.float32)
        arr, shape, stride = NDArray._numpyasarray(np_arr)
        check_call(_LIB.DLArrayCopyFromTo(
            self.handle, ctypes.byref(arr), None))
        _ = shape
        _ = stride
        return np_arr

    def copyto(self, target):
        """Copy array to target
        Parameters
        ----------
        target : NDArray
            The target array to be copied, must have same shape as this array.
        """
        self.wrapped_lazy_callback()
        if isinstance(target, DLContext):
            target = empty(self.shape, target)
        if isinstance(target, NDArray):
            check_call(_LIB.DLArrayCopyFromTo(
                self.handle, target.handle, None))
        else:
            raise ValueError("Unsupported target type %s" % str(type(target)))
        return target

    def reshape(self, shape, target):
        """Reshape the array to target array.
        Parameters
        ----------
        shape : tuple (int)
            The target shape.
        target : NDArray
            The target array.
        """
        self.wrapped_lazy_callback()
        arr = DLArray()
        arr.data = self.handle.contents.data
        arr.ctx = self.handle.contents.ctx
        arr.ndim = len(shape)
        arr.shape = c_array(ctypes.c_int64, shape)
        arr.stride = c_array(ctypes.c_int64, shape_to_stride(shape))
        target.handle = ctypes.pointer(arr)
        target.no_free = True

    def inplace_copy(self, target):
        """Move the array to target array.
        Parameters
        ----------
        target : NDArray
            The target array.
        """
        self.wrapped_lazy_callback()
        arr = DLArray()
        arr.data = self.handle.contents.data
        arr.ctx = self.handle.contents.ctx
        arr.ndim = self.handle.contents.ndim
        arr.shape = self.handle.contents.shape
        arr.stride = self.handle.contents.stride
        target.handle = ctypes.pointer(arr)
        target.no_free = True

    def broadcast_to(self, shape, target, add_axes=None):
        """Broadcast the array to target array (lazy).
        Parameters
        ----------
        shape : tuple (int)
            The target shape.
        target : NDArray
            The target array.
        add_axes(Optional): list (int)
            Add axes if needed, using index of shape parameter.
            This is for gradient node of reduce_sum_op when there exists keepdims == False.
        """
        if add_axes is None:
            add_axes = []
        arr_ndim = len(shape)
        self_ndim = len(self.shape) + len(add_axes)
        ori_self_shape = list(self.shape)
        ori_self_stride = list(self.stride)
        if self_ndim > arr_ndim:
            assert self_ndim == arr_ndim + 1 and tuple(self.shape) == (1,)
            ori_self_shape = []
            ori_self_stride = []
        self_ndim = len(ori_self_shape)
        self_shape = [1] * arr_ndim
        self_stride = [0] * arr_ndim
        idx = self_ndim - 1
        target_stride = [0] * arr_ndim
        rule = True
        for i in range(arr_ndim):
            pos = arr_ndim - 1 - i
            if pos not in add_axes and idx >= 0:
                self_shape[pos] = ori_self_shape[idx]
                self_stride[pos] = ori_self_stride[idx]
                idx -= 1
            if self_shape[pos] == shape[pos]:
                target_stride[pos] = self_stride[pos]
            elif self_shape[pos] != 1:
                rule = False
                break
        assert rule
        arr = DLArray()
        arr.data = self.handle.contents.data
        arr.ctx = self.handle.contents.ctx
        arr.ndim = arr_ndim
        arr.shape = c_array(ctypes.c_int64, tuple(shape))
        arr.stride = c_array(ctypes.c_int64, tuple(target_stride))
        target.handle = ctypes.pointer(arr)
        target.no_free = True

    def lazy_callback(self, stream=None):
        assert self.handle.contents.ctx.device_type == 2
        assert self.lazy
        shape = c_array(ctypes.c_int64, self.shape)
        stride = c_array(ctypes.c_int64, shape_to_stride(self.shape))
        ndim = ctypes.c_int(len(self.shape))
        handle = DLArrayHandle()
        check_call(_LIB.DLArrayAlloc(shape, stride, ndim,
                                     self.handle.contents.ctx, ctypes.byref(handle)))
        check_call(_LIB.DLGpuArrayLazyCallback(
            self.handle, handle, stream.handle if stream else None))
        self.handle = handle

    def wrapped_lazy_callback(self, stream=None):
        # TODO: reshape / copyto / asnumpy may have more efficient implementation
        # This is just a workaround.
        if self.lazy:
            # here we move the judgement for lazy into forward hooks, shouldn't have callbacks.
            assert False
            self.lazy_callback(stream)


def array(arr, ctx, data_type=np.float32):
    """Create an array from source arr.
    Parameters
    ----------
    arr : numpy.ndarray
        The array to be copied from
    ctx : DLContext, optional
        The device context to create the array
    Returns
    -------
    ret : NDArray
        The created array
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=data_type)
    ret = empty(arr.shape, ctx)
    ret._sync_copyfrom(arr, data_type=data_type)
    return ret


def empty(shape, ctx=cpu(0)):
    """Create an empty array given shape and device
    Parameters
    ----------
    shape : tuple of int
        The shape of the array
    ctx : DLContext
        The context of the array
    Returns
    -------
    arr : ndarray
        The array hetusys supported.
    """
    shape = c_array(ctypes.c_int64, shape)
    stride = c_array(ctypes.c_int64, shape_to_stride(shape))
    ndim = ctypes.c_int(len(shape))
    handle = DLArrayHandle()
    check_call(_LIB.DLArrayAlloc(
        shape, stride, ndim, ctx, ctypes.byref(handle)))
    return NDArray(handle)


def numpyasdlarrayhandle(data):
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    arr = DLArray()
    shape = c_array(ctypes.c_int64, data.shape)
    arr.data = data.ctypes.data_as(ctypes.c_void_p)
    arr.shape = shape
    arr.stride = c_array(ctypes.c_int64, shape_to_stride(data.shape))
    arr.ndim = data.ndim
    arr.ctx = cpu(0)
    return arr


class ND_Sparse_Array(object):
    __slots__ = ["data", "row", "col", "nrow", "ncol", "lazy"]

    def __init__(self, data, row, col, nrow, ncol):
        self.data = data
        self.row = row
        self.col = col
        self.nrow = nrow
        self.ncol = ncol
        self.lazy = False

    @property
    def shape(self):
        """Shape of this array"""
        return tuple((self.nrow, self.ncol))


def sparse_array(values, indices, shape, ctx=cpu(0)):
    """Create an sparse array from source arrs.
    ----------
    values : numpy.ndarray
        The value array to be copied from
    indices : tuple(numpy.ndarray, numpy.ndarray)
        The index array to be copied from
    ctx : DLContext, optional
        The device context to create the array
    Returns
    -------
    ret : NDArray
        The created array
    """
    assert len(shape) == len(indices) == 2
    assert len(values) == len(indices[0]) == len(indices[1])
    assert isinstance(indices, tuple)
    mat = scipy.sparse.csr_matrix((values, indices), shape)
    values = mat.data
    rows = mat.indptr
    cols = mat.indices
    values_ret = empty(values.shape, ctx)
    values_ret._sync_copyfrom(values)
    row_ret = empty(rows.shape, ctx)
    row_ret._sync_copyfrom(rows, np.int32)
    col_ret = empty(cols.shape, ctx)
    col_ret._sync_copyfrom(cols, np.int32)
    return ND_Sparse_Array(values_ret, row_ret, col_ret, shape[0], shape[1])


class IndexedSlices(object):
    __slots__ = ["indices", "values", "dense_shape",
                 "deduplicated", "lazy", "to_dense_flag"]

    def __init__(self, indices=None, values=None, dense_shape=None):
        self.indices = indices
        self.values = values
        self.dense_shape = dense_shape
        self.deduplicated = False
        self.lazy = False
        self.to_dense_flag = False

    def get_dense_shape(self):
        assert self.dense_shape is not None
        return self.dense_shape

    def get_sparse_shape(self):
        assert isinstance(self.values, NDArray)
        return self.values.shape

    def update(self, indices, values, dense_shape):
        self.indices = indices
        self.values = values
        if self.dense_shape is not None:
            assert tuple(self.dense_shape) == tuple(dense_shape)
        else:
            self.dense_shape = dense_shape

    def deduplicate(self, stream):
        assert is_gpu_ctx(self.indices.ctx)
        np_indices = self.indices.asnumpy()
        unique_indices, inverse = np.unique(np_indices, return_inverse=True)
        indices_on_ctx = array(unique_indices, ctx=self.indices.ctx)
        self.indices = indices_on_ctx
        inverse_on_ctx = array(inverse, ctx=self.indices.ctx)
        new_value_shape = list(unique_indices.shape)
        new_value_shape.append(self.values.shape[-1])
        new_values = empty(new_value_shape, ctx=self.values.ctx)
        _LIB.DLGpuArraySet(new_values.handle, ctypes.c_float(
            0), stream.handle if stream else None)
        _LIB.DeduplicateIndexedSlices(
            self.values.handle, inverse_on_ctx.handle, new_values.handle, stream.handle if stream else None)
        self.values = new_values
        self.deduplicated = True

    def cpu_deduplicate(self):
        assert not is_gpu_ctx(self.indices.ctx)
        np_indices = self.indices.asnumpy()
        unique_indices, inverse = np.unique(np_indices, return_inverse=True)
        new_value_shape = list(unique_indices.shape)
        last_dim = self.values.shape[-1]
        new_value_shape.append(last_dim)
        new_values = np.zeros(new_value_shape).astype(np.float32)
        flatten_ind = np_indices.reshape(-1)
        flatten = self.values.asnumpy().reshape((-1, last_dim))
        for i, ind in enumerate(inverse):
            new_values[ind] += flatten[i]
        self.values = array(new_values, cpu(0))
        self.indices = array(unique_indices, cpu(0))
        self.deduplicated = True

    def free_deduplicate(self):
        if self.deduplicated:
            del self.indices
            del self.values
            self.indices = None
            self.values = None
            self.deduplicated = False

    def to_dense(self, stream):
        assert is_gpu_ctx(self.indices.ctx)
        np_indices = self.indices.asnumpy()
        indicies_all = np.arange(self.get_dense_shape()[0])
        indices_all_on_ctx = array(indicies_all, ctx=self.indices.ctx)
        new_value_shape = self.get_dense_shape()
        new_values = empty(new_value_shape, ctx=self.values.ctx)
        _LIB.DLGpuArraySet(new_values.handle, ctypes.c_float(
            0), stream.handle if stream else None)
        _LIB.IndexedSlices2Dense(self.values.handle, self.indices.handle,
                                 new_values.handle, stream.handle if stream else None)
        self.free_deduplicate()
        self.values = new_values
        self.indices = indices_all_on_ctx
        self.to_dense_flag = True

    def free_dense(self):
        if self.to_dense_flag:
            del self.indices
            del self.values
            self.indices = None
            self.values = None
            self.to_dense_flag = False

    def merge(self, node):
        assert isinstance(node, IndexedSlices)
        if self.indices and self.values:
            vocab_size = self.values.shape[-1]
            new_indices = array(np.concatenate([self.indices.asnumpy(
            ).reshape(-1), node.indices.asnumpy().reshape(-1)]), ctx=self.indices.ctx)
            new_values = array(np.concatenate([self.values.asnumpy(
            ).reshape(-1, vocab_size), node.values.asnumpy().reshape(-1, vocab_size)], axis=0), ctx=self.values.ctx)
            self.update(new_indices, new_values, node.dense_shape)
        else:
            self.update(node.indices, node.values, node.dense_shape)
