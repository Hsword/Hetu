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
        self_hostname = socket.gethostname()
        if hostname in ('localhost', self_hostname):
            self.hostname = self_hostname
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

    def full_repr(self):
        return "%s:%s:%d" % (
            self.hostname, DLContext.MASK2STR[self.device_type], self.device_id)

    def relocalize(self):
        self.local = (self.hostname in ('localhost', socket.gethostname()))

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


def convert_dtype(dtype):
    # now only support 4 bytes int and float
    if isinstance(dtype, np.dtype):
        dtype = dtype.type
    if dtype is int or issubclass(dtype, np.signedinteger):
        dtype = np.int32
    if dtype is float or issubclass(dtype, np.floating):
        dtype = np.float32
    if issubclass(dtype, np.unsignedinteger):
        dtype = np.uint32
    assert dtype in (np.int32, np.uint32, np.float32)
    return dtype


class NDArray(object):
    """Lightweight NDArray class of DL runtime.
    Strictly this is only an Array Container(a buffer object)
    No arthimetic operations are defined.
    """
    __slots__ = ["handle", "no_free", "dtype"]

    def __init__(self, handle, dtype=np.float32):
        """Initialize the function with handle
        Parameters
        ----------
        handle : DLArrayHandle
            the handle to the underlying C++ DLArray
        """
        self.handle = handle
        self.no_free = False
        self.dtype = convert_dtype(dtype)

    def __del__(self):
        if self.no_free:
            return
        check_call(_LIB.DLArrayFree(self.handle))

    def __repr__(self):
        return 'array{{shape={}, dtype={}, ctx={}}}'.format(self.shape, self.dtype.__name__, self.ctx)

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
                assert self.dtype == value.dtype
                value.copyto(self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self._sync_copyfrom(value, dtype=self.dtype)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def _sync_copyfrom(self, source_array, dtype=np.float32):
        """Peform an synchronize copy from the array.
        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.
        """
        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=dtype)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported'
                                % str(type(source_array)))
        source_array = np.ascontiguousarray(source_array, dtype=dtype)
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
        assert source_array.dtype == self.dtype
        check_call(_LIB.DLArrayCopyFromTo(
            source_array.handle, self.handle, stream_handle.handle))
        if not event_handle is None:
            check_call(_LIB.DLEventRecord(
                stream_handle.handle, event_handle.handle))

    def _async_copyfrom_offset(self, source_array, stream_handle, foffset, toffset, copy_size, event_handle=None):
        assert source_array.dtype == self.dtype
        check_call(_LIB.DLArrayCopyFromToOffset(
            source_array.handle, ctypes.c_size_t(4 * foffset), self.handle, ctypes.c_size_t(4 * toffset), ctypes.c_size_t(4 * copy_size), stream_handle.handle))
        if not event_handle is None:
            check_call(_LIB.DLEventRecord(
                stream_handle.handle, event_handle.handle))

    def async_h2d(self, source_array, stream_handle, event_handle=None):
        if isinstance(source_array, np.ndarray):
            source_array = array(source_array, cpu(0), dtype=self.dtype)
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
        np_arr = np.empty(self.shape, dtype=self.dtype)
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
        assert self.dtype == target.dtype
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


def array(arr, ctx, dtype=np.float32):
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
        arr = np.array(arr, dtype=dtype)
    ret = empty(arr.shape, ctx, dtype=dtype)
    ret._sync_copyfrom(arr, dtype=dtype)
    return ret


def empty(shape, ctx=cpu(0), dtype=np.float32):
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
    return NDArray(handle, dtype=dtype)


def empty_like(arr):
    return empty(arr.shape, arr.ctx, arr.dtype)


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
    row_ret = empty(rows.shape, ctx, dtype=np.int32)
    row_ret._sync_copyfrom(rows, np.int32)
    col_ret = empty(cols.shape, ctx, dtype=np.int32)
    col_ret._sync_copyfrom(cols, np.int32)
    return ND_Sparse_Array(values_ret, row_ret, col_ret, shape[0], shape[1])


class IndexedSlices(object):
    __slots__ = ["indices", "values", "dense_shape", "lazy",
                 "dedup_ind", "dedup_val", "dedup_args", "dense_arr", ]

    def __init__(self, indices=None, values=None, dense_shape=None):
        self.indices = indices
        self.values = values
        self.dense_shape = dense_shape
        self.lazy = False
        self.dense_arr = None
        self.dedup_args = None

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

    def deduplicate(self, stream=None):
        if is_gpu_ctx(self.indices.ctx):
            self.gpu_deduplicate(stream)
        else:
            self.cpu_deduplicate()

    def gpu_deduplicate(self, stream):
        assert is_gpu_ctx(self.indices.ctx)
        self.try_init_deduplicate(True)
        from .gpu_links import reduce_indexedslice
        reduce_indexedslice(self.indices, self.values, self.dedup_ind, self.dedup_val,
                            self.dedup_args['sp'], self.dedup_args['size'], self.dedup_args['eb'], stream)

    def cpu_deduplicate(self):
        assert not is_gpu_ctx(self.indices.ctx)
        self.try_init_deduplicate(False)
        from .cpu_links import reduce_indexedslice
        reduce_indexedslice(self.indices, self.values,
                            self.dedup_ind, self.dedup_val)

    def to_dense(self, stream=None):
        self.try_init_dense()
        self.deduplicate(stream)
        if is_gpu_ctx(self.indices.ctx):
            _LIB.DLGpuArraySet(self.dense_arr.handle, ctypes.c_float(
                0), stream.handle if stream else None)
            _LIB.IndexedSlices2Dense(self.dedup_val.handle, self.dedup_ind.handle,
                                     self.dense_arr.handle, stream.handle if stream else None)
        else:
            _LIB.cpu_IndexedSlices2Dense(
                self.dedup_ind.handle, self.dedup_val.handle, self.dense_arr.handle)
        return self.dense_arr

    def try_init_deduplicate(self, on_gpu):
        if self.dedup_args is None:
            if on_gpu:
                from .gpu_links import reduce_indexedslice_get_workspace_size
                ind_size = int(np.prod(self.indices.shape))
                ws_size = reduce_indexedslice_get_workspace_size(ind_size)
                all_ws_size = 2 * ind_size + 2 + (ws_size + 3) // 4
                self.dedup_args = {
                    'sp': empty((all_ws_size, ), ctx=self.indices.ctx),
                    'size': ws_size,
                    'eb': int(np.ceil(np.log2(self.get_dense_shape()[0]))),
                }
            else:
                self.dedup_args = {}
            self.dedup_ind = empty_like(self.indices)
            self.dedup_val = empty_like(self.values)

    def try_init_dense(self):
        shape = self.get_dense_shape()
        if self.dense_arr is None:
            self.dense_arr = empty(
                shape, ctx=self.values.ctx, dtype=self.values.dtype)
