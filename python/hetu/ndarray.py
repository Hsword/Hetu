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


DataType = {
    np.float32: 0,
    np.int32: 1,
    np.uint32: 2,
    np.int8: 3,
    np.uint8: 4,
    np.int16: 5,
    np.uint16: 6,
}


def get_dtype(dtype):
    if isinstance(dtype, np.dtype):
        dtype = dtype.type
    return DataType[dtype]


class DLArray(ctypes.Structure):
    """DLArray in C API"""
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("ctx", DLContext),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("stride", ctypes.POINTER(ctypes.c_int64)),
        ("nbits", ctypes.c_int),
        ("dtype", ctypes.c_int8),
    ]


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


def get_nbits(dtype):
    if isinstance(dtype, np.dtype):
        dtype = dtype.type
    if dtype in (int, float):
        res = 32
    else:
        name = dtype.__name__
        if name.endswith('32'):
            res = 32
        elif name.endswith('16'):
            res = 16
        elif name.endswith('64'):
            res = 64
        else:
            res = int(name[-1])
    return res


class NDArray(object):
    """Lightweight NDArray class of DL runtime.
    Strictly this is only an Array Container(a buffer object)
    No arthimetic operations are defined.
    """
    __slots__ = ["handle", "no_free", "dtype"]

    def __init__(self, handle, dtype=np.float32, force32=True):
        """Initialize the function with handle
        Parameters
        ----------
        handle : DLArrayHandle
            the handle to the underlying C++ DLArray
        """
        self.handle = handle
        self.no_free = False
        if force32:
            self.dtype = convert_dtype(dtype)
        else:
            self.dtype = dtype

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
        arr.nbits = get_nbits(data.dtype)
        # CPU device
        arr.ctx = cpu(0)
        arr.dtype = get_dtype(data.dtype)
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
        arr.nbits = self.handle.contents.nbits
        arr.dtype = self.handle.contents.dtype
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
        arr.nbits = self.handle.contents.nbits
        arr.dtype = self.handle.contents.dtype
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
        arr.nbits = self.handle.contents.nbits
        arr.dtype = self.handle.contents.dtype
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
                                     self.handle.contents.ctx, ctypes.byref(handle), ctypes.c_int(get_nbits(self.dtype))))
        check_call(_LIB.DLGpuArrayLazyCallback(
            self.handle, handle, stream.handle if stream else None))
        handle.contents.dtype = get_dtype(self.dtype)
        self.handle = handle

    def wrapped_lazy_callback(self, stream=None):
        # TODO: reshape / copyto / asnumpy may have more efficient implementation
        # This is just a workaround.
        if self.lazy:
            # here we move the judgement for lazy into forward hooks, shouldn't have callbacks.
            assert False
            self.lazy_callback(stream)


def array(arr, ctx, dtype=np.float32, force32=True):
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
    ret = empty(arr.shape, ctx, dtype=dtype, force32=force32)
    ret._sync_copyfrom(arr, dtype=dtype)
    return ret


def empty(shape, ctx=cpu(0), dtype=np.float32, force32=True):
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
    if force32:
        dtype = convert_dtype(dtype)
    handle = DLArrayHandle()
    check_call(_LIB.DLArrayAlloc(
        shape, stride, ndim, ctx, ctypes.byref(handle), ctypes.c_int(get_nbits(dtype))))
    handle.contents.dtype = get_dtype(dtype)
    return NDArray(handle, dtype=dtype, force32=force32)


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
    arr.nbits = get_nbits(data.dtype)
    arr.dtype = get_dtype(data.dtype)
    arr.ctx = cpu(0)
    return arr


class ND_Sparse_Array(object):
    __slots__ = ["form", "data", "row", "col", "nrow", "ncol", "lazy", "ctx"]

    def __init__(self, nrow, ncol, data=None, row=None, col=None, form='csr', ctx=None):
        assert form in ('csr', 'coo')
        self.nrow = nrow
        self.ncol = ncol
        self.data = data
        self.row = row
        self.col = col
        self.form = form
        self.lazy = False
        if ctx is None:
            ctx = data.ctx
        self.ctx = ctx

    @property
    def shape(self):
        """Shape of this array"""
        return tuple((self.nrow, self.ncol))

    def get_copy(self, value):
        new_value = empty(value.shape, ctx=value.ctx,
                          dtype=value.dtype, force32=False)
        new_value[:] = value
        return new_value

    def get_copy_np(self, value, dtype):
        new_value = empty(value.shape, self.ctx, dtype=dtype, force32=False)
        new_value._sync_copyfrom(value, dtype=dtype)
        return new_value

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if (not isinstance(in_slice, slice) or
                in_slice.start is not None
                or in_slice.stop is not None):
            raise ValueError(
                'Sparse array only support set from entire sparse array')
        if isinstance(value, ND_Sparse_Array):
            self.nrow = value.nrow
            self.ncol = value.ncol
            self.ctx = value.ctx
            self.data = self.get_copy(value.data)
            self.row = self.get_copy(value.row)
            self.col = self.get_copy(value.col)
            self.form = value.form
        elif isinstance(value, scipy.sparse.csr.csr_matrix):
            self.data = self.get_copy_np(value.data, dtype=np.float32)
            self.row = self.get_copy_np(value.indptr, dtype=np.int32)
            self.col = self.get_copy_np(value.indices, dtype=np.int32)
            self.nrow, self.ncol = value.shape
            self.form = 'csr'
        elif isinstance(value, scipy.sparse.coo.coo_matrix):
            rec = np.rec.fromarrays([value.row, value.col, value.data])
            rec.sort()
            row, col, data = rec.f0, rec.f1, rec.f2
            self.data = self.get_copy_np(
                data.astype(np.float32), dtype=np.float32)
            self.row = self.get_copy_np(row.astype(np.int32), dtype=np.int32)
            self.col = self.get_copy_np(col.astype(np.int32), dtype=np.int32)
            self.nrow, self.ncol = value.shape
            self.form = 'coo'
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def asnumpy(self):
        if self.form == 'csr':
            result = scipy.sparse.csr_matrix(
                (self.data.asnumpy(), self.col.asnumpy(), self.row.asnumpy()),
                shape=(self.nrow, self.ncol))
        else:
            result = scipy.sparse.coo_matrix(
                (self.data.asnumpy(), (self.row.asnumpy(), self.col.asnumpy())),
                shape=(self.nrow, self.ncol))
        return result


def csr_sparse_array(mat, shape, ctx=cpu(0)):
    assert isinstance(mat, scipy.sparse.csr.csr_matrix) and len(shape) == 2
    new_array = ND_Sparse_Array(shape[0], shape[1], ctx=ctx)
    new_array[:] = mat
    return new_array


def coo_sparse_array(mat, shape, ctx=cpu(0)):
    assert isinstance(mat, scipy.sparse.coo.coo_matrix) and len(shape) == 2
    new_array = ND_Sparse_Array(shape[0], shape[1], ctx=ctx)
    new_array[:] = mat
    return new_array


def sparse_array(values, indices, shape, form='csr', ctx=cpu(0)):
    """Create an sparse array from source coo arrs.
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
    if form == 'csr':
        mat = scipy.sparse.csr_matrix((values, indices), shape)
        result = csr_sparse_array(mat, shape, ctx)
    else:
        mat = scipy.sparse.coo_matrix((values, indices), shape)
        result = coo_sparse_array(mat, shape, ctx)
    return result


def dense_to_sparse(arr, form='csr'):
    ctx = arr.ctx
    arr = arr.asnumpy()
    shape = arr.shape
    if form == 'csr':
        mat = scipy.sparse.csr_matrix(arr, shape)
        result = csr_sparse_array(mat, shape, ctx)
    else:
        mat = scipy.sparse.coo_matrix(arr, shape)
        result = coo_sparse_array(mat, shape, ctx)
    return result


class IndexedSlices(object):
    __slots__ = ["indices", "values", "dense_shape", "lazy", "dense_arr", ]

    def __init__(self, indices=None, values=None, dense_shape=None):
        self.indices = indices
        self.values = values
        self.dense_shape = dense_shape
        self.lazy = False
        self.dense_arr = None

    def get_dense_shape(self):
        assert self.dense_shape is not None
        return self.dense_shape

    def get_sparse_shape(self):
        assert isinstance(self.values, NDArray)
        return self.values.shape

    def to_dense(self, stream=None):
        self.try_init_dense()
        if is_gpu_ctx(self.indices.ctx):
            _LIB.DLGpuArraySet(self.dense_arr.handle, ctypes.c_float(
                0), stream.handle if stream else None)
            _LIB.IndexedSlices2Dense(self.values.handle, self.indices.handle,
                                     self.dense_arr.handle, stream.handle if stream else None)
        else:
            _LIB.cpu_IndexedSlices2Dense(
                self.indices.handle, self.values.handle, self.dense_arr.handle)
        return self.dense_arr

    def try_init_dense(self):
        shape = self.get_dense_shape()
        if self.dense_arr is None:
            self.dense_arr = empty(
                shape, ctx=self.values.ctx, dtype=self.values.dtype)

    def asnumpy(self):
        return (self.indices.asnumpy(), self.values.asnumpy())
