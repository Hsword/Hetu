import numpy as np
import scipy.sparse
from hetu import ndarray
from hetu import gpu_links as gpu_op
from hetu import gpu_ops as autodiff


def test_sparse_matrix_multiply():
    density = 1e-3
    ctx = ndarray.gpu(0)
    x = scipy.sparse.rand(500, 7000, density=density,
                          format='coo', dtype=np.float32)
    y = np.random.uniform(0, 10, size=(7000, 100)).astype(np.float32)
    mat_x = ndarray.sparse_array(
        x.data, (x.row, x.col), shape=[500, 7000], ctx=ctx)
    mat_y = ndarray.array(y, ctx=ctx)
    mat_z = ndarray.empty((500, 100), ctx=ctx)
    gpu_op.CuSparse_Csrmm(mat_x, False, mat_y, False, mat_z)
    z = mat_z.asnumpy()
    np.testing.assert_allclose(x.dot(y), z, rtol=1e-5)

    # # following codes are invalid in cuda
    # density = 1e-3
    # ctx = ndarray.gpu(0)
    # x = scipy.sparse.rand(1000, 500 ,density=density,format='coo',dtype=np.float32)
    # y = np.random.uniform(0, 10, size=(2000, 500)).astype(np.float32)
    # mat_x = ndarray.sparse_array(x.data, (x.row, x.col), shape = [1000, 500], ctx=ctx)
    # mat_y = ndarray.array(y, ctx=ctx)
    # mat_z = ndarray.empty((1000, 2000), ctx=ctx)
    # gpu_op.CuSparse_Csrmm(mat_x, False, mat_y, True, mat_z)
    # z = mat_z.asnumpy()
    # np.testing.assert_allclose(x.dot(np.transpose(y)), z, rtol=1e-5)

    # x = scipy.sparse.rand(500, 1000, density=density,format='coo',dtype=np.float32)
    # y = np.random.uniform(0, 10, size=(2000, 500)).astype(np.float32)
    # mat_x = ndarray.sparse_array(x.data, (x.row, x.col), shape = [500, 1000], ctx=ctx)
    # mat_y = ndarray.array(y, ctx=ctx)
    # mat_z = ndarray.empty((1000, 2000), ctx=ctx)
    # gpu_op.CuSparse_Csrmm(mat_x, True, mat_y, True, mat_z)
    # z = mat_z.asnumpy()
    # np.testing.assert_allclose(x.T.dot(np.transpose(y)), z, rtol=1e-5)


def test_sparse_array_dense_vector_multiply():
    density = 1e-3
    ctx = ndarray.gpu(0)
    x = scipy.sparse.rand(500, 70000, density=density,
                          format='coo', dtype=np.float32)
    y = np.random.uniform(0, 10, size=(70000, 1)).astype(np.float32)
    mat_x = ndarray.sparse_array(
        x.data, (x.row, x.col), shape=[500, 70000], ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((500, 1), ctx=ctx)
    trans = False
    gpu_op.CuSparse_Csrmv(mat_x, trans, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x.dot(y), z, rtol=1e-5)

    x = scipy.sparse.rand(70000, 500, density=density,
                          format='coo', dtype=np.float32)
    y = np.random.uniform(0, 10, size=(70000, 1)).astype(np.float32)
    mat_x = ndarray.sparse_array(
        x.data, (x.row, x.col), shape=[70000, 500], ctx=ctx)
    arr_y = ndarray.array(y, ctx=ctx)
    arr_z = ndarray.empty((500, 1), ctx=ctx)
    trans = True
    gpu_op.CuSparse_Csrmv(mat_x, trans, arr_y, arr_z)
    z = arr_z.asnumpy()
    np.testing.assert_allclose(x.transpose().dot(y), z, rtol=1e-5)


test_sparse_matrix_multiply()
test_sparse_array_dense_vector_multiply()
