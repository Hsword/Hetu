""" ctypes library of hetusys and helper functions """
from __future__ import absolute_import

import os
import ctypes


def _check_functions(lib, func_dict):
    for func in func_dict:
        if hasattr(lib, func):
            func_dict[func] = True


# Defines a dictionary to indicate whether to use DNNL.True is use,False not use.
DNNL_LIB = {
    'DnnlMatrixMultiply': False,
    'DnnlMatrixElementwiseMultiplyByConst': False,
    'DnnlMatrixElementwiseMultiply': False,
    'DnnlMatrixElementwiseAddByConst': False,
    'DnnlMatrixElementwiseAdd': False,
    'DnnlMatrixElementwiseDivideByConst': False,
    'DnnlMatrixElementwiseDivide': False,
    'cpu_BroadcastTo': False,  # c++
    'cpu_ReduceSumAxisZero': False,
    'cpu_ArraySet': False,
    'cpu_Reshape': False,  # c++
    'DnnlSoftmax': False,
    'DnnlSoftmaxCrossEntropy': False,  # c++
    'DnnlSoftmaxCrossEntropy_Gradient': False,  # c++
    'DnnlSqrt': False,
    'DnnlReciprocalSqrt': False,
    'DnnlTanh': False,
    'DnnlOpposite': False,
    'DnnlSigmoid': False,
    'DnnlConv2d': False,
    'DnnlConv2d_Gradient_of_Filter': False,
    'DnnlConv2d_Gradient_of_Data': False,
    'DnnlAvgPool': False,
    'DnnlAvgPool_Gradient': False,
    'DnnlMaxPool': False,
    'DnnlMaxPool_Gradient': False,
    'DnnlRelu': False,
    'DnnlRelu_Gradient': False,
    'DnnlBatchNorm': False,
    'DnnlBatchNorm_Gradient': False,
    'DnnlBatchNorm_Inference': False,
    'DnnlConcat': False,
    'cpu_Concat_Gradient': False,  # c++
    'cpu_Dropout': False,  # c++
    'cpu_Dropout_Gradient': False,  # c++
    'cpu_Pad': False,  # c++
    'cpu_Pad_Gradient': False,  # c++
    'cpu_EmbeddingLookup': False,  # c++
    'cpu_Transpose': False,  # c++
    'cpu_IndexedSlices2Dense': False,
    'cpu_SGDOptimizerUpdate': False,  # c++
    'cpu_SGDOptimizerSparseUpdate': False,  # c++
    'cpu_SGDUpdateIndexedSlices': False,
    'cpu_MomentumOptimizerUpdate': False,  # c++
    'cpu_AdaGradOptimizerUpdate': False,  # c++
    'cpu_AdaGradOptimizerSparseUpdate': False,  # c++
    'cpu_AdaGradUpdateIndexedSlices': False,
    'cpu_BetatsUpdate': False,  # c++
    'cpu_AdamOptimizerUpdate': False,  # c++
    'cpu_AdamOptimizerSparseUpdate': False,
    'cpu_AdamUpdateIndexedSlices': False,
    'cpu_UniformInit': False,  # c++
    'cpu_NormalInit': False,  # c++
    'cpu_TruncatedNormalInit': False,  # c++
    'cpu_ReversedTruncatedNormalInit': False,  # c++
}


def _load_lib():
    """Load libary in build/lib."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, '../../build/lib/')
    path_to_so_file = os.path.join(lib_path, "libc_runtime_api.so")
    lib = ctypes.CDLL(path_to_so_file, ctypes.RTLD_GLOBAL)
    _check_functions(lib, DNNL_LIB)
    return lib


# global library instance
_LIB = _load_lib()


##################
# Helper Methods #
##################

def check_call(ret):
    """Check the return value of C API call

    This function will crash when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret : int
        return value from API calls
    """
    assert(ret == 0)


def c_array(ctype, values):
    """Create ctypes array from a python array

    Parameters
    ----------
    ctype : ctypes data type
        data type of the array we want to convert to

    values : tuple or list
        data content

    Returns
    -------
    out : ctypes array
        Created ctypes array
    """
    return (ctype * len(values))(*values)
