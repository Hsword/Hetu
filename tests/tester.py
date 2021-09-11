import hetu as ht
import numpy as np


class HetuTester(object):
    def __init__(self, op, num_inputs, *args, **kargs):
        self.cpu_inputs = [ht.Variable(name='input%d' % i, ctx=ht.cpu())
                           for i in range(num_inputs)]
        self.cpu_op = op(*self.cpu_inputs, *args, **kargs)
        self.cpu_executor = ht.Executor([self.cpu_op], ctx=ht.cpu())
        self.gpu_inputs = [ht.Variable(name='input%d' % i, ctx=ht.gpu(0))
                           for i in range(num_inputs)]
        self.gpu_op = op(*self.gpu_inputs, *args, **kargs)
        self.gpu_executor = ht.Executor([self.gpu_op], ctx=ht.gpu(0))

    def test(self, input_shapes, rtol=1e-7, atol=0):
        input_vals = [np.random.normal(size=shape) for shape in input_shapes]
        cpu_result = self.cpu_executor.run(
            feed_dict={k: v for k, v in zip(self.cpu_inputs, input_vals)}, convert_to_numpy_ret_vals=True)
        gpu_result = self.gpu_executor.run(
            feed_dict={k: v for k, v in zip(self.gpu_inputs, input_vals)}, convert_to_numpy_ret_vals=True)
        np.testing.assert_allclose(
            cpu_result[0], gpu_result[0], rtol=rtol, atol=atol)
        print('Op %s pass the test with shapes: %s' %
              (self.gpu_op, input_shapes))
