import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op
from tester import HetuTester


def test_add_const():
    tester = HetuTester(ht.addbyconst_op, 1, np.random.normal())
    tester.test([(2, 3, 4, 5, 6)])


def test_add_elewise():
    tester = HetuTester(ht.add_op, 2)
    tester.test([(2, 3, 4, 5), (2, 3, 4, 5)])
    tester.test([(3, 4, 1), (2, 3, 4, 5)])
    tester.test([(2, 3, 4, 5), (1, 5)])


def test_broadcast_to():
    tester = HetuTester(ht.broadcastto_op, 2)
    tester.test([(200, 300), (130, 200, 300)])
    tester.test([(3, 1), (2, 3, 5)])


def test_concatenate():
    def temp_op(*args, axis=0, **kargs):
        return ht.concatenate_op(list(args), axis=axis, **kargs)
    tester0 = HetuTester(temp_op, 3, axis=2)
    tester0.test([(2, 3, 4, 5), (2, 3, 7, 5), (2, 3, 6, 5)])
    tester0.test([(2, 3, 4), (2, 3, 7), (2, 3, 6)])
    tester1 = HetuTester(temp_op, 4, axis=1)
    tester1.test([(2, 1, 4, 5), (2, 2, 4, 5), (2, 3, 4, 5), (2, 4, 4, 5)])
    tester1.test([(2, 1), (2, 2), (2, 3), (2, 4)])
    tester2 = HetuTester(temp_op, 2, axis=0)
    tester2.test([(31, 4, 5), (23, 4, 5)])
    tester2.test([(31,), (23,)])


def test_concatenate_gradient():
    def temp_op(grad, input, axis, offset, ctx=None):
        res = ht.concatenate_gradient_op(grad, input, axis, ctx=ctx)
        res.offset = offset
        return res
    tester0 = HetuTester(temp_op, 2, axis=2, offset=13)
    tester0.test([(2, 3, 44, 5), (2, 3, 7, 5)])
    tester0.test([(2, 3, 24), (2, 3, 11)])
    tester1 = HetuTester(temp_op, 2, axis=1, offset=11)
    tester1.test([(2, 22, 4, 5), (2, 4, 4, 5)])
    tester1.test([(2, 18), (2, 1)])
    tester2 = HetuTester(temp_op, 2, axis=0, offset=7)
    tester2.test([(31, 4, 5), (23, 4, 5)])
    tester2.test([(31,), (23,)])


def test_sum():
    def temp_op(*args, **kargs):
        return ht.sum_op(list(args), **kargs)
    tester = HetuTester(temp_op, 3)
    tester.test([(2, 3, 4), (2, 3, 4), (2, 3, 4)])
    tester.test([(5, 7), (5, 7), (5, 7)])
    tester.test([(3, 4), (1,), (2, 1, 1)])


def test_ns_like_set():
    tester0 = HetuTester(ht.zeroslike_op, 1)
    tester0.test([(500, 200)])
    tester0.test([(2, 3, 7, 11)])
    tester1 = HetuTester(ht.oneslike_op, 1)
    tester1.test([(500, 200)])
    tester1.test([(2, 3, 7, 11)])


def test_linear():
    tester = HetuTester(ht.linear_op, 3)
    tester.test([(7, 9), (9, 11), (11,)], atol=1e-6)
    tester.test([(1, 13), (13, 2), (2,)], atol=1e-6)
    tester.test([(5, 1), (1, 3), (3,)], atol=1e-6)
    tester = HetuTester(ht.linear_op, 3, trans_A=True)
    tester.test([(9, 7), (9, 11), (11,)], atol=1e-6)
    tester.test([(13, 1), (13, 2), (2,)], atol=1e-6)
    tester.test([(1, 5), (1, 3), (3,)], atol=1e-6)
    tester = HetuTester(ht.linear_op, 3, trans_B=True)
    tester.test([(7, 9), (11, 9), (11,)], atol=1e-6)
    tester.test([(1, 13), (2, 13), (2,)], atol=1e-6)
    tester.test([(5, 1), (3, 1), (3,)], atol=1e-6)
    tester = HetuTester(ht.linear_op, 3, trans_A=True, trans_B=True)
    tester.test([(9, 7), (11, 9), (11,)], atol=1e-6)
    tester.test([(13, 1), (2, 13), (2,)], atol=1e-6)
    tester.test([(1, 5), (3, 1), (3,)], atol=1e-6)


def test_conv2d_add_bias():
    tester = HetuTester(ht.conv2d_add_bias_op, 3)
    tester.test([(4, 3, 28, 28), (7, 3, 5, 5), (7,)], atol=1e-5)
    tester.test([(2, 5, 13, 13), (3, 5, 3, 3), (3,)], atol=1e-5)


def test_tanh_gradient():
    tester = HetuTester(ht.tanh_gradient_op, 2)
    tester.test([(2, 3, 4), (2, 3, 4)], atol=3e-7)
    tester.test([(40, 50), (40, 50)], atol=3e-7)


def test_batch_norm_inference():
    tester = HetuTester(ht.batch_normalization_op, 3)
    # test training forward first
    for subexe in tester.cpu_executor.subexecutor.values():
        subexe.inference = False
    for subexe in tester.gpu_executor.subexecutor.values():
        subexe.inference = False
    tester.test([(32, 128, 16, 16), (128,), (128,)], atol=1e-5)
    tester.test([(32, 128, 16, 16), (128,), (128,)], atol=1e-5)
    tester.test([(32, 128, 16, 16), (128,), (128,)], atol=1e-5)
    # test inference next
    for subexe in tester.cpu_executor.subexecutor.values():
        subexe.inference = True
    for subexe in tester.gpu_executor.subexecutor.values():
        subexe.inference = True
    tester.test([(32, 128, 16, 16), (128,), (128,)], atol=5e-4)
    tester.test([(32, 128, 16, 16), (128,), (128,)], atol=5e-4)
    tester.test([(32, 128, 16, 16), (128,), (128,)], atol=5e-4)


def test_batch_norm_train():
    shape = (32, 128, 16, 16)
    channel = shape[1]
    param_shape = (channel,)
    input_shapes = (shape, shape, param_shape, param_shape)
    eps = 1e-5
    np.random.seed(123)

    # complex operator
    num_inputs = 4
    cpu_ctx = ht.cpu()
    cpu_inputs = [ht.Variable(name='input%d' % i, ctx=cpu_ctx)
                  for i in range(num_inputs)]
    cpu_forward_op = ht.batch_normalization_op(*cpu_inputs[1:])
    cpu_op = ht.batch_normalization_gradient_op(
        *cpu_inputs[:3], cpu_forward_op, eps)
    cpu_computing_ops = [
        cpu_forward_op,
        ht.batch_normalization_gradient_of_data_op(cpu_op, cpu_inputs[1]),
        ht.batch_normalization_gradient_of_scale_op(cpu_op, cpu_inputs[2]),
        ht.batch_normalization_gradient_of_bias_op(cpu_op, cpu_inputs[3])
    ]
    cpu_executor = ht.Executor(cpu_computing_ops, ctx=cpu_ctx)

    gpu_ctx = ht.gpu(0)
    gpu_inputs = [ht.Variable(name='input%d' % i, ctx=gpu_ctx)
                  for i in range(num_inputs)]
    gpu_forward_op = ht.batch_normalization_op(*gpu_inputs[1:])
    gpu_op = ht.batch_normalization_gradient_op(
        *gpu_inputs[:3], gpu_forward_op, eps)
    gpu_computing_ops = [
        gpu_forward_op,
        ht.batch_normalization_gradient_of_data_op(gpu_op, gpu_inputs[1]),
        ht.batch_normalization_gradient_of_scale_op(gpu_op, gpu_inputs[2]),
        ht.batch_normalization_gradient_of_bias_op(gpu_op, gpu_inputs[3])
    ]
    gpu_executor = ht.Executor(gpu_computing_ops, ctx=gpu_ctx)

    for subexe in cpu_executor.subexecutor.values():
        subexe.inference = False
    for subexe in gpu_executor.subexecutor.values():
        subexe.inference = False

    input_vals = [np.random.normal(size=shape) for shape in input_shapes]
    cpu_result = cpu_executor.run(
        feed_dict={k: v for k, v in zip(cpu_inputs, input_vals)}, convert_to_numpy_ret_vals=True)
    gpu_result = gpu_executor.run(
        feed_dict={k: v for k, v in zip(gpu_inputs, input_vals)}, convert_to_numpy_ret_vals=True)
    np.testing.assert_allclose(
        cpu_result[0], gpu_result[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        cpu_result[1], gpu_result[1], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        cpu_result[2], gpu_result[2], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        cpu_result[3], gpu_result[3], rtol=1e-4, atol=1e-4)
    print('Op BatchNormalizeGradientOp pass the test with shapes: {}'.format(input_shapes))


test_add_const()
test_add_elewise()
test_broadcast_to()
test_concatenate()
test_concatenate_gradient()
test_sum()
test_ns_like_set()
test_linear()
test_conv2d_add_bias()
test_tanh_gradient()
test_batch_norm_inference()
test_batch_norm_train()
