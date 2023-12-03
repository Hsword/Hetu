import numpy as np
import hetu as ht
from tester import HetuTester, HetuOptimizerTester


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


def test_reduce_mul():
    tester = HetuTester(ht.reduce_mul_op, 1, axes=[1], keepdims=True)
    tester.test([(7, 9)], rtol=1e-6)
    tester.test([(1, 13, 2, 4)], rtol=1e-6)
    tester.test([(5, 1)], rtol=1e-6)
    tester = HetuTester(ht.reduce_mul_op, 1, axes=[0, 2], keepdims=False)
    tester.test([(7, 9, 3, 4)], rtol=1e-6)
    tester.test([(1, 13, 2)], rtol=1e-6)
    tester.test([(2, 2, 2, 2)], rtol=1e-6)


def test_minus_elewise():
    tester = HetuTester(ht.minus_op, 2)
    tester.test([(2, 3, 4, 5), (2, 3, 4, 5)])
    tester.test([(3, 4, 1), (3, 4, 1)])
    tester.test([(2, 3), (2, 3)])


def test_reduce_min():
    tester = HetuTester(ht.reduce_mean_op, 1, axes=[
                        1, 3], keepdims=[True, False])
    tester.test([(1, 2, 3, 4, 5)], atol=3e-7)
    tester.test([(2, 3, 4, 5)], atol=3e-7)
    tester.test([(1, 2, 3, 4, 5, 6)], atol=3e-7)
    tester = HetuTester(ht.reduce_mean_op, 1, axes=[2], keepdims=False)
    tester.test([(1, 2, 3, 4, 5)], atol=3e-7)
    tester.test([(2, 3, 4, 5)], atol=3e-7)
    tester.test([(1, 2, 3, 4, 5, 6)], atol=3e-7)


def test_power():
    tester = HetuTester(ht.power_op, 1, p=2)
    tester.test([(1, 2, 3)], atol=1e-7)
    tester.test([(14, 6)], atol=1e-6)
    tester = HetuTester(ht.power_op, 1, p=0)
    tester.test([(1, 2, 3)], atol=1e-7)
    tester.test([(14, 6)], atol=1e-7)
    tester = HetuTester(ht.power_op, 1, p=1)
    tester.test([(1, 2, 3)], atol=1e-7)
    tester.test([(14, 6)], atol=1e-7)
    tester = HetuTester(ht.power_op, 1, in_dtype='uf', p=4.5)
    tester.test([(1, 2, 3)], rtol=1e-6)
    tester.test([(14, 6)], rtol=1e-6)


def test_tile():
    tester = HetuTester(ht.tile_op, 1, (2, 1, 3, 4))
    tester.test([(2, 3)])
    tester.test([(1, 2, 3)])
    tester.test([(2, 3, 2, 3)])
    tester.test([(2, 3, 2, 3, 4)])


def test_hash():
    tester = HetuTester(ht.mod_hash_op, 1, 97, in_dtype='ui')
    tester.test([(2, 3, 4)])
    tester = HetuTester(ht.mod_hash_negative_op, 1, 3, in_dtype='i')
    tester.test([(2, 3, 4, 5, 6)])
    tester = HetuTester(ht.div_hash_op, 1, 97, in_dtype='ui')
    tester.test([(2, 3, 4)])
    tester = HetuTester(ht.compo_hash_op, 1, 3, 97, in_dtype='ui')
    tester.test([(2, 3, 4)])
    tester = HetuTester(ht.learn_hash_op, 4, 1000, 'uniform', in_dtype='ui')
    tester.test([(2, 3, 4), (128,), (128,), (128,)], atol=1e-6)
    tester = HetuTester(ht.learn_hash_op, 4, 1000, 'normal', in_dtype='ui')
    tester.test([(2, 3, 4), (128,), (128,), (128,)], atol=1e-6)
    tester = HetuTester(ht.robe_hash_op, 2, 123, 33, 3, True, in_dtype='ui')
    tester.test([(13, 17), (10,)])
    tester = HetuTester(ht.robe_hash_op, 2, 101, 42, 6, False, in_dtype='ui')
    tester.test([(13, 17), (10,)])
    tester = HetuTester(ht.robe_sign_op, 2, 33, True, in_dtype='ui')
    tester.test([(13, 17), (10,)])
    tester = HetuTester(ht.robe_sign_op, 2, 42, False, in_dtype='ui')
    tester.test([(13, 17), (10,)])


def test_tril_lookup():
    tester = HetuTester(ht.tril_lookup_op, 1)
    tester.test([(2, 23, 23)])
    tester.test([(2, 83, 83)])
    tester.test([(17, 9, 9)])
    tester.test([(13, 19, 19)])
    tester.test([(7, 9, 6, 6)])
    tester = HetuTester(ht.tril_lookup_op, 1, -1)
    tester.test([(3, 3)])
    tester.test([(2, 23, 23)])
    tester.test([(2, 83, 83)])
    tester.test([(17, 9, 9)])
    tester.test([(13, 19, 19)])
    tester.test([(7, 9, 6, 6)])
    tester = HetuTester(ht.tril_lookup_op, 1, 1)
    tester.test([(2, 23, 23)])
    tester.test([(2, 83, 83)])
    tester.test([(17, 9, 9)])
    tester.test([(13, 19, 19)])
    tester.test([(7, 9, 6, 6)])
    tester = HetuTester(ht.tril_lookup_op, 1, 3)
    tester.test([(2, 23, 23)])
    tester.test([(2, 83, 83)])
    tester.test([(17, 9, 9)])
    tester.test([(13, 19, 19)])
    tester.test([(7, 9, 6, 6)])
    tester = HetuTester(ht.tril_lookup_op, 1, -2)
    tester.test([(2, 23, 23)])
    tester.test([(2, 83, 83)])
    tester.test([(17, 9, 9)])
    tester.test([(13, 19, 19)])
    tester.test([(7, 9, 6, 6)])


def test_tril_gradient_lookup():
    tester = HetuTester(ht.tril_lookup_gradient_op, 1)
    tester.test([(2, 276)])
    tester.test([(2, 3486)])
    tester.test([(17, 45)])
    tester.test([(13, 190)])
    tester.test([(7, 9, 21)])
    tester = HetuTester(ht.tril_lookup_gradient_op, 1, -1)
    tester.test([(2, 253)])
    tester.test([(2, 3403)])
    tester.test([(17, 36)])
    tester.test([(13, 171)])
    tester.test([(7, 9, 15)])
    tester = HetuTester(ht.tril_lookup_gradient_op, 1, 1)
    tester.test([(2, 298)])
    tester.test([(2, 3568)])
    tester.test([(17, 53)])
    tester.test([(13, 208)])
    tester.test([(7, 9, 26)])
    tester = HetuTester(ht.tril_lookup_gradient_op, 1, 3)
    tester.test([(2, 339)])
    tester.test([(2, 3729)])
    tester.test([(17, 66)])
    tester.test([(13, 241)])
    tester.test([(7, 9, 33)])
    tester = HetuTester(ht.tril_lookup_gradient_op, 1, -2)
    tester.test([(2, 231)])
    tester.test([(2, 3321)])
    tester.test([(17, 28)])
    tester.test([(13, 153)])
    tester.test([(7, 9, 10)])


def test_transpose():
    tester = HetuTester(ht.transpose_op, 1, (1, 0))
    tester.test([(3, 7)])
    tester.test([(89, 93)])
    tester = HetuTester(ht.transpose_op, 1, (2, 0, 3, 1))
    tester.test([(3, 7, 11, 13)])
    tester.test([(89, 93, 2, 3)])
    tester = HetuTester(ht.transpose_op, 1, (0, 2, 1))
    tester.test([(89, 93, 7)])
    tester.test([(1, 2, 3)])


def test_argmax():
    tester = HetuTester(ht.argmax_op, 1, 1)
    tester.test([(3, 4)])
    tester.test([(89, 93, 71)])
    tester = HetuTester(ht.argmax_op, 1, 0)
    tester.test([(3, 4, 5)])
    tester.test([(89, 93, 71, 13)])
    tester = HetuTester(ht.argmax_op, 1, 3)
    tester.test([(3, 4, 5, 5, 6)])
    tester.test([(89, 93, 71, 13, 1, 2)])


def test_log_softmax():
    tester = HetuTester(ht.log_softmax_op, 1)
    tester.test([(3, 4)], rtol=1e-6)
    tester.test([(89, 93, 71)], rtol=1e-6)
    tester = HetuTester(ht.log_softmax_gradient_op, 2)
    tester.test([(3, 4), (3, 4)], rtol=1e-6)
    tester.test([(8, 9, 7), (8, 9, 7)], atol=1e-5)


def test_softmax():
    tester = HetuTester(ht.softmax_op, 1)
    tester.test([(3, 4)], rtol=2e-6)
    tester.test([(89, 71)], rtol=2e-6)
    tester = HetuTester(ht.softmax_gradient_op, 2)
    tester.test([(3, 4), (3, 4)], rtol=2e-6)
    tester.test([(8, 7), (8, 7)], rtol=1e-5)


def test_exp():
    tester = HetuTester(ht.exp_op, 1)
    tester.test([(3, 4)], rtol=1e-6)
    tester.test([(89, 71)], rtol=1e-6)
    tester.test([(89, 71, 13, 101)], rtol=1e-6)


def test_argmax_partial():
    tester = HetuTester(ht.argmax_partial_op, 2, 3, 1)
    tester.test([(3, 4), (3,)])
    tester.test([(89, 93, 71), (89,)])
    tester = HetuTester(ht.argmax_partial_op, 2, 2, 2)
    tester.test([(3, 4, 5), (3,)])
    tester.test([(89, 93, 71, 13), (89,)])
    tester = HetuTester(ht.argmax_partial_op, 2, 4, 3)
    tester.test([(3, 4, 5, 5, 6), (3,)])
    tester.test([(89, 93, 71, 13, 1, 2), (89,)])


def test_abs():
    tester = HetuTester(ht.abs_op, 1)
    tester.test([(3, 4)])
    tester.test([(89, 93, 71)])


def test_sign():
    tester = HetuTester(ht.sign_op, 1)
    tester.test([(3, 4)])
    tester.test([(89, 93, 71)])


def test_mask():
    tester = HetuTester(ht.mask_op, 2, in_dtype=['f', 'ui'])
    tester.test([(3, 4), (3, 4)])
    tester.test([(89, 93, 71), (89, 93, 71)])


def test_log():
    tester = HetuTester(ht.log_op, 1, in_dtype='uf')
    tester.test([(3, 4)])
    tester.test([(89, 93, 71)], rtol=1e-6)


def test_sum_sparse_gradient():
    def build_op(*ops, is_sparses=None, dense_shape=None, dtype=None):
        pairs_or_denses = []
        cur_ind = 0
        for sp in is_sparses:
            if sp:
                pairs_or_denses.append((ops[cur_ind], ops[cur_ind + 1]))
                cur_ind += 2
            else:
                pairs_or_denses.append(ops[cur_ind])
                cur_ind += 1
        assert cur_ind == len(ops)
        return ht.sum_sparse_gradient_op(dense_shape, *pairs_or_denses, dtype=dtype)
    tester = HetuTester(build_op, 3, in_dtype=['f', 'ui', 'f'], is_sparses=[
                        False, True], dense_shape=(100, 7), dtype=np.float32)
    tester.test([(100, 7), (3, 4), (3, 4, 7)])
    tester = HetuTester(build_op, 7, in_dtype=['f', 'ui', 'f', 'f', 'f', 'ui', 'f'], is_sparses=[
                        False, True, False, False, True], dense_shape=(1000, 13), dtype=np.float32)
    tester.test([(1000, 13), (3, 4, 5), (3, 4, 5, 13),
                (1000, 13), (1000, 13), (17, ), (17, 13)])


def test_reduce_norm1():
    tester = HetuTester(ht.reduce_norm1_op, 1, axes=[1], keepdims=True)
    tester.test([(7, 9)], rtol=1e-6)
    tester.test([(1, 13, 2, 4)], rtol=1e-6)
    tester.test([(5, 1)], rtol=1e-6)
    tester = HetuTester(ht.reduce_norm1_op, 1, axes=[0, 2], keepdims=False)
    tester.test([(7, 9, 3, 4)], rtol=1e-6)
    tester.test([(1, 13, 2)], rtol=1e-6)
    tester.test([(2, 2, 2, 2)], rtol=1e-6)


def test_reduce_sum():
    tester = HetuTester(ht.reduce_sum_op, 1, axes=[1], keepdims=True)
    tester.test([(7, 9)], rtol=1e-6)
    tester.test([(1, 13, 2, 4)], rtol=1e-6)
    tester.test([(5, 1)], rtol=1e-6)
    tester = HetuTester(ht.reduce_sum_op, 1, axes=[0, 2], keepdims=False)
    tester.test([(7, 9, 3, 4)], rtol=1e-5)
    tester.test([(1, 13, 2)], rtol=1e-5)
    tester.test([(2, 2, 2, 2)], rtol=1e-5)


def test_reduce_norm2():
    tester = HetuTester(ht.reduce_norm2_op, 1, axes=[1], keepdims=True)
    tester.test([(7, 9)], rtol=1e-6)
    tester.test([(1, 13, 2, 4)], rtol=1e-6)
    tester.test([(5, 1)], rtol=1e-6)
    tester = HetuTester(ht.reduce_norm2_op, 1, axes=[0, 2], keepdims=False)
    tester.test([(7, 9, 3, 4)], rtol=1e-6)
    tester.test([(1, 13, 2)], rtol=1e-6)
    tester.test([(2, 2, 2, 2)], rtol=1e-6)


def test_binary_step():
    tester = HetuTester(ht.binary_step_op, 1)
    tester.test([(7, 9)], rtol=1e-6)
    tester.test([(1, 13, 2, 4)], rtol=1e-6)
    tester.test([(5, 1)], rtol=1e-6)
    tester = HetuTester(ht.binary_step_gradient_op, 1)
    tester.test([(7, 9, 3, 4)], rtol=1e-6)
    tester.test([(1, 13, 2)], rtol=1e-6)
    tester.test([(2, 2, 2, 2)], rtol=1e-6)


def test_bce_with_logits():
    tester = HetuTester(ht.binarycrossentropywithlogits_op, 2)
    tester.test([(7, 9), (7, 9)], rtol=1e-6)
    tester.test([(256,), (256,)], rtol=1e-6)


def test_bce_with_logits_gradient():
    tester = HetuTester(ht.binarycrossentropywithlogits_gradient_op, 3)
    tester.test([(7, 9), (7, 9), (7, 9)], rtol=1e-6)
    tester.test([(256,), (256,), (256,)], rtol=1e-6)


def test_div_handle_zero():
    tester = HetuTester(ht.div_handle_zero_op, 2)
    tester.test([(2, 3, 4, 5), (2, 3, 4, 5)])
    tester.test([(3, 4, 1), (3, 4, 1)])


def test_optimizers():
    test_shapes = [
        (1000, 8),
        (2, 3, 4),
        (4, 5, 6),
        (87, 91, 73),
    ]
    tests = {
        'sgd': True,
        'momentum': False,
        'nesterov': False,
        'adagrad': True,
        'adam': True,
        'amsgrad': True,
    }

    # np.random.seed(123)
    if tests['sgd']:
        tester = HetuOptimizerTester(
            ht.optim.SGDOptimizer(100), test_shapes)
        tester.test(rtol=1e-6, atol=1e-6)
        tester = HetuOptimizerTester(
            ht.optim.SGDOptimizer(100, 0.3), test_shapes)
        tester.test(iters=3, rtol=1e-3, atol=1e-3)
    if tests['momentum']:
        tester = HetuOptimizerTester(
            ht.optim.MomentumOptimizer(100), test_shapes)
        tester.test(rtol=1e-6, atol=1e-6)
        tester = HetuOptimizerTester(
            ht.optim.MomentumOptimizer(100, l2reg=0.3), test_shapes)
        tester.test(rtol=1e-4, atol=1e-4)
    if tests['nesterov']:
        tester = HetuOptimizerTester(
            ht.optim.MomentumOptimizer(100, nesterov=True), test_shapes)
        tester.test(rtol=1e-6, atol=1e-6)
        tester = HetuOptimizerTester(
            ht.optim.MomentumOptimizer(100, nesterov=True, l2reg=0.3), test_shapes)
        tester.test(rtol=1e-4, atol=1e-4)
    if tests['adagrad']:
        tester = HetuOptimizerTester(
            ht.optim.AdaGradOptimizer(100), test_shapes)
        tester.test(atol=1e-7)
        # tester = HetuOptimizerTester(
        #     ht.optim.AdaGradOptimizer(100, l2reg=0.001), test_shapes)
        # tester.test(rtol=1e-5, atol=2e-5)
    if tests['adam']:
        tester = HetuOptimizerTester(
            ht.optim.AdamOptimizer(100), test_shapes)
        tester.test(atol=1e-6, rtol=1e-6)
        # tester = HetuOptimizerTester(
        #     ht.optim.AdamOptimizer(100, l2reg=0.3), test_shapes)
        # tester.test(rtol=1e-4, atol=2e-5)
    if tests['amsgrad']:
        tester = HetuOptimizerTester(
            ht.optim.AdamOptimizer(100, amsgrad=True), test_shapes)
        tester.test(atol=1e-6, rtol=1e-6)
        # tester = HetuOptimizerTester(
        #     ht.optim.AdamOptimizer(100, l2reg=0.3, amsgrad=True), test_shapes)
        # tester.test(rtol=1e-3, atol=2e-5)


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
test_reduce_mul()
test_minus_elewise()
test_reduce_min()
test_power()
test_tile()
test_hash()
test_tril_lookup()
test_tril_gradient_lookup()
test_transpose()
test_argmax()
test_log_softmax()
test_softmax()
test_exp()
test_argmax_partial()
test_abs()
test_sign()
test_mask()
test_log()
test_sum_sparse_gradient()
test_reduce_norm1()
test_reduce_sum()
test_reduce_norm2()
test_binary_step()
test_bce_with_logits()
test_bce_with_logits_gradient()
test_div_handle_zero()
test_optimizers()
