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


test_add_const()
test_add_elewise()
test_broadcast_to()
test_concatenate()
test_concatenate_gradient()
test_sum()
test_ns_like_set()
test_linear()
