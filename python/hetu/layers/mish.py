from .base import BaseLayer
import hetu as ht


class Mish(BaseLayer):
    def __call__(self, x):
        return ht.mul_op(x, ht.tanh_op(ht.log_op(ht.addbyconst_op(ht.exp_op(x), 1))))
