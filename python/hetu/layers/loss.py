from .base import BaseLayer
import hetu as ht


class BaseLossLayer(BaseLayer):
    def __init__(self, reduction='mean'):
        assert reduction in ('mean', 'sum', 'none', None)
        self.reduction = reduction

    def reduce(self, loss):
        if self.reduction == 'mean':
            loss = ht.reduce_mean_op(loss, [0])
        elif self.reduction == 'sum':
            loss = ht.reduce_sum_op(loss, [0])
        return loss


class MSELoss(BaseLossLayer):
    def __call__(self, inputs, targets):
        loss = ht.power_op(ht.minus_op(inputs, targets), 2)
        return super().reduce(loss)


class MAELoss(BaseLossLayer):
    def __call__(self, inputs, targets):
        loss = ht.abs_op(ht.minus_op(inputs, targets))
        return super().reduce(loss)


class BCEWithLogitsLoss(BaseLossLayer):
    def __call__(self, inputs, targets):
        loss = ht.binarycrossentropywithlogits_op(inputs, targets)
        return super().reduce(loss)


# TODO: debug! how to handle 0 or 1 in inputs?
class BCELoss(BaseLossLayer):
    def __call__(self, inputs, targets):
        loss = ht.binarycrossentropy_op(inputs, targets)
        return super().reduce(loss)


class SoftmaxCrossEntropyLoss(BaseLossLayer):
    def __init__(self, reduction='mean', sparse=True):
        super().__init__(reduction)
        self.sparse = sparse

    def __call__(self, inputs, targets):
        if self.sparse:
            loss = ht.softmaxcrossentropy_sparse_op(inputs, targets)
        else:
            loss = ht.softmaxcrossentropy_op(inputs, targets)
        return super().reduce(loss)
