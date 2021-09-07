import hetu as ht


def MaxPool2d(kernel_size, stride, padding=0):
    return lambda x: ht.max_pool2d_op(
        x, kernel_size, kernel_size, padding, stride)


def AvgPool2d(kernel_size, stride, padding=0):
    return lambda x: ht.avg_pool2d_op(
        x, kernel_size, kernel_size, padding, stride)
