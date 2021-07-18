import hetu as ht
from hetu import init
import numpy as np


def lstm(x, y_):
    '''
    LSTM model, for MNIST dataset.

    Parameters:
        x: Variable(hetu.gpu_ops.Node.Node), shape (N, dims)
        y_: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    Return:
        loss: Variable(hetu.gpu_ops.Node.Node), shape (1,)
        y: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    '''
    diminput = 28
    dimhidden = 128
    dimoutput = 10
    nsteps = 28

    forget_gate_w = init.random_normal(
        shape=(diminput, dimhidden), stddev=0.1, name="lstm_forget_gate_w")
    forget_gate_u = init.random_normal(
        shape=(dimhidden, dimhidden), stddev=0.1, name="lstm_forget_gate_u")
    forget_gate_b = init.random_normal(
        shape=(dimhidden,), stddev=0.1, name="lstm_forget_gate_b")
    input_gate_w = init.random_normal(
        shape=(diminput, dimhidden), stddev=0.1, name="lstm_input_gate_w")
    input_gate_u = init.random_normal(
        shape=(dimhidden, dimhidden), stddev=0.1, name="lstm_input_gate_u")
    input_gate_b = init.random_normal(
        shape=(dimhidden,), stddev=0.1, name="lstm_input_gate_b")
    output_gate_w = init.random_normal(
        shape=(diminput, dimhidden), stddev=0.1, name="lstm_output_gate_w")
    output_gate_u = init.random_normal(
        shape=(dimhidden, dimhidden), stddev=0.1, name="lstm_output_gate_u")
    output_gate_b = init.random_normal(
        shape=(dimhidden,), stddev=0.1, name="lstm_output_gate_b")
    tanh_w = init.random_normal(
        shape=(diminput, dimhidden), stddev=0.1, name="lstm_tanh_w")
    tanh_u = init.random_normal(
        shape=(dimhidden, dimhidden), stddev=0.1, name="lstm_tanh_u")
    tanh_b = init.random_normal(
        shape=(dimhidden,), stddev=0.1, name="lstm_tanh_b")
    out_weights = init.random_normal(
        shape=(dimhidden, dimoutput), stddev=0.1, name="lstm_out_weight")
    out_bias = init.random_normal(
        shape=(dimoutput,), stddev=0.1, name="lstm_out_bias")
    initial_state = ht.Variable(value=np.zeros((1,)).astype(
        np.float32), name='initial_state', trainable=False)

    for i in range(nsteps):
        cur_x = ht.slice_op(x, (0, i * diminput), (-1, diminput))
        # forget gate
        if i == 0:
            temp = ht.matmul_op(cur_x, forget_gate_w)
            last_c_state = ht.broadcastto_op(initial_state, temp)
            last_h_state = ht.broadcastto_op(initial_state, temp)
            cur_forget = ht.matmul_op(last_h_state, forget_gate_u) + temp
        else:
            cur_forget = ht.matmul_op(
                last_h_state, forget_gate_u) + ht.matmul_op(cur_x, forget_gate_w)
        cur_forget = cur_forget + ht.broadcastto_op(forget_gate_b, cur_forget)
        cur_forget = ht.sigmoid_op(cur_forget)
        # input gate
        cur_input = ht.matmul_op(
            last_h_state, input_gate_u) + ht.matmul_op(cur_x, input_gate_w)
        cur_input = cur_input + ht.broadcastto_op(input_gate_b, cur_input)
        cur_input = ht.sigmoid_op(cur_input)
        # output gate
        cur_output = ht.matmul_op(
            last_h_state, output_gate_u) + ht.matmul_op(cur_x, output_gate_w)
        cur_output = cur_output + ht.broadcastto_op(output_gate_b, cur_output)
        cur_output = ht.sigmoid_op(cur_output)
        # tanh
        cur_tanh = ht.matmul_op(last_h_state, tanh_u) + \
            ht.matmul_op(cur_x, tanh_w)
        cur_tanh = cur_tanh + ht.broadcastto_op(tanh_b, cur_tanh)
        cur_tanh = ht.tanh_op(cur_tanh)

        last_c_state = ht.mul_op(last_c_state, cur_forget) + \
            ht.mul_op(cur_input, cur_tanh)
        last_h_state = ht.tanh_op(last_c_state) * cur_output

    x = ht.matmul_op(last_h_state, out_weights)
    y = x + ht.broadcastto_op(out_bias, x)
    loss = ht.softmaxcrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    return loss, y
