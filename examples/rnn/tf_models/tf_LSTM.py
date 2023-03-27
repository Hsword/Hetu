import numpy as np
import tensorflow as tf


def tf_lstm(x, y_):
    '''
    LSTM model in TensorFlow, for MNIST dataset.

    Parameters:
        x: Variable(tensorflow.python.framework.ops.Tensor), shape (N, dims)
        y_: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
    Return:
        loss: Variable(tensorflow.python.framework.ops.Tensor), shape (1,)
        y: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
    '''

    print("Building LSTM model in tensorflow...")
    diminput = 28
    dimhidden = 128
    dimoutput = 10
    nsteps = 28

    forget_gate_w = tf.Variable(np.random.normal(
        scale=0.1, size=(diminput, dimhidden)).astype(np.float32))
    forget_gate_u = tf.Variable(np.random.normal(
        scale=0.1, size=(dimhidden, dimhidden)).astype(np.float32))
    forget_gate_b = tf.Variable(np.random.normal(
        scale=0.1, size=(dimhidden,)).astype(np.float32))
    input_gate_w = tf.Variable(np.random.normal(
        scale=0.1, size=(diminput, dimhidden)).astype(np.float32))
    input_gate_u = tf.Variable(np.random.normal(
        scale=0.1, size=(dimhidden, dimhidden)).astype(np.float32))
    input_gate_b = tf.Variable(np.random.normal(
        scale=0.1, size=(dimhidden,)).astype(np.float32))
    output_gate_w = tf.Variable(np.random.normal(
        scale=0.1, size=(diminput, dimhidden)).astype(np.float32))
    output_gate_u = tf.Variable(np.random.normal(
        scale=0.1, size=(dimhidden, dimhidden)).astype(np.float32))
    output_gate_b = tf.Variable(np.random.normal(
        scale=0.1, size=(dimhidden,)).astype(np.float32))
    tanh_w = tf.Variable(np.random.normal(
        scale=0.1, size=(diminput, dimhidden)).astype(np.float32))
    tanh_u = tf.Variable(np.random.normal(
        scale=0.1, size=(dimhidden, dimhidden)).astype(np.float32))
    tanh_b = tf.Variable(np.random.normal(
        scale=0.1, size=(dimhidden,)).astype(np.float32))
    out_weights = tf.Variable(np.random.normal(
        scale=0.1, size=(dimhidden, dimoutput)).astype(np.float32))
    out_bias = tf.Variable(np.random.normal(
        scale=0.1, size=(dimoutput,)).astype(np.float32))
    initial_state = tf.zeros((tf.shape(x)[0], dimhidden), dtype=tf.float32)

    last_c_state = initial_state
    last_h_state = initial_state

    for i in range(nsteps):
        cur_x = tf.slice(x, (0, i * diminput), (-1, diminput))
        # forget gate
        cur_forget = tf.matmul(last_h_state, forget_gate_u) + \
            tf.matmul(cur_x, forget_gate_w) + forget_gate_b
        cur_forget = tf.sigmoid(cur_forget)
        # input gate
        cur_input = tf.matmul(last_h_state, input_gate_u) + \
            tf.matmul(cur_x, input_gate_w) + input_gate_b
        cur_input = tf.sigmoid(cur_input)
        # output gate
        cur_output = tf.matmul(last_h_state, output_gate_u) + \
            tf.matmul(cur_x, output_gate_w) + output_gate_b
        cur_output = tf.sigmoid(cur_output)
        # tanh
        cur_tanh = tf.matmul(last_h_state, tanh_u) + \
            tf.matmul(cur_x, tanh_w) + tanh_b
        cur_tanh = tf.tanh(cur_tanh)

        last_c_state = last_c_state * cur_forget + cur_input * cur_tanh
        last_h_state = tf.tanh(last_c_state) * cur_output

    y = tf.matmul(last_h_state, out_weights) + out_bias
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = tf.reduce_mean(loss)
    return loss, y
