import hetu as ht
import numpy as np
from .layer import GCN, SageConv


def convert_to_one_hot(vals, max_val=0):
    """Helper method to convert label array to one-hot array."""
    if max_val == 0:
        max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals


def sparse_model(int_feature, hidden_layer_size, embedding_idx_max, embedding_width, num_classes, lr):
    y_ = ht.GNNDataLoaderOp(lambda g: ht.array(convert_to_one_hot(
        g.i_feat[:, -2], max_val=num_classes), ctx=ht.cpu()))
    mask_ = ht.Variable(name="mask_")
    index_ = ht.GNNDataLoaderOp(lambda g: ht.array(
        g.i_feat[:, 0:-2], ctx=ht.cpu()), ctx=ht.cpu())
    embedding = ht.init.random_normal(
        [embedding_idx_max, embedding_width], stddev=0.1)
    embed = ht.embedding_lookup_op(embedding, index_)
    feat = ht.array_reshape_op(embed, (-1, int_feature * embedding_width))

    norm_adj_ = ht.Variable("message_passing", trainable=False, value=None)
    gcn1 = GCN(int_feature * embedding_width,
               hidden_layer_size, norm_adj_, activation="relu")
    gcn2 = GCN(gcn1.output_width, num_classes, norm_adj_)
    x = gcn1(feat)
    y = gcn2(x)
    loss = ht.softmaxcrossentropy_op(y, y_)
    train_loss = loss * mask_
    train_loss = ht.reduce_mean_op(train_loss, [0])
    opt = ht.optim.SGDOptimizer(lr)
    train_op = opt.minimize(train_loss)
    # model input & model output
    return [loss, y, train_op], [mask_, norm_adj_]


def dense_model(feature_dim, hidden_layer_size, num_classes, lr, arch=GCN):
    y_ = ht.GNNDataLoaderOp(lambda g: ht.array(convert_to_one_hot(
        g.i_feat[:, -2], max_val=num_classes), ctx=ht.cpu()))
    mask_ = ht.Variable(name="mask_")
    feat = ht.GNNDataLoaderOp(lambda g: ht.array(
        g.f_feat, ctx=ht.cpu()), ctx=ht.cpu())

    norm_adj_ = ht.Variable("message_passing", trainable=False, value=None)
    gcn1 = arch(feature_dim, hidden_layer_size, norm_adj_, activation="relu")
    gcn2 = arch(gcn1.output_width, hidden_layer_size,
                norm_adj_, activation="relu")
    classifier = ht.init.xavier_uniform(shape=(gcn2.output_width, num_classes))
    x = gcn1(feat)
    x = gcn2(x)
    y = ht.matmul_op(x, classifier)
    loss = ht.softmaxcrossentropy_op(y, y_)
    train_loss = loss * mask_
    train_loss = ht.reduce_mean_op(train_loss, [0])
    opt = ht.optim.SGDOptimizer(lr)
    train_op = opt.minimize(train_loss)
    # model input & model output
    return [loss, y, train_op], [mask_, norm_adj_]
