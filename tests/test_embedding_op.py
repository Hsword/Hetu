import numpy as np
import tensorflow as tf
import hetu as ht


def test_embedding(executor_ctx=ht.gpu(0)):
    embedding = ht.Variable('embeddingtable', value=np.random.rand(5, 5))
    index = ht.Variable(name="index")
    ids = [[0, 1], [0, 1]]
    ids = np.array(ids)
    ids = ht.array(ids, ctx=executor_ctx)
    y = ht.embedding_lookup_op(embedding, index)
    opt = ht.optim.SGDOptimizer(0.1)
    train_op = opt.minimize(y)
    executor = ht.Executor([y, train_op], ctx=executor_ctx)

    print("embedding:",
          executor.config.placeholder_to_arr_map[embedding].asnumpy())
    print("ids:", ids.asnumpy())
    out, _ = executor.run(feed_dict={index: ids})
    print(out.asnumpy())
    print(executor.config.placeholder_to_arr_map[embedding].asnumpy())


def test_embedding_with_tf(opt_name, iters=10000, executor_ctx=ht.gpu(0)):
    from time import time

    value = np.random.rand(5, 5)
    ids = [[0, 1], [0, 1]]
    ids = np.array(ids)

    # tf part
    tf_embedding = tf.Variable(value, dtype=tf.float32)
    tf_ids = tf.placeholder(tf.int32)
    tf_y = tf.nn.embedding_lookup(tf_embedding, tf_ids)
    tf_opts = {
        'sgd': tf.train.GradientDescentOptimizer(0.1),
        'momentum': tf.train.MomentumOptimizer(0.1, momentum=0.9),
        'nesterov': tf.train.MomentumOptimizer(0.1, momentum=0.9, use_nesterov=True),
        'adagrad': tf.train.AdagradOptimizer(0.1, initial_accumulator_value=1e-7, use_locking=True),
        'adam': tf.train.AdamOptimizer(0.1, epsilon=1e-7, use_locking=True),
    }
    tf_opt = tf_opts[opt_name]

    tf_trainop = tf_opt.minimize(tf_y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start = time()
        for i in range(iters):
            tf_out, _ = sess.run([tf_y, tf_trainop], feed_dict={tf_ids: ids})
        end = time()
        print('tensorflow time using: ', end - start)
        tf_new_embedding = sess.run([tf_embedding])[0]
        print(tf_out)
        print(tf_new_embedding)

    print()

    # hetu part
    embedding = ht.Variable('embeddingtable', value=value)
    index = ht.Variable(name="index")

    ids = ht.array(ids, ctx=executor_ctx)
    y = ht.embedding_lookup_op(embedding, index)
    hetu_opts = {
        'sgd': ht.optim.SGDOptimizer(0.1),
        'momentum': ht.optim.MomentumOptimizer(0.1),
        'nesterov': ht.optim.MomentumOptimizer(0.1, nesterov=True),
        'adagrad': ht.optim.AdaGradOptimizer(0.1),
        'adam': ht.optim.AdamOptimizer(0.1),
    }
    opt = hetu_opts[opt_name]

    train_op = opt.minimize(y)
    executor = ht.Executor([y, train_op], ctx=executor_ctx)

    start = time()
    for i in range(iters):
        out, _ = executor.run(feed_dict={index: ids})
    end = time()
    print('hetu time using: ', end - start)
    out = out.asnumpy()
    new_embedding = executor.config.placeholder_to_arr_map[embedding].asnumpy()
    print(out)
    print(new_embedding)

    np.testing.assert_allclose(out, tf_out, rtol=1e-5)
    np.testing.assert_allclose(new_embedding, tf_new_embedding, rtol=1e-5)

def test_embedding_with_torch(opt_name, iters=10000, executor_ctx=ht.gpu(0), l2reg = 0, lr=0.1):
    from time import time

    np.random.seed(123)

    value = np.random.rand(5, 5)
    ids = [[0,1],[2,0]]
    ids = np.array(ids,dtype=int)

    print("Old embedding:")
    print(value)
    print()

    label_np = np.random.randint(0,5,size=(2,2))

    # torch part
    from torch import nn
    import torch
    torch_embedding = nn.Embedding(5,5)
    torch_embedding.weight = torch.nn.Parameter(torch.Tensor(value))
    model = torch_embedding
    
    model.to('cuda:1')
    torch_opts={
        'sgd': torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2reg),
        'adam': torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=l2reg),
    }

    torch_opt = torch_opts[opt_name]

    for i in range(iters):
        torch_opt.zero_grad()
        torch_y = model(torch.LongTensor(ids).to('cuda:1'))
        loss = nn.CrossEntropyLoss()(torch_y.view(-1,5),torch.LongTensor(label_np).to('cuda:1').view(-1))
        loss.backward()
        torch_opt.step()

    if opt_name == 'adam':
        adam_state = torch_opt.state
        key = list(adam_state.keys())[0]
        torch_m = adam_state[key]['exp_avg'].detach().cpu().numpy()
        torch_v = adam_state[key]['exp_avg_sq'].detach().cpu().numpy()

    torch_out = torch_y.detach().cpu().numpy()
    torch_new_embedding = model.weight.detach().cpu().numpy()

    print("Pytorch:")
    print("Loss:",loss.item())
    #print("Output:\n",torch_out)
    print("New embedding:\n",torch_new_embedding)

    print("Weight gradient:\n",torch_embedding.weight.grad.detach().cpu().numpy())

    if opt_name == 'adam':
        print("torch_m:")
        print(torch_m)
        print("torch_v:")
        print(torch_v)

    print()

    # hetu part
    embedding = ht.Variable('embeddingtable', value=value)
    index = ht.Variable(name="index")
    y_ = ht.Variable(name='label')

    ids = ht.array(ids, ctx=executor_ctx)
    y = ht.embedding_lookup_op(embedding, index)
    loss = ht.softmaxcrossentropy_sparse_op(y,y_,ignored_index=-1)
    loss = ht.reduce_mean_op(loss, [0,1])
    hetu_opts = {
        'sgd': ht.optim.SGDOptimizer(lr,l2reg = l2reg),
        'adam': ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg = l2reg),
    }
    opt = hetu_opts[opt_name]

    train_op = opt.minimize(loss)
    executor = ht.Executor([y, loss, train_op], ctx=executor_ctx)

    for i in range(iters):
        out, loss_out,  _ = executor.run(feed_dict={index: ids, y_:label_np})

    out = out.asnumpy()
    loss_out = loss_out.asnumpy()
    new_embedding = executor.config.placeholder_to_arr_map[embedding].asnumpy()

    print("Hetu:")
    print("Loss:",loss_out)
    #print("Output:\n",out)
    print("New embedding:\n",new_embedding)

    # print("Weight gradient:")
    # for node in executor.computing_nodes:
    #     if node.name == 'EmbeddingLookUp_Gradient10':
    #         grad_hetu = executor.config.node_to_arr_map[node]
    #         print(grad_hetu.values.asnumpy())
    #         print(grad_hetu.indices.asnumpy())
    
    if opt_name == 'adam':
        print('hetu_m:')
        print(opt.m[0].asnumpy())
        print('hetu_v:')
        print(opt.v[0].asnumpy())

    #np.testing.assert_allclose(out, torch_out, rtol=1e-5)
    np.testing.assert_allclose(new_embedding, torch_new_embedding, rtol=1e-5)

test_embedding()
test_embedding(ht.cpu(0))
test_embedding_with_tf(opt_name='sgd')
test_embedding_with_tf(opt_name='sgd', executor_ctx=ht.cpu(0))
test_embedding_with_tf(opt_name='momentum')
test_embedding_with_tf(opt_name='nesterov', iters=1000)
test_embedding_with_tf(opt_name='adagrad')
test_embedding_with_tf(opt_name='adam')
test_embedding_with_torch(opt_name = 'adam', iters=1,l2reg=0.1,lr=0.1)
