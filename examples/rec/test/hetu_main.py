import os
import os.path as osp
import time
import argparse
import numpy as np
from tqdm import tqdm

import hetu as ht
import hetu.layers as htl

import hetu_ncf
import hetu_data

cur_dir = osp.split(osp.abspath(__file__))[0]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    default='ml-1m',
                    help="dataset to be used",
                    choices=['ml-1m', 'pinterest-20'])
parser.add_argument("--model",
                    type=str,
                    default='NeuMF-end',
                    help="model to be used",
                    choices=['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre'])
parser.add_argument("--data_path",
                    type=str,
                    default=osp.join(cur_dir, './datasets'),
                    help="path to sampled data")
parser.add_argument("--model_path",
                    type=str,
                    default=osp.join(cur_dir, './models'),
                    help="path to model ckpts")
parser.add_argument("--lr",
                    type=float,
                    default=0.001,
                    help="learning rate")
parser.add_argument("--dropout",
                    type=float,
                    default=0.0,
                    help="dropout rate")
parser.add_argument("--batch_size",
                    type=int,
                    default=256,
                    help="batch size for training")
parser.add_argument("--epochs",
                    type=int,
                    default=20,
                    help="training epoches")
parser.add_argument("--top_k",
                    type=int,
                    default=10,
                    help="compute metrics@top_k")
parser.add_argument("--factor_num",
                    type=int,
                    default=32,
                    help="predictive factors numbers in the model")
parser.add_argument("--num_layers",
                    type=int,
                    default=3,
                    help="number of layers in MLP model")
parser.add_argument("--num_ng",
                    type=int,
                    default=4,
                    help="sample negative items for training")
parser.add_argument("--test_num_ng",
                    type=int,
                    default=99,
                    help="sample part of negative items for testing")
parser.add_argument("--out",
                    type=int,
                    default=0,
                    help="save model or not")
parser.add_argument("--ctx",
                    type=int,
                    default=0,
                    help="gpu card ID")
args = parser.parse_args()


args.train_rating = osp.join(args.data_path, f'{args.dataset}.train.rating')
args.test_rating = osp.join(args.data_path, f'{args.dataset}.test.rating')
args.test_negative = osp.join(args.data_path, f'{args.dataset}.test.negative')

args.GMF_model_path = osp.join(args.model_path, 'GMF.pth')
args.MLP_model_path = osp.join(args.model_path, 'MLP.pth')
args.NeuMF_model_path = osp.join(args.model_path, 'NeuMF.pth')


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0


def metrics(executor, test_loader, top_k):
    HR, NDCG = [], []

    for user, item, label in tqdm(test_loader, desc='test'):
        # user = user.numpy().astype(np.int32)
        # item = item.numpy().astype(np.int32)
        predictions = executor.run('validate', feed_dict={
                                   users: user, items: item})[0].asnumpy()
        indices = np.argsort(predictions)[::-1][:top_k]
        recommends = item[indices].tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)


############################## PREPARE DATASET ##########################
train_data, test_data, user_num, item_num, train_mat = hetu_data.load_all(args)

# construct the train and test datasets
train_dataset = hetu_data.NCFData(
    train_data, item_num, train_mat, args.num_ng, True)
test_dataset = hetu_data.NCFData(
    test_data, item_num, train_mat, 0, False)
train_loader = hetu_data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
test_loader = hetu_data.DataLoader(
    test_dataset, batch_size=args.test_num_ng+1, shuffle=False, drop_last=False)

########################### CREATE MODEL #################################
if args.model == 'NeuMF-pre':
    raise NotImplementedError
    assert os.path.exists(args.GMF_model_path), 'lack of GMF model'
    assert os.path.exists(args.MLP_model_path), 'lack of MLP model'
    GMF_model = torch.load(args.GMF_model_path)
    MLP_model = torch.load(args.MLP_model_path)
else:
    GMF_model = None
    MLP_model = None

model = hetu_ncf.NCF(user_num, item_num, args.factor_num,
                     args.num_layers, args.dropout, args.model)
loss_function = htl.BCEWithLogitsLoss()

if args.model == 'NeuMF-pre':
    optimizer = ht.optim.SGDOptimizer(learning_rate=args.lr)
else:
    optimizer = ht.optim.AdamOptimizer(learning_rate=args.lr)

users = ht.Variable(name='users', trainable=False, dtype=np.int32)
items = ht.Variable(name='items', trainable=False, dtype=np.int32)
labels = ht.Variable(name='labels', trainable=False, dtype=np.float32)
prediction = model(users, items)
loss = loss_function(prediction, labels)
train_op = optimizer.minimize(loss)
if args.ctx >= 0:
    ctx = ht.gpu(args.ctx)
else:
    ctx = ht.cpu()
executor = ht.Executor(
    {'train': [loss, prediction, train_op], 'validate': [prediction]}, ctx=ctx)

# with open('tcparam.pkl', 'rb') as fr:
#     state_dict = pickle.load(fr)
# mapping = {
#     'embed_user_GMF.weight': 'user_gmf',
#     'embed_item_GMF.weight': 'item_gmf',
#     'embed_user_MLP.weight': 'user_mlp',
#     'embed_item_MLP.weight': 'item_mlp',
#     'MLP_layers.1.weight': 'dnn_0_weight',
#     'MLP_layers.1.bias': 'dnn_0_bias',
#     'MLP_layers.4.weight': 'dnn_1_weight',
#     'MLP_layers.4.bias': 'dnn_1_bias',
#     'MLP_layers.7.weight': 'dnn_2_weight',
#     'MLP_layers.7.bias': 'dnn_2_bias',
#     'predict_layer.weight': 'pred_weight',
#     'predict_layer.bias': 'pred_bias',
# }
# new_state_dict = {}
# for k, v in state_dict.items():
#     newk = mapping[k]
#     newv = v.cpu().numpy()
#     if newk.endswith('_weight'):
#         newv = newv.transpose()
#     new_state_dict[newk] = newv
# executor.load_dict(new_state_dict)

########################### TRAINING #####################################
count, best_hr = 0, 0
for epoch in range(args.epochs):
    start_time = time.time()
    train_loader.dataset.ng_sample()

    for user, item, label in tqdm(train_loader, desc='train'):
        # results = executor.run('train', feed_dict={
        #     users: user.numpy().astype(np.int32), items: item.numpy().astype(np.int32), labels: label.numpy().astype(np.float32)})
        # lossvalue, predvalue = results[0].asnumpy(), results[1].asnumpy()
        executor.run('train', feed_dict={
            users: user, items: item, labels: label})
        count += 1

    HR, NDCG = metrics(executor, test_loader, args.top_k)

    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
          time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

    if HR > best_hr:
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
        if args.out:
            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            executor.save(args.model_path, f'{args.model}.pth')

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
    best_epoch, best_hr, best_ndcg))
