import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import torch_ncf
import config
import torch_data


parser = argparse.ArgumentParser()
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
                    default=True,
                    help="save model or not")
parser.add_argument("--gpu",
                    type=str,
                    default="0",
                    help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0


def metrics(model, test_loader, top_k):
    HR, NDCG = [], []

    for user, item, label in tqdm(test_loader, desc='test'):
        user = user.cuda()
        item = item.cuda()

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
            item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)


############################## PREPARE DATASET ##########################
train_data, test_data, user_num, item_num, train_mat = torch_data.load_all()

# construct the train and test datasets
train_dataset = torch_data.NCFData(
    train_data, item_num, train_mat, args.num_ng, True)
test_dataset = torch_data.NCFData(
    test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset,
                               batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,
                              batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
    assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
    assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
    GMF_model = torch.load(config.GMF_model_path)
    MLP_model = torch.load(config.MLP_model_path)
else:
    GMF_model = None
    MLP_model = None

model = torch_ncf.NCF(user_num, item_num, args.factor_num, args.num_layers,
                      args.dropout, config.model, GMF_model, MLP_model)
# with open('tcparam.pkl', 'rb') as fr:
#     state_dict = pickle.load(fr)
# model.load_state_dict(state_dict)
model.cuda()
loss_function = nn.BCEWithLogitsLoss()

if config.model == 'NeuMF-pre':
    sp_opt = None
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
    # pytorch implementation https://github.dev/pytorch/pytorch/blob/main/torch/optim/sparse_adam.py
    # in sparse adam, the order of bias correction and epsilon is not the same as dense
    # so the result is somewhat different from hetu
    sp_opt = optim.SparseAdam(
        [p for n, p in model.named_parameters() if n.startswith('embed_')], lr=args.lr)
    optimizer = optim.Adam([p for n, p in model.named_parameters(
    ) if not n.startswith('embed_')], lr=args.lr)

# writer = SummaryWriter() # for visualization

########################### TRAINING #####################################
count, best_hr = 0, 0
for epoch in range(args.epochs):
    model.train()  # Enable dropout (if have).
    start_time = time.time()
    train_loader.dataset.ng_sample()

    for user, item, label in tqdm(train_loader, desc='train'):
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()

        model.zero_grad()
        prediction = model(user, item)
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()
        if sp_opt is not None:
            sp_opt.step()
        # writer.add_scalar('data/loss', loss.item(), count)
        count += 1

    model.eval()
    HR, NDCG = metrics(model, test_loader, args.top_k)

    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
          time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

    if HR > best_hr:
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
        if args.out:
            if not os.path.exists(config.model_path):
                os.mkdir(config.model_path)
            torch.save(model,
                       '{}{}.pth'.format(config.model_path, config.model))

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
    best_epoch, best_hr, best_ndcg))
