import hetu as ht
import torch
import numpy as np
import argparse


if __name__ == "__main__":
    # TODO: debug hetu bceloss
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        default=6, help='batch size')
    parser.add_argument('--test-num', type=int, default=10,
                        help='number of test iterations')
    args = parser.parse_args()

    # use synthetic data
    preds = [np.random.uniform(0, 1, size=(args.batch_size, 1)).astype(
        np.float32) for _ in range(args.test_num)]
    # make 1/3 of preds to 0 and 1/3 of preds to 1
    for j in range(args.test_num):
        for i in range(0, args.batch_size, 3):
            if i + 1 < args.batch_size:
                preds[j][i + 1] = 0
            if i + 2 < args.batch_size:
                preds[j][i + 2] = 1
    labels = [np.random.randint(0, 2, size=(args.batch_size, 1)).astype(
        np.int32) for _ in range(args.test_num)]
    labels = [np.zeros((args.batch_size, 1)) for _ in range(args.test_num)]
    # make 1/2 to 1
    for j in range(args.test_num):
        for i in range(args.batch_size // 2, args.batch_size):
            labels[j][i] = 1

    # initialize hetu model
    ctx = ht.gpu(0)
    hetu_pred = ht.placeholder_op(name='pred')
    hetu_label = ht.placeholder_op(name='label')
    loss = ht.binarycrossentropy_op(hetu_pred, hetu_label)
    # loss = ht.reduce_mean_op(loss, [0])
    # opt = ht.optim.SGDOptimizer(learning_rate=0.01)
    # train_op = opt.minimize(loss)
    executor = ht.Executor([loss], ctx=ctx)

    # initialize torch model
    torch_loss = torch.nn.BCELoss(reduction="none")

    # hetu_dict = {}
    # for k, v in torch_embed.state_dict().items():
    #     param_value = v.cpu().detach().numpy()
    #     hetu_dict[k] = param_value
    # for k, v in torch_model.state_dict().items():
    #     hetu_name = k.replace('.', '_')
    #     param_value = v.cpu().detach().numpy()
    #     if hetu_name.endswith('weight'):
    #         param_value = param_value.transpose()
    #     hetu_dict[hetu_name] = param_value
    # executor.load_dict(hetu_dict)

    # testing
    for i in range(args.test_num):
        pred = preds[i]
        label = labels[i]

        hetu_loss_val = executor.run(
            feed_dict={hetu_pred: pred, hetu_label: label.astype(np.float32)}, convert_to_numpy_ret_vals=True)
        hetu_loss_val = hetu_loss_val[0]

        torch_loss_val = torch_loss(
            torch.FloatTensor(pred), torch.FloatTensor(label))
        torch_loss_val = torch_loss_val.cpu().detach().numpy()

        # numpy
        eps = 1e-12
        flog = np.log(pred + eps)
        slog = np.log(1 - pred + eps)
        # boundary = np.log(eps)
        # flog = np.log(pred)
        # slog = np.log(1 - pred)
        # for i in range(len(flog)):
        #     if np.isnan(flog[i]) or np.isinf(flog[i]) or flog[i] < boundary:
        #         flog[i] = boundary
        #     if np.isnan(slog[i]) or np.isinf(slog[i]) or slog[i] < boundary:
        #         slog[i] = boundary
        numpy_loss = -label * flog - (1 - label) * slog
        # numpy_loss = np.mean(numpy_loss)

        print(pred.reshape(-1), label.reshape(-1))
        print('hetu', hetu_loss_val.reshape(-1))
        print('torch', torch_loss_val.reshape(-1))
        print('numpy', numpy_loss.reshape(-1))
        # np.testing.assert_allclose(hetu_loss_val, numpy_loss) #, rtol=1e-4)
        # np.testing.assert_allclose(numpy_loss, torch_loss_val) #, rtol=1e-4)
        np.testing.assert_allclose(
            hetu_loss_val, torch_loss_val)  # , rtol=1e-4)
