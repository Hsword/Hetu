import numpy as np
import tensorflow as tf
import time
import argparse
from tqdm import tqdm
from sklearn import metrics


def train_criteo(model, args):
    if args.all:
        from models.load_data import process_all_criteo_data
        dense, sparse, all_labels = process_all_criteo_data()
        dense_feature, val_dense = dense
        sparse_feature, val_sparse = sparse
        labels, val_labels = all_labels
    else:
        from models.load_data import process_sampled_criteo_data
        dense_feature, sparse_feature, labels = process_sampled_criteo_data()

    batch_size = 128
    dense_input = tf.compat.v1.placeholder(tf.float32, [batch_size, 13])
    sparse_input = tf.compat.v1.placeholder(tf.int32, [batch_size, 26])
    y_ = y_ = tf.compat.v1.placeholder(tf.float32, [batch_size, 1])

    loss, y, opt = model(dense_input, sparse_input, y_)
    train_op = opt.minimize(loss)

    init = tf.compat.v1.global_variables_initializer()
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    sess.run(init)

    my_feed_dict = {
        dense_input: np.empty(shape=(batch_size, 13)),
        sparse_input: np.empty(shape=(batch_size, 26)),
        y_: np.empty(shape=(batch_size, 1)),
    }

    if args.all:
        raw_log_file = './logs/tf_local_%s.log' % (args.model)
        print('Processing all data, log to', raw_log_file)
        log_file = open(raw_log_file, 'w')
        iterations = dense_feature.shape[0] // batch_size
        total_epoch = 11
        start_index = 0
        for ep in range(total_epoch):
            print("epoch %d" % ep)
            st_time = time.time()
            train_loss, train_acc, train_auc = [], [], []
            for it in tqdm(range(iterations // 10 + (ep % 10 == 9) * (iterations % 10))):
                my_feed_dict[dense_input][:] = dense_feature[start_index: start_index + batch_size]
                my_feed_dict[sparse_input][:] = sparse_feature[start_index: start_index + batch_size]
                my_feed_dict[y_][:] = labels[start_index: start_index+batch_size]
                start_index += batch_size
                if start_index + batch_size > dense_feature.shape[0]:
                    start_index = 0
                loss_val = sess.run([loss, y, y_, train_op],
                                    feed_dict=my_feed_dict)
                pred_val = loss_val[1]
                true_val = loss_val[2]
                acc_val = np.equal(
                    true_val,
                    pred_val > 0.5)
                train_loss.append(loss_val[0])
                train_acc.append(acc_val)
                train_auc.append(metrics.roc_auc_score(true_val, pred_val))
            tra_accuracy = np.mean(train_acc)
            tra_loss = np.mean(train_loss)
            tra_auc = np.mean(train_auc)
            en_time = time.time()
            train_time = en_time - st_time
            printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, train_time: %.4f"\
                % (tra_loss, tra_accuracy, tra_auc, train_time)
            print(printstr)
            log_file.write(printstr + '\n')
            log_file.flush()

    else:
        iteration = dense_feature.shape[0] // batch_size

        epoch = 50
        for ep in range(epoch):
            print('epoch', ep)
            if ep == 5:
                start = time.time()
            ep_st = time.time()
            train_loss = []
            train_acc = []
            for idx in range(iteration):
                start_index = idx * batch_size
                my_feed_dict[dense_input][:] = dense_feature[start_index: start_index + batch_size]
                my_feed_dict[sparse_input][:] = sparse_feature[start_index: start_index + batch_size]
                my_feed_dict[y_][:] = labels[start_index: start_index+batch_size]

                loss_val = sess.run([loss, y, y_, train_op],
                                    feed_dict=my_feed_dict)
                pred_val = loss_val[1]
                true_val = loss_val[2]
                if pred_val.shape[1] == 1:  # for criteo case
                    acc_val = np.equal(
                        true_val,
                        pred_val > 0.5)
                else:
                    acc_val = np.equal(
                        np.argmax(pred_val, 1),
                        np.argmax(true_val, 1)).astype(np.float32)
                train_loss.append(loss_val[0])
                train_acc.append(acc_val)
            tra_accuracy = np.mean(train_acc)
            tra_loss = np.mean(train_loss)
            ep_en = time.time()
            print("train_loss: %.4f, train_acc: %.4f, train_time: %.4f"
                  % (tra_loss, tra_accuracy, ep_en - ep_st))
        print('all time:', (time.time() - start))


def train_adult(model):
    batch_size = 128
    total_epoch = 50
    dim_wide = 809

    X_deep = []
    for i in range(8):
        X_deep.append(tf.compat.v1.placeholder(tf.int32, [batch_size, 1]))
    for i in range(4):
        X_deep.append(tf.compat.v1.placeholder(tf.float32, [batch_size, 1]))
    X_wide = tf.compat.v1.placeholder(tf.float32, [batch_size, dim_wide])
    y_ = tf.compat.v1.placeholder(tf.float32, [batch_size, 2])
    loss, y, train_op = model(X_deep, X_wide, y_)

    init = tf.global_variables_initializer()

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    sess.run(init)

    from models.load_data import load_adult_data
    x_train_deep, x_train_wide, y_train = load_adult_data(return_val=False)

    iterations = x_train_deep.shape[0] // batch_size
    for ep in range(total_epoch):
        print('epoch', ep)
        if ep == 5:
            start = time.time()
        ep_st = time.time()
        train_loss = []
        train_acc = []
        pre_index = 0

        for it in range(iterations):
            batch_x_deep = x_train_deep[pre_index:pre_index + batch_size]
            batch_x_wide = x_train_wide[pre_index:pre_index + batch_size]
            batch_y = y_train[pre_index:pre_index + batch_size]
            pre_index += batch_size

            my_feed_dict = dict()
            for i in range(12):
                my_feed_dict[X_deep[i]] = np.array(
                    batch_x_deep[:, 1]).reshape(-1, 1)

            my_feed_dict[X_wide] = np.array(batch_x_wide)
            my_feed_dict[y_] = batch_y
            loss_val = sess.run([loss, y, y_, train_op],
                                feed_dict=my_feed_dict)
            acc_val = np.equal(
                np.argmax(loss_val[1], 1),
                np.argmax(loss_val[2], 1)).astype(np.float32)
            train_loss.append(loss_val[0])
            train_acc.append(acc_val)
        tra_accuracy = np.mean(train_acc)
        tra_loss = np.mean(train_loss)
        ep_en = time.time()
        print("train_loss: %.4f, train_acc: %.4f, train_time: %.4f"
              % (tra_loss, tra_accuracy, ep_en - ep_st))
    print('all time:', (time.time() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="model to be tested")
    parser.add_argument("--all", action="store_true",
                        help="whether to use all data")
    args = parser.parse_args()
    raw_model = args.model
    import tf_models
    model = eval('tf_models.' + raw_model)
    dataset = raw_model.split('_')[-1]
    print('Model:', raw_model)

    if dataset == 'criteo':
        train_criteo(model, args)
    elif dataset == 'adult':
        train_adult(model)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
