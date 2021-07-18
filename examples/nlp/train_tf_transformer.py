import tensorflow as tf

from tqdm import tqdm
import os
import math
import logging
from hparams import Hparams
from tf_transformer import Transformer
from data_load import DataLoader
# import time

logging.basicConfig(level=logging.INFO)


logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
print(hp)
# save_hparams(hp, hp.logdir)

logging.info("# Prepare train/eval batches")
dataloader = DataLoader(hp.train1, hp.train2, hp.maxlen1, hp.maxlen2, hp.vocab)

xs = tf.placeholder(name='xs', dtype=tf.int32, shape=[16, 100])
ys1 = tf.placeholder(name='ys1', dtype=tf.int32, shape=[16, 99])
ys2 = tf.placeholder(name='ys2', dtype=tf.int32, shape=[16, 99])

logging.info("# Load model")
m = Transformer(hp)
loss = m.train(xs, (ys1, ys2))
nonpadding = tf.to_float(tf.not_equal(ys2, dataloader.get_pad()))  # 0: <pad>
loss = tf.reduce_sum(loss * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

global_step = tf.train.get_or_create_global_step()
optimizer = tf.train.GradientDescentOptimizer(hp.lr)
train_op = optimizer.minimize(loss, global_step=global_step)
# y_hat, eval_summaries = m.eval(xs, ys)
# y_hat = m.infer(xs, ys)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        # save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    _gs = sess.run(global_step)

    for ep in range(hp.num_epochs):
        dataloader.make_epoch_data(hp.batch_size)
        for i in tqdm(range(dataloader.batch_num)):
            xs_val, ys_val = dataloader.get_batch()
            # st = time.time()
            _loss, _, _gs = sess.run([loss, train_op, global_step], feed_dict={
                                     xs: xs_val[0], ys1: ys_val[0][:, :-1], ys2: ys_val[0][:, 1:]})
            # en = time.time()
            # if i == 100:
            #     exit()
            # epoch = math.ceil(_gs / num_train_batches)

            log_str = 'Iteration %d, loss %f' % (i, _loss)
            print(log_str)
            # print('time: ', (en - st))

        # logging.info("epoch {} is done".format(ep))
        # _loss = sess.run(loss) # train loss

        # logging.info("# test evaluation")
        # _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
        # summary_writer.add_summary(_eval_summaries, _gs)

        # logging.info("# get hypotheses")
        # hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)

        # logging.info("# write results")
        # model_output = "iwslt2016_E%02dL%.2f" % (epoch, _loss)
        # if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)
        # translation = os.path.join(hp.evaldir, model_output)
        # with open(translation, 'w') as fout:
        #     fout.write("\n".join(hypotheses))

        # logging.info("# calc bleu score and append it to translation")
        # calc_bleu(hp.eval3, translation)

        # logging.info("# save models")
        # ckpt_name = os.path.join(hp.logdir, model_output)
        # saver.save(sess, ckpt_name, global_step=_gs)
        # logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

        # logging.info("# fall back to train mode")


logging.info("Done")
