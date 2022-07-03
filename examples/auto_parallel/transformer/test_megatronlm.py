from models import HetuGPT2, GPT2Config
import hetu as ht
import numpy as np
import os
import argparse

# this file is DEPRECATED

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int,
                        default=2, help='batch size')
    parser.add_argument('--test-num', type=int, default=5,
                        help='number of test iterations')
    parser.add_argument('--single', action='store_true',
                        help='whether to test single device case')
    args = parser.parse_args()

    config = GPT2Config(
        batch_size=args.batch_size,
        vocab_size=50256,
        n_embd=768,
        n_layer=2,
        n_inner=None,
        n_positions=512,
        n_head=12,
        activation_function="relu",
        resid_pdrop=0.,
        embd_pdrop=0.,
        attn_pdrop=0.,
        layer_norm_epsilon=1e-12,
        initializer_range=0.02,
    )

    work_dir = 'test_strategy'

    def get_file(path):
        return os.path.join(work_dir, path)
    if args.single:
        os.makedirs(work_dir, exist_ok=True)
        in_ids = []
        at_masks = []
        tt_ids = []
        for i in range(args.test_num):
            length = np.random.randint(400, 512, size=args.batch_size)
            input_ids = np.random.randint(50256, size=(args.batch_size, 512))
            attention_mask = np.array([[x <= length[i] for x in range(
                512)] for i in range(args.batch_size)], dtype=int)
            token_type_ids = np.random.randint(
                50256, size=(args.batch_size, 512))
            in_ids.append(input_ids)
            at_masks.append(attention_mask)
            tt_ids.append(token_type_ids)
        np.save(get_file('input_ids.npy'), in_ids)
        np.save(get_file('attention_mask.npy'), at_masks)
        np.save(get_file('token_type_ids.npy'), tt_ids)
    else:
        in_ids = np.load(get_file('input_ids.npy'))
        at_masks = np.load(get_file('attention_mask.npy'))
        tt_ids = np.load(get_file('token_type_ids.npy'))

    # initialize hetu model
    lr = 1e-4
    if args.single:
        # TODO: divide the oneslike op in loss if we have multiple losses!
        # now for workaround, we only test 2 workers case
        # and ignore l2reg
        lr *= 2
    # opt = ht.optim.AdamOptimizer(
    #     learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg=0.01)
    opt = ht.optim.SGDOptimizer(
        learning_rate=lr)
    hetu_model = HetuGPT2(config)
    hetu_input_ids = ht.Variable(name='input_ids', trainable=False)
    hetu_token_type_ids = ht.Variable(name='token_type_ids', trainable=False)
    hetu_attention_mask = ht.Variable(name='attention_mask', trainable=False)

    hetu_loss, hetu_logits = hetu_model(
        hetu_input_ids, hetu_attention_mask, hetu_token_type_ids, hetu_input_ids)

    train_op = opt.minimize(hetu_loss)
    if args.single:
        executor = ht.Executor(
            [hetu_loss, hetu_logits, train_op], ctx=ht.gpu(0))
        executor.save(work_dir, 'gpt2.bin')
    else:
        # strategy = ht.dist.ModelParallel4LM()
        strategy = ht.dist.MegatronLM()
        executor = ht.Executor(
            [hetu_loss, hetu_logits, train_op], dist_strategy=strategy)
        executor.load(work_dir, 'gpt2.bin', consider_splits=True)

    # testing
    for i in range(args.test_num):
        # (batch_size=2, max_position=512)
        input_ids = in_ids[i]
        attention_mask = at_masks[i]
        token_type_ids = tt_ids[i]
        labels = input_ids

        hetu_feed_dict = {
            hetu_input_ids: input_ids.reshape((-1,)).astype(np.float32),
            hetu_token_type_ids: token_type_ids.reshape((-1,)).astype(np.float32),
            hetu_attention_mask: attention_mask.astype(np.float32),
        }

        loss_val, logits_val, _ = executor.run(
            feed_dict=hetu_feed_dict, convert_to_numpy_ret_vals=True)
        if args.single:
            np.save(get_file('gt_loss{}.npy'.format(i)), loss_val)
            np.save(get_file('gt_predict_y{}.npy'.format(i)), logits_val)
        else:

            # loss_val = executor.reduceMean(loss_val)
            # logits_val = executor.gatherPredict(logits_val)
            # split_logits_val = np.split(
            #     logits_val, executor.config.nrank, axis=0)
            # logits_val = np.concatenate(
            #     split_logits_val, axis=1)
            if executor.rank == 0:
                gt_loss_val = np.load(get_file('gt_loss{}.npy'.format(i)))
                gt_logits_val = np.load(
                    get_file('gt_predict_y{}.npy'.format(i)))
                np.testing.assert_allclose(
                    gt_loss_val, loss_val, rtol=1e-3)
                np.testing.assert_allclose(
                    gt_logits_val, logits_val, atol=0.1)
                print('Pass test with {}, {}'.format(gt_loss_val, loss_val))
