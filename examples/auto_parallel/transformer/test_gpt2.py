from models import HetuGPT2, TorchGPT2, GPT2Config
import hetu as ht
import torch
import numpy as np
import argparse


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int,
                        default=2, help='batch size')
    parser.add_argument('--test-num', type=int, default=5,
                        help='number of test iterations')
    args = parser.parse_args()

    config = GPT2Config(
        batch_size=args.batch_size,
        vocab_size=50257,
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

    # initialize hetu model
    opt = ht.optim.AdamOptimizer(
        learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg=0.01)
    hetu_model = HetuGPT2(config)
    hetu_input_ids = ht.Variable(name='input_ids', trainable=False)
    hetu_attention_mask = ht.Variable(name='attention_mask', trainable=False)

    hetu_loss, hetu_logits = hetu_model(
        hetu_input_ids, hetu_attention_mask, hetu_input_ids)

    train_op = opt.minimize(hetu_loss)
    executor = ht.Executor(
        [hetu_loss, hetu_logits, train_op], ctx=ht.gpu(0))

    # initialize torch model
    torch_model = TorchGPT2(config).cuda(1)
    torch_opt = torch.optim.Adam(
        torch_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    # synchronize parameters
    hetu_dict = dict()
    for k, v in torch_model.state_dict().items():
        if k.startswith('lm_head'):
            continue
        if k[-9:] == 'attn.bias':
            continue
        if k[-16:] == 'attn.masked_bias':
            continue
        hetu_name = k.replace('.', '_')
        hetu_value = v.cpu().detach().numpy()
        hetu_dict[hetu_name] = hetu_value
    executor.load_dict(hetu_dict)

    # testing
    for i in range(args.test_num):
        # (batch_size=2, max_position=512)
        length = np.random.randint(400, 512, size=args.batch_size)
        input_ids = np.random.randint(50257, size=(args.batch_size, 512))
        attention_mask = np.array([[x <= length[i] for x in range(
            512)] for i in range(args.batch_size)], dtype=int)
        labels = input_ids

        hetu_feed_dict = {
            hetu_input_ids: input_ids.reshape((-1,)).astype(np.float32),
            hetu_attention_mask: attention_mask.astype(np.float32),
        }

        hetu_loss_val, hetu_logits_val, _ = executor.run(
            feed_dict=hetu_feed_dict, convert_to_numpy_ret_vals=True)

        torch_opt.zero_grad()
        torch_feed_dict = {
            'input_ids': torch.LongTensor(input_ids).cuda(1),
            'attention_mask': torch.LongTensor(attention_mask).cuda(1),
            'labels': torch.LongTensor(labels).cuda(1),
        }
        torch_loss, torch_logits = torch_model(**torch_feed_dict)

        torch_loss.backward()
        torch_opt.step()
        torch_loss_val = torch_loss.cpu().detach().numpy()
        print(hetu_loss_val, torch_loss_val)

        np.testing.assert_allclose(hetu_loss_val, torch_loss_val, rtol=1e-6)
        np.testing.assert_allclose(
            hetu_logits_val.reshape(args.batch_size, -1, config.vocab_size), torch_logits.cpu().detach().numpy(), atol=2e-3*(i+1))
