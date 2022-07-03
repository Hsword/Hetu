from models import HetuBert, TorchBert, BertConfig
from load_data import DataLoader
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

    config = BertConfig(vocab_size=30522,
                        hidden_size=768,
                        num_hidden_layers=2,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        max_position_embeddings=512,
                        attention_probs_dropout_prob=0.0,
                        hidden_dropout_prob=0.0,
                        batch_size=args.batch_size)

    dataloader = DataLoader(dataset='bookcorpus', doc_num=200,
                            save_gap=200, batch_size=args.batch_size)

    # initialize hetu model
    opt = ht.optim.AdamOptimizer(
        learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg=0.01)
    hetu_model = HetuBert(config)
    input_ids = ht.Variable(name='input_ids', trainable=False)
    token_type_ids = ht.Variable(name='token_type_ids', trainable=False)
    attention_mask = ht.Variable(name='attention_mask', trainable=False)

    masked_lm_labels = ht.Variable(name='masked_lm_labels', trainable=False)
    next_sentence_label = ht.Variable(
        name='next_sentence_label', trainable=False)

    hetu_res0, hetu_res1, masked_lm_loss, next_sentence_loss = hetu_model(
        input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label)

    masked_lm_loss_mean = ht.reduce_mean_op(masked_lm_loss, [0, ])
    next_sentence_loss_mean = ht.reduce_mean_op(next_sentence_loss, [0])

    loss = masked_lm_loss_mean + next_sentence_loss_mean
    train_op = opt.minimize(loss)
    executor = ht.Executor(
        [hetu_res0, hetu_res1, loss, masked_lm_loss, next_sentence_loss, train_op], ctx=ht.gpu(0))

    # initialize torch model
    torch_model = TorchBert(config).cuda(1)
    torch_opt = torch.optim.Adam(
        torch_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    # synchronize parameters
    hetu_dict = dict()
    for k, v in torch_model.state_dict().items():
        if k == 'cls.predictions.decoder.weight':
            continue
        param_type = k.split('.')[-2:]
        hetu_name = k.replace('.', '_')
        hetu_value = v.cpu().detach().numpy()
        if param_type[1] == 'weight' and (param_type[0] in ("query", "key", "value", "dense", "decoder", "seq_relationship")):
            hetu_value = hetu_value.transpose()
        hetu_dict[hetu_name] = hetu_value
    executor.load_dict(hetu_dict)

    dataloader.make_epoch_data()
    # testing
    for i in range(args.test_num):
        batch_data = dataloader.get_batch(i)
        hetu_feed_dict = {
            input_ids: batch_data['input_ids'].reshape((-1,)),
            token_type_ids: batch_data['token_type_ids'].reshape((-1,)),
            attention_mask: batch_data['attention_mask'],
            masked_lm_labels: batch_data['masked_lm_labels'].reshape((-1,)),
            next_sentence_label: batch_data['next_sentence_label'],
        }

        results = executor.run(
            feed_dict=hetu_feed_dict, convert_to_numpy_ret_vals=True)
        loss_out = results[2]

        torch_opt.zero_grad()
        torch_feed_dict = {
            'input_ids': torch.LongTensor(batch_data['input_ids']).cuda(1),
            'token_type_ids': torch.LongTensor(batch_data['token_type_ids']).cuda(1),
            'attention_mask': torch.LongTensor(batch_data['attention_mask']).cuda(1),
            'masked_lm_labels': torch.LongTensor(batch_data['masked_lm_labels']).cuda(1),
            'next_sentence_label': torch.LongTensor(batch_data['next_sentence_label']).cuda(1),
        }
        torch_res0, torch_res1, torch_masked_lm_loss, torch_next_sentence_loss = torch_model(
            **torch_feed_dict)

        torch_loss = torch.mean(torch_masked_lm_loss) + torch.mean(torch_next_sentence_loss)
        torch_loss.backward()
        torch_opt.step()
        torch_loss_val = torch_loss.cpu().detach().numpy()

        np.testing.assert_allclose(results[0].reshape(
            2, -1, config.vocab_size), torch_res0.cpu().detach().numpy(), atol=1.2e-3*(i+1))
        np.testing.assert_allclose(
            results[1], torch_res1.cpu().detach().numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(results[3], torch_masked_lm_loss.cpu().detach().numpy(), rtol=1e-5)
        np.testing.assert_allclose(results[4], torch_next_sentence_loss.cpu().detach().numpy(), rtol=1.5e-5*(i+1))
        np.testing.assert_allclose(loss_out, torch_loss_val, rtol=1e-6*(i+1), atol=1e-6)
        print(loss_out, torch_loss_val)
