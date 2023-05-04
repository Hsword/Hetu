import pickle
import numpy as np

mapping = {
    'embed_user_GMF.weight': 'user_gmf',
    'embed_item_GMF.weight': 'item_gmf',
    'embed_user_MLP.weight': 'user_mlp',
    'embed_item_MLP.weight': 'item_mlp',
    'MLP_layers.1.weight': 'dnn_0_weight',
    'MLP_layers.1.bias': 'dnn_0_bias',
    'MLP_layers.4.weight': 'dnn_1_weight',
    'MLP_layers.4.bias': 'dnn_1_bias',
    'MLP_layers.7.weight': 'dnn_2_weight',
    'MLP_layers.7.bias': 'dnn_2_bias',
    'predict_layer.weight': 'pred_weight',
    'predict_layer.bias': 'pred_bias',
}


def load(f):
    with open(f, 'rb') as fr:
        items = pickle.load(fr)
    return items


def test(a, b, hard=True):
    if hard:
        np.testing.assert_allclose(a, b, atol=1e-6)
    else:
        try:
            np.testing.assert_allclose(a, b, atol=1e-6)
        except Exception as ex:
            print(ex)


def compare_prediction(tname, hname, hard=True):
    tparams = load(tname)
    hparams = load(hname)['state_dict']
    # print(tparams.keys(), len(tparams))
    # print(hparams.keys(), len(hparams))
    for k, v in tparams.items():
        print('testing ', k)
        hk = mapping[k]
        hv = hparams[hk]
        if hk.endswith('_weight'):
            v = v.transpose()
        test(v, hv, hard)


for i in range(3):
    print(i)
    hpred = load(f'hetu_{i}.pkl')
    tpred = load(f'torch_{i}.pkl')
    print('testing loss')
    test(hpred[1], tpred[1], False)
    print('testing pred')
    test(hpred[0], tpred[0], False)
    print('testing params')
    compare_prediction(f'torch_model_{i}.pkl', f'hetu_model_{i}.pkl', False)
