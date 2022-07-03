import torch
import torch.nn.functional as F
import hetu as ht
from hetu import init
import numpy as np

inputs = []
np.random.seed(0)
torch.manual_seed(0)
for i in range(10):
    inputs.append(np.random.rand(128, 3, 32, 32))

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, bias=False, padding=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(16, momentum=0.5, eps=1e-5)
        self.fc = torch.nn.Linear(16, 100, bias=False)
        self.weight = self.conv1.weight.detach().numpy().copy()
        self.fc_weight = self.fc.weight.detach().numpy().T.copy()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = x.mean(axis=(2,3))
        x = self.fc(x)
        return x

torch_result = []
mlp = MLP()
opt = torch.optim.SGD(mlp.parameters(), lr=0.01)

for i in range(10):
    x = torch.Tensor(inputs[i])
    opt.zero_grad()
    result = mlp(x)
    loss = result.mean()
    loss.backward()
    opt.step()
    torch_result.append(result.detach().numpy())

def batch_norm(x, hidden):
    scale = init.ones(shape=(hidden,), name='scale')
    bias = init.zeros(shape=(hidden,), name='bias')
    x = ht.batch_normalization_op(x, scale, bias, momentum=0.5, eps=1e-5)
    return x

def conv2d(x, stride=1, padding=1):
    weight = ht.Variable(name='_weight', value=mlp.weight, ctx=ctx)
    x = ht.conv2d_op(x, weight, stride=stride, padding=padding)
    return x

def fc(x, dim):
    weight = ht.Variable(name='_weight', value=mlp.fc_weight, ctx=ctx)
    x = ht.matmul_op(x, weight)
    return x

hetu_result = []
ctx = ht.ndarray.gpu(0)
x_ = ht.placeholder_op(name='x', ctx=ctx)
x = conv2d(x_)
x = batch_norm(x, 16)
x = ht.relu_op(x)
# x = ht.avg_pool2d_op(x, kernel_H=32, kernel_W=32, padding=0, stride=1)
# x = ht.array_reshape_op(x, (-1, 16))
x = ht.reduce_mean_op(x, (2, 3))
x = fc(x, 100)
loss = ht.reduce_mean_op(x, [0, 1])
opt = ht.optim.SGDOptimizer(learning_rate=0.01)
train_op = opt.minimize(loss)
executor = ht.Executor([x, train_op], ctx=ctx)
for i in range(10):
    result, _ = executor.run(feed_dict={x_ : inputs[i]})
    hetu_result.append(result.asnumpy())

for i in range(10):
    diff = torch_result[i] - hetu_result[i]
    print(i, np.max(np.abs(diff)), np.mean(np.abs(torch_result[i])))
