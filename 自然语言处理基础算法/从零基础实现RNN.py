import matplotlib as plt
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import dltools


batch_size, num_steps = 32, 35
train_iter, vocab = dltools.load_data_time_machine(batch_size=batch_size, num_steps=num_steps)

for x, y in train_iter:
    print(x, y)
    break

# print(x.shape)

# print(y.T.reshape(-1))
# print(vocab.idx_to_token)
# print(vocab.token_to_idx)

# onehot编码数据
a = F.one_hot(torch.tensor([0, 2]), num_classes=len(vocab))
X = torch.arange(10).reshape((2, 5))
x = F.one_hot(X.T, 28)
# print(x)


# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)

    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


params = get_params(28, 512, 'cuda:0')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def init_rnn_state(batch_size, num_hiddens, divice):

    # return (torch.zeros((batch_size, num_hiddens), device=device), )
    return (torch.zeros((batch_size, num_hiddens), device=device),)


# rnn主体结构
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H,  = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, )


# 包装成类
class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens

        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, dltools.try_gpu(), get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], dltools.try_gpu())
Y, new_state = net(X.to(dltools.try_gpu()), state)


# 预测
def predict(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1,1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

        for _ in range(num_preds):
            y, state = net(get_input(), state)
            outputs.append(int(y.argmax(dim=1).reshape(1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])


predict('time traveller', 10, net, vocab, dltools.try_gpu())

# 梯度裁剪
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# 训练
def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, dltools.Timer()
    metric = dltools.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shapep[0], device=device)
        else:
            # 梯度释放
            if isinstance(net, nn.Module) and isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                     s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.T.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        print(1, y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


# 组合在一起
def train(net, train_iter, vocab, lr, num_epoch, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = dltools.Animator(xlabel='epoch', ylabel='perlexity', legend=['train'], xlim=[10, num_epoch])
    