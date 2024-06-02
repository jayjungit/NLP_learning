import torch
import torch.nn as nn
import dltools
import time
#
#
# class Timer:
#     """Record multiple running times."""
#     def __init__(self):
#         self.times = []
#         self.start()
#
#     def start(self):
#         """Start the timer."""
#         self.tik = time.time()
#
#     def stop(self):
#         """Stop the timer and record the time in a list."""
#         self.times.append(time.time() - self.tik)
#         return self.times[-1]
#
#     def avg(self):
#         """Return the average time."""
#         return sum(self.times) / len(self.times)
#
#     def sum(self):
#         """Return the sum of time."""
#         return sum(self.times)
#
#     def cumsum(self):
#         """Return the accumulated time."""
#         return np.array(self.times).cumsum().tolist()
#
#
#
# class Accumulator:
#     """For accumulating sums over `n` variables."""
#     def __init__(self, n):
#         self.data = [0.0] * n
#
#     def add(self, *args):
#         for a, b in zip(self.data, args):
#             print("a:", a)
#             print("b:", b)
#             print([a + float(b)])
#         self.data = [a + float(b) for a, b in zip(self.data, args)]
#
#     def reset(self):
#         self.data = [0.0] * len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]
#
#
#
# class Encoder(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def forward(self, X, *args):
#         raise NotImplementedError
#
#
# class Decoder(nn.Module):
#     def __len__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def init_state(self, enc_outputs, *args):
#         raise NotImplementedError
#
#     def forward(self, X, state, *args):
#         raise NotImplementedError
#
#
# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder, **kwargs):
#         super().__init__(**kwargs)
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, enc_X, dec_X, *args):
#         enc_outputs = self.encoder(enc_X, *args)
#         dec_state = self.decoder.init_state(enc_outputs, *args)
#         return self.decoder(dec_X, dec_state)
#
#
# class Seq2SeqEncoder(Encoder):
#     def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
#         super().__init__(**kwargs)
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
#
#     def forward(self, X, *args):
#         # X 在sel.embedding之前的形状一般为（batch_size, num_steps, vocab_size)
#         X = self.embedding(X)  # X的形状为（ batch_size, num_steps,embed_size)
#         X = X.permute(1, 0, 2)  # rnn期望输入的形状为（num_steps, batch_size, embed_size)
#         output, state = self.rnn(X)
#         return output, state
#         # output的形状 (nun_steps, batch_size, nun_hiddens state的形状为 (nun_layers, batch_size, num_hiddens)
#
#
# seq2seqEncoder = Seq2SeqEncoder(10, 8, 16, 2)
# seq2seqEncoder.eval()
# X = torch.zeros((4, 7), dtype=torch.long)
# output, state = seq2seqEncoder(X)
# print(output.shape)
# print(len(state))
# print(state.shape)
# #
# #
# class Seq2SeqDecoder(Decoder):
#     def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
#         super().__init__(**kwargs)
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
#         self.dense = nn.Linear(num_hiddens, vocab_size)
#
#     def init_state(self, enc_outputs, *args):
#         return enc_outputs[1]
#
#     def forward(self, X, state, *args):
#         X = self.embedding(X)
#         X = X.permute(1, 0, 2)
#         context = state[-1].repeat(X.shape[0], 1, 1)
#         X_and_context = torch.cat((X, context), 2)
#         output, state = self.rnn(X_and_context, state)
#         output = self.dense(output).permute(1, 0, 2)
#         return output, state
#
#
# def sequence_mask(X, valid_len, value=0):
#     # 找到最大序列长度
#     maxlen = X.size(1)
#     mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None] < valid_len[:, None]
# #     print(mask)
#     X[~mask] = value
#     return X
#
#
#
# class MaskedSoftMaxCELoss(nn.CrossEntropyLoss):
#     def forward(self, pred, label, valid_len):
#         weights = torch.ones_like(label)
#         weights = sequence_mask(weights, valid_len)
#         self.reduction = 'none'
#         unweigthted_loss = super().forward(pred.permute(0, 2, 1), label)
#         weighted_loss = (weights * unweigthted_loss).mean(dim=1)
#         return weighted_loss
#
#
#
#
# def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
#     def xavier_init_weights(m):
#         if type(m) == nn.Linear:
#             nn.init.xavier_uniform_(m.weight)
#         if type(m) == nn.GRU:
#             for param in m._flat_weights_names:
#                 if 'weight' in param:
#                     nn.init.xavier_uniform_(m._parameters[param])
#
#     net.apply(xavier_init_weights)
#     net.to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#     loss = MaskedSoftMaxCELoss( )
#
#     for epoch in range(num_epochs):
#         metric = Accumulator(2)
#         timer = Timer()
#         for batch in data_iter:
#             optimizer.zero_grad()
#             X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
#             bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1,1)
#             # dec_input = torch.cat([bos, Y[:, :-1]], 1)
#             dec_input = torch.cat([bos, Y[:, :-1]], 1)
#             y_hat, _ = net(X, dec_input, X_valid_len)
#             l = loss(y_hat, Y, Y_valid_len)
#             l.sum().backward()
#             dltools.grad_clipping(net, 1)
#             num_tokens = Y_valid_len.sum()
#             optimizer.step()
#             with torch.no_grad():
#                 metric.add(l.sum(), num_tokens)
# #
#         print(f"loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop()} tokens/sec")
#
#
# embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
# batch_size, num_steps = 64, 10
# lr, num_epochs, device = 0.005, 100, dltools.try_gpu()
# train_iter, src_vocab, tgt_vocab = dltools.load_data_nmt(batch_size, num_steps)
#
# encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
# decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
# net = EncoderDecoder(encoder, decoder)
# train_seq2seq(net,train_iter, lr, num_epochs, tgt_vocab, device)
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, embedd_size, num_layers, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embedd = nn.Embedding(vocab_size, embedd_size)
        self.rnn = nn.GRU(embedd_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X):
        x = self.embedd(X)
        x = x.permute(1, 0, 2)
        output, state = self.rnn(x)
        return output, state


class Decoder(nn.Module):
    def __init__(self, tag_vocab, num_hiddens, embedd_size, num_layers, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed = nn.Embedding(tag_vocab, embedd_size)
        self.rnn = nn.GRU(embedd_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.fc = nn.Linear(num_hiddens, tag_vocab)

    def init_state(self, encod_outputs, *args):
        return encod_outputs[1]

    def forward(self, X, state, *args):
        x = self.embed(X)
        x = x.permute(1, 0, 2)
        context = state[-1].repeat(x.shape[0], 1, 1)
        x_context = torch.cat((x, context), 2)
        output, state = self.rnn(x_context)
        output = self.fc(output).permute(1, 0, 2)
        return output, state


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X, Y, *args):
        encod_output = self.encoder(X)
        decod_state = self.decoder.init_state(encod_output, *args)
        return self.decoder(Y, decod_state)


def sequence_mask(X, valid_len, value=0):
    # 找到最大序列长度
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None] < valid_len[:, None]
#     print(mask)
    X[~mask] = value
    return X

# 重写交叉熵损失, 添加屏蔽无效内容的部分.
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    # 重写forward
    # pred的形状: (batch_size, num_steps, vocab_size)
    # label的形状: (batch_size, num_steps)
    # valid_len的形状: (batch_size, )
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        # 先调用原始的交叉熵损失, 就可以计算没有被mask的损失.
        self.reduction = 'none'
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def train(net, train_iterm, lr, num_epochs, tgt_vocab, device):
    # 初始化
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr)
    loss = MaskedSoftmaxCELoss()
    for epoch in range(num_epochs):
        accumulator = dltools.Accumulator(2)
        for batch in train_iterm:
            X, X_valid_len, Y, Y_valid_len, = [x.to(device) for x in batch]
            optimizer.zero_grad()
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(y_hat, Y, Y_valid_len)
            l.sum().backward()
            dltools.grad_clipping(net, 1)
            num_tokens = len(Y_valid_len)
            optimizer.step()
            with torch.no_grad():
                accumulator.add(l.sum(), num_tokens)

        print(f'Epoch:{epoch + 1}  loss:{accumulator[0] / accumulator[1]:.3f}')


if __name__ == '__main__':
    epochs = 200
    batch_size = 64
    lr = 0.01
    num_steps = 10
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    embed_size, num_layers, num_hiddens, dropout = 32, 2, 32, 0.1
    train_iter, src_vocab, tag_vocab = dltools.load_data_nmt(batch_size, num_steps)
    encoder = Encoder(len(src_vocab),num_hiddens, embed_size, num_layers, dropout)
    decoder = Decoder(len(tag_vocab),num_hiddens, embed_size, num_layers, dropout)
    net = EncoderDecoder(encoder,decoder)
    train(net, train_iter, lr, epochs, tag_vocab, device)















