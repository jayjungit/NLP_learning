import os
import re
import collections


def read_time_machine():
    with open('../data/article.txt', 'r') as f:
        lines = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


lines = read_time_machine()
# print(f'text lines {len(lines)}')
# print(lines)
# print(lines[0])
# print(lines[10])

# line = lines[0]
# print(line)
# print(list(line))


def tokenize(lines_, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元的类型：' + token)


tokens = tokenize(lines, 'word')
# for i in range(11):
#     print(tokens[i])


def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens, list):
        tokens = [token for line in tokens for token in line]

        return collections.Counter(tokens)


# print(sorted(count_corpus(tokens).items(), key=lambda x: x[1], reverse=True))
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_token=None):
        if tokens is None:
            tokens = []
        if reserved_token is None:
            reserved_token = []

        counter = count_corpus(tokens).items()
        self.token_freqs = sorted(counter, key=lambda x: x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_token
        uniq_tokens += [token for token, freq in self.token_freqs if freq > min_freq and token not in uniq_tokens]
        self.id_to_token, self.token_to_id = [], dict()

        for token in uniq_tokens:
            self.id_to_token.append(token)
            self.token_to_id[token] = len(self.id_to_token) - 1

    def __len__(self):
        return len(self.id_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_id.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.id_to_token[indices]
        return [self.id_to_token[index] for index in indices]


vocab = Vocab(tokens)
# print(len(vocab))
# print(vocab.token_to_id)
# print(vocab.id_to_token)
# for i in [0, 10]:
#     print('word:', tokens[i])
#     print('indices', vocab[tokens[i]])


# 整合所有的功能
def load_corpus_time_machine(max_tokens=-1):
    line = read_time_machine()
    tokens = tokenize(lines, "word")
    vocab = Vocab(tokens)
    # 把所有的文本展平到一个列表
    corpus = [vocab[token] for line in lines for token in line]
    if max_tokens > 0:
        corpus = corpus[: max_tokens]
    return corpus, vocab


corpus, vocab = load_corpus_time_machine()
# print(vocab.token_freqs)

freqs = [freq for token, freq in vocab.token_freqs]
# print(freqs)
# 把所有文本行拼接在一起.
corpus = [token for line in tokens for token in line]
# 词汇表
vocab = Vocab(corpus)

# 二元语法 bi-grams
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
print(bigram_tokens)


