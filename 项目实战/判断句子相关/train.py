import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from finetruning_net import Net
from transformers.optimization import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')


# data = pd.read_csv('../../data/train_data.csv', sep='\t', header=None)


class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = self.read_data()

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, item):
        sentence = self.dataset[0][item]
        label = self.dataset[1][item]
        return sentence, label

    def read_data(self):
        sentences = []
        labels = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for row in lines:
                row = row.strip().split('\t')
                sentence = (row[1], row[2])
                sentences.append(sentence)
                labels.append(row[0])
        return sentences, labels


data_path = '../../data/train_data.csv'
dataset = Dataset(data_path)


# 划分训练集，测试集
sentences, labels = dataset.dataset
x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2)
train_dataset = list(zip(x_train, y_train))
valid_dataset = list(zip(x_test, y_test))
# print(len(train_dataset))
# print(len(valid_dataset))


# print(train_dataset[2])
# print(len(train_dataset))


def collate_fn(data):
    sens = [i[0] for i in data]
    labels = [int(i[1]) for i in data]
    encode = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens,
                                         max_length=20,
                                         padding='max_length',
                                         truncation=True,
                                         return_tensors='pt',
                                         return_length=True)
    input_ids = encode['input_ids']
    attention_mask = encode['attention_mask']
    token_type_ids = encode['token_type_ids']
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=16,
                          collate_fn=collate_fn,
                          shuffle=True,
                          drop_last=True
                          )

for part in train_loader:
    break

# input_ids, attention_mask, token_type_ids, _ = part
# print(input_ids)

model = Net()
# x = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
# print(x.shape)
# print(x)
# print("===========================")
# print(torch.max(x, 1))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def train():
    optimizer = AdamW(model.parameters(), lr=2e-5, no_deprecation_warning=True)
    criterion = nn.CrossEntropyLoss()
    model.train()
    correct = 0
    total = 0
    train_loss = 0.0
    train_loader_iter = tqdm(train_loader)
    for train_data in train_loader_iter:
        input_ids, attention_mask, token_type_ids, labels = train_data
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(
                                                            device), token_type_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        outs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = criterion(outs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        correct += (outs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        train_loader_iter.set_postfix(loss=loss.item())

    accuracy = round(correct / total, 4)
    e_loss = round(train_loss/len(train_loader),  4)

    print(f"准确率:{accuracy}   损失为:{e_loss}")

    return accuracy, e_loss


def test():
    criterion = nn.CrossEntropyLoss()
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=16,
                              shuffle=True,
                              drop_last=True,
                              collate_fn=collate_fn)
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    valid_loader_iter = tqdm(valid_loader)
    for valid_data in valid_loader_iter:
        input_ids, attention_mask, token_type_ids, labels= valid_data
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
        with torch.no_grad():
            outs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outs, labels)
        correct += (outs.argmax(dim=-1) == labels).sum().item()
        val_loss += loss.item()
        total += labels.size(0)
    accuracy = round(correct/total, 4)
    e_loss = round(val_loss/len(valid_loader), 4)
    print(f'验证的准确率为{accuracy}  损失为:{e_loss}')
    return accuracy, e_loss


if __name__ == '__main__':
    best_val = 100
    epochs = 2
    train_accuracy_list = []
    valid_accuracy_list = []
    train_loss_list = []
    val_loss_list = []
    for epoch in range(epochs):
        print('Epoch:', epoch+1)
        accuracy, e_loss = train()
        train_loss_list.append(e_loss)
        train_accuracy_list.append(accuracy)

        accuracy_, e_loss_ = test()
        val_loss_list.append(e_loss_)
        train_accuracy_list.append(accuracy_)
        if e_loss_ < best_val:
            best_val = e_loss_
            torch.save(model.state_dict(), './bert_net.pth')
            # torch.save(model, './bert_net.model')

    # 画图
    plt.rcParams['font.family'] = 'KaiTi'
    plt.figure(0, figsize=(10, 8))
    x = [i for i in range(epochs)]
    plt.plot(x, train_loss_list, label='train_loss')
    plt.plot(x, val_loss_list, color='red', label='val_loss')
    plt.title('每一轮的损失')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('./loss.png')
    plt.legend(loc='upper right')

    plt.figure(1, figsize=(10, 8))
    plt.plot(x, train_accuracy_list, label='train_accuracy')
    plt.plot(x, valid_accuracy_list, color='red', label='val_accuracy')
    plt.title('每一轮的准确率')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='upper left')
    plt.savefig("./accuracy.png")
    plt.show()









