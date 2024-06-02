import torch
import torch.nn as nn
from transformers import GPT2Model, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
# from transformers.data.data_collator import default_data_collator
from transformers.data.data_collator import default_data_collator
from transformers.optimization import get_scheduler
import datasets
from transformers import AdamW

dataset = datasets.load_from_disk('../../data/sst')
# print(dataset['train'][0])
loader = DataLoader(dataset=dataset['train'],
                    batch_size=16,
                    collate_fn=default_data_collator,
                    shuffle=False,
                    drop_last=True,
                    )


for data in loader:
    break

# data['input_ids'][:, -1] = 0
# print(data['input_ids'])
# print(len(loader))
# print(data['input_ids'])
# print(data['labels'] == data['input_ids'])
tokenizer = AutoTokenizer.from_pretrained('distilgpt2', use_fast=True)
#
#
# # decode = tokenizer.batch_decode(data['input_ids'])
# # print(decode)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = GPT2Model.from_pretrained('distilgpt2')
        self.fc = nn.Linear(768, tokenizer.vocab_size, bias=False)

        parameters = AutoModelForCausalLM.from_pretrained('distilgpt2')
        self.fc.parameters(parameters.lm_head.state_dict())

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.pretrained(input_ids=input_ids, attention_mask=attention_mask)
        logits = logits.last_hidden_state
        logits = self.fc(logits)

        if labels is not None:
            labels = labels.long()
            shift_logits = logits[:, :-1].reshape(-1, tokenizer.vocab_size)
            shift_labels = labels[:, 1:]
            shift_labels = shift_labels.reshape(-1)
            loss = self.criterion(shift_logits, shift_labels)
        return {'loss': loss, 'logits': logits}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 简单写法 model = AutoModelForCausalLM.from_pretrained('distilgpt2')
        self.pretrained = GPT2Model.from_pretrained('distilgpt2')
        self.fc = torch.nn.Linear(768, tokenizer.vocab_size, bias=False)

        # 给fc这一层加载预训练权重
        parameters = AutoModelForCausalLM.from_pretrained('distilgpt2')
        self.fc.load_state_dict(parameters.lm_head.state_dict())

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        labels = labels.long()
        logits = self.pretrained(input_ids=input_ids, attention_mask=attention_mask)
        logits = logits.last_hidden_state
        logits = self.fc(logits)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, tokenizer.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = self.criterion(shift_logits, shift_labels)
        return {'loss': loss, 'logits': logits}


model = Model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# result = model(**data)
# print(result['loss'])
# print(result['logits'].shape)


def test(model):
    model.eval()
    test_loader = DataLoader(dataset=dataset['test'],
                             batch_size=16,
                             collate_fn=default_data_collator,
                             shuffle=True,
                             drop_last=True)
    correct = 0
    total = 0
    print(len(test_loader))
    for i, data in enumerate(test_loader):
        label = data['input_ids'][:, -1].clone()
        data['input_ids'][:, -1] = 0
        data['labels'][:, :] = 0
        with torch.no_grad():
            outs = model(**data)
        logit = outs['logits'].argmax(dim=2)[:, -2]
        # print(label)
        # print("=========================")
        # print(logit)
        correct += (label == logit).sum().item()
        total += 16
        if i % 10 == 0:
            print(i)
            print('预测为:', logit)
            print('真实为为:', label)
        if i == 50:
            break
    print('准确率为:', correct / total)

    for i in range(8):
        print(tokenizer.decode(data['input_ids'][i, :-1]))
        print("真实为：", tokenizer.decode(label[i]))
        print("预测为：", tokenizer.decode(logit[i]))


def train(epochs, model):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    scheduler = get_scheduler(name='linear',
                              num_warmup_steps=0,
                              num_training_steps=len(loader),
                              optimizer=optimizer)
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(loader):
            intput_ids, attention_mask, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['labels'].to(device)
            outs = model(intput_ids, attention_mask, labels)
            loss = outs['loss']
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            scheduler.step()
            optimizer.zero_grad()

            if i % 10 == 0:
                labels = labels[:, 1:]
                out = outs['logits'].argmax(dim=2)[:, :-1]
                correct = (labels == out).sum().item()
                accuracy = correct / (16*7)
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(i)
                print('损失为：', loss.item(), '准确率:', accuracy, 'lr:', lr)

    torch.save(model, './预测最后一个词.model')


# train(1, model)

model = torch.load('./预测最后一个词.model', map_location='cpu')
test(model)


