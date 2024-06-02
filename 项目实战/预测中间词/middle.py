import datasets
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
import torch
import torch.nn as nn
from transformers import AdamW
from transformers.optimization import get_scheduler

tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
# print(tokenizer)

# encode = tokenizer.batch_encode_plus(
#     ['UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files',
#      'but your machine does not support them in'])
# print(tokenizer.get_vocab()['<mask>'])

dataset = datasets.load_from_disk('../../data/sst3')
# print(dataset['train'][0])
loader = DataLoader(dataset=dataset['train'],
                    batch_size=8,
                    collate_fn=default_data_collator,
                    drop_last=True,
                    shuffle=True)
for data in loader:
    break


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = AutoModel.from_pretrained('distilroberta-base')
        decoder = nn.Linear(768, tokenizer.vocab_size)
        decoder.bias = nn.Parameter(torch.zeros(tokenizer.vocab_size))

        self.fc = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768, eps=1e-5),
            decoder
        )

        parameters = AutoModelForCausalLM.from_pretrained('distilroberta-base')
        self.fc[0].load_state_dict(parameters.lm_head.dense.state_dict())
        self.fc[2].load_state_dict(parameters.lm_head.layer_norm.state_dict())
        self.fc[3].load_state_dict(parameters.lm_head.decoder.state_dict())

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.pretrained(input_ids=input_ids, attention_mask=attention_mask)
        logits = logits.last_hidden_state
        logits = self.fc(logits)

        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1].reshape(-1, tokenizer.vocab_size)
            shifted_labels = labels[:, 1:].reshape(-1)
            loss = self.criterion(shifted_logits, shifted_labels)

        return {'logits': logits, 'loss': loss}


model = Model()
# outs = model(**data)
# print(outs['logits'].shape)
# print(outs['loss'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def train():
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_scheduler(name='linear',
                              num_warmup_steps=0,
                              num_training_steps=len(loader),
                              optimizer=optimizer)
    for i, data in enumerate(loader):
        input_ids, attention_mask, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['labels'].to(device)
        outs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outs['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1)
        scheduler.step()
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

        if i % 50 == 0:
            label = data['labels'][:, 4].to(device)
            out = outs['logits'].argmax(dim=2)[:, 4]
            correct = (label == out).sum().item()
            accuracy = correct / 8
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(f'{i}  准确率为:{accuracy}  损失为:{loss}  学习率为:{lr}')

    torch.save(model, './预测中间词.model')


def test():
    test_loader = DataLoader(dataset=dataset['test'],
                             batch_size=8,
                             collate_fn=default_data_collator,
                             shuffle=True,
                             drop_last=True)

    model.eval()
    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        input_ids, attention_mask, labels = data['input_ids'], data['attention_mask'], data['labels']
        label = labels[:, 4]
        labels = None
        with torch.no_grad():
            outs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out = outs['logits'].argmax(dim=2)[:, 4]
        correct += (out == label).sum().item()
        total += 8
        if i % 10 == 0:
            print(i)
            print("预测为：", out)
            print("真实为：", label)
            print()
        if i == 50:
            break
    print("准确率为：", correct/total)

    for i in range(8):
        print(tokenizer.decode(data['input_ids'][i]))
        print("预测值为：", tokenizer.decode(out[i]))
        print("真实值为：", tokenizer.decode(label[i]))
        print()


if __name__ == '__main__':
    # train()
    model = torch.load("./预测中间词.model", map_location='cpu')
    test()







