from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import datasets
import torch
import warnings
from transformers import AdamW
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

tokenizer = AutoTokenizer.from_pretrained('hfl/rbt6')
# print(tokenizer)
#
#
encode = tokenizer.batch_encode_plus([[
    '海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间','的', '海', '域', '。'],
    ['这', '座', '依', '山', '傍', '水', '的', '博', '物', '馆', '由', '国', '内', '一','流', '的', '设', '计', '师', '主', '持', '设', '计', '，', '整', '个', '建', '筑','群', '精', '美', '而', '恢', '宏', '。']],
    truncation=True,
    return_tensors='pt',
    padding=True,
    max_length=25,
    is_split_into_words=True)

# encode = tokenizer.batch_encode_plus([[
#     '海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间','的', '海', '域', '。'],
#     ['这座山傍水的博物馆由国内流的', '设', '计', '师', '主', '持', '设', '计', '，', '整', '个', '建', '筑','群', '精', '美', '而', '恢', '宏', '。']],
#     truncation=True,
#     return_tensors='pt',
#     padding=True,
#     max_length=25,
#     is_split_into_words=True)

print(encode)
#
# dataset = datasets.load_from_disk('../../data/peoples_daily_ner')
# print(dataset['train'][0:2])
#
# """
# O：表示不属于一个命名实体。   0
# B-PER：表示人名的开始。1
# I-PER：表示人名的中间和结尾部分。2
# B-ORG：表示组织机构名的开始。3
# I-ORG：表示组织机构名的中间和结尾部分。4
# B-LOC：表示地名的开始。5
# I-LOC：表示地名的中间和结尾部分。6
# """
#
#
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, split):
#         dataset = datasets.load_from_disk('../../data/peoples_daily_ner')[split]
#
#         def f(data):
#             return len(data['tokens']) <= 512 - 2
#
#         dataset = dataset.filter(f)
#         self.dataset = dataset
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, i):
#         tokens = self.dataset[i]['tokens']
#         labels = self.dataset[i]['ner_tags']
#         return tokens, labels
#
#
# def collate(data):
#     tokens = [i[0] for i in data]
#     labels = [i[1] for i in data]
#     inputs = tokenizer.batch_encode_plus(
#         tokens,
#         truncation=True,
#         padding=True,
#         return_tensors='pt',
#         is_split_into_words=True
#     )
#     lens = inputs['input_ids'].shape[1]
#     for i in range(len(labels)):
#         labels[i] = [7] + labels[i]
#         labels[i] += [7] * lens
#         labels[i] = labels[i][:lens]
#
#     return inputs, torch.LongTensor(labels)
#
#
# loader = torch.utils.data.DataLoader(
#     dataset=Dataset('train'),
#     batch_size=16,
#     collate_fn=collate,
#     shuffle=True,
#     drop_last=True
# )
#
# for (inputs, labels) in loader:
#     # print(inputs)
#     # print(inputs['input_ids'].shape)
#     # print(labels.shape)
#
#     break
#
# # encode = tokenizer.decode(inputs['input_ids'][0])
# # print(encode)
# # print(labels[0])
# # print(len(encode))
# # print(len(labels[0]))
#
# # for (k, v) in inputs.items():
# #     print(k, v.shape)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# pretrained = AutoModel.from_pretrained('hfl/rbt6').to(device)
#
#
# # 打印有多少个参数
# # print(sum(i.numel() for i in pretrained.parameters()))
# # print(pretrained(**inputs).last_hidden_state.shape)
#
#
# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.tuneing = False
#         self.pretrained = None
#         self.gru = nn.GRU(768, 768, batch_first=True)
#         self.fc = nn.Linear(768, 8)
#
#     def forward(self, input_ids, token_type_ids, attention_mask):
#         if self.tuneing:
#             out = self.pretrained(input_ids, token_type_ids, attention_mask).last_hidden_state
#         else:
#             out = pretrained(input_ids, token_type_ids, attention_mask).last_hidden_state
#
#         out, _ = self.gru(out)
#         out = self.fc(out).softmax(dim=2)
#         return out
#
#     def fine_tuneing(self, tuneing):
#         self.tuneing = tuneing
#         if tuneing:
#             for i in pretrained.parameters():
#                 i.requires_grad = True
#             pretrained.train()
#             self.pretrained = pretrained
#         else:
#             for i in pretrained.parameters():
#                 i.requires_grad = False
#             pretrained.eval()
#             self.pretrained = None
#
#
# model = Model().to(device)
#
#
# # print(model(**inputs).shape)
# # print(labels.shape)
#
#
# def reshape_and_remove_pad(outs, labels, attention_mask):
#     # print(outs)
#     # print("-----------------------------")
#     outs = outs.reshape(-1, 8)
#     # print(outs)
#     # print("-----------------------------")
#     labels = labels.reshape(-1)
#     condition = attention_mask.reshape(-1) == 1
#     # print("-----------------------------")
#     outs = outs[condition]
#     # print(outs)
#     labels = labels[condition]
#
#     return outs, labels
#
#
# # attention_mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
# # outs, labels = reshape_and_remove_pad(torch.randn(2, 3, 8), torch.ones(2, 3), attention_mask)
# # print(outs)
# # print(labels)
#
#
# # 获取正确数量的总数
# def get_correct_and_total_count(labels, outs):
#     outs = outs.argmax(dim=1)
#     correct = (labels == outs).sum().item()
#     total = len(labels)
#
#     select = labels != 0
#     labels = labels[select]
#     outs = outs[select]
#     correct_content = (outs == labels).sum().item()
#     total_content = len(labels)
#
#     return correct, total, correct_content, total_content
#
#
# # correct, total, correct_content, total_content = get_correct_and_total_count(torch.ones(16), torch.randn(16, 8))
# # print(correct)
# # print(total)
#
#
# # 定义训练函数
# def train(epochs):
#     lr = 2e-5 if model.tuneing else 5e-4
#     optimizer = AdamW(model.parameters(), lr=lr)
#     criterion = torch.nn.CrossEntropyLoss()
#
#     for epoch in range(epochs):
#         for step, (inputs, labels) in enumerate(loader):
#             input_ids, input_type_ids, attention_mask = inputs['input_ids'], inputs['token_type_ids'], inputs[
#                 'attention_mask']
#             input_ids, input_type_ids, attention_mask = input_ids.to(device), input_type_ids.to(
#                 device), attention_mask.to(device)
#             labels = labels.to(device)
#             outs = model(input_ids, input_type_ids, attention_mask)
#
#             outs, labels = reshape_and_remove_pad(outs, labels, attention_mask)
#
#             loss = criterion(outs, labels)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             if step % 50 == 0:
#                 count = get_correct_and_total_count(labels, outs)
#                 accurate = count[0] / count[1]
#                 accurate_content = count[2] / count[3]
#                 print(f'epoch={epoch}  train_step={step}  {loss.item()}  {accurate}  {accurate_content}')
#
#         torch.save(model, './中文命名体识别.model')
#
#
# # model.fine_tuneing(False)
# # train(10)
#
#
#
# # def test():
# #     val_loader = DataLoader(dataset=Dataset('validation'),
# #                             batch_size=128,
# #                             collate_fn=collate,
# #                             shuffle=True,
# #                             drop_last=True)
# #     model = torch.load('./中文命名体识别.model')
# #     model.to(device)
# #     model.eval()
# #     correct = 0
# #     total = 0
# #     correct_content = 0
# #     total_content = 0
# #     for step, (inputs, labels) in enumerate(val_loader):
# #         if step == 10:
# #             break
# #         input_ids, token_type_ids, attention_mask = inputs['input_ids'].to(device), inputs['token_type_ids'].to(device), \
# #             inputs['attention_mask'].to(device)
# #         labels = labels.to(device)
# #         with torch.no_grad():
# #             outs = model(input_ids, token_type_ids, attention_mask)
# #         outs, labels = reshape_and_remove_pad(outs, labels, attention_mask)
# #         count = get_correct_and_total_count(labels, outs)
# #         correct += count[0]
# #         total += count[1]
# #         correct_content += count[2]
# #         total_content += count[3]
# #     print(total)
# #     print(total_content)
# #     print(f'correct={correct / total}  correct_content={correct_content / total_content}')
#
#
# # test()
#
# def predict():
#     test_loader = DataLoader(dataset=Dataset('test'),
#                              batch_size=32,
#                              collate_fn=collate,
#                              shuffle=True,
#                              drop_last=True)
#
#     model = torch.load('./中文命名体识别.model')
#     model.to(device)
#     model.eval()
#     for inputs, labels in test_loader:
#         break
#     input_ids, token_type_ids, attention_mask = inputs['input_ids'].to(device), inputs['token_type_ids'].to(device), \
#         inputs['attention_mask'].to(device)
#     labels = labels.to(device)
#     # print(labels.shape)
#     # print(labels)
#     with torch.no_grad():
#         outs = model(input_ids, token_type_ids, attention_mask)
#     outs = outs.argmax(dim=2)
#     # print(outs)
#     # print(outs.shape)
#
#     for i in range(32):
#         select = attention_mask[i] == 1
#         # print(input_ids[i])
#         input_id = input_ids[i, select]
#         out = outs[i, select]
#         label = labels[i, select]
#         print('原句子为：', tokenizer.decode(input_id).replace(' ', ''))
#         for sen in [label, out]:
#             s = ''
#             for j in range(len(label)):
#                 if sen[j] == 0:
#                     s += '*'
#                     continue
#                 s += str(sen[j].item())
#                 s += tokenizer.decode(input_id[j])
#             if torch.equal(sen,label):
#                 print('正确值为：', s)
#             elif torch.equal(sen,out):
#                 print('预测值为：', s)
#         print('------------------------------------------------------------------')
#
#
# # predict()
