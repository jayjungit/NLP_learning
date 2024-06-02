import torch
import torch.nn as nn
from transformers import AutoTokenizer
from finetruning_net import Net

model = Net()
model.load_state_dict(torch.load('./bert_net.pth'))
# print(model)

text1 = input('请输入第一句话:')
text2 = input('请输入第二句话:')

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
encode = tokenizer.encode_plus(text1, text2,
                               max_length=20,
                               truncation=True,
                               padding='max_length',
                               return_tensors='pt',)
                               # return_attention_mask=True,
                               # return_token_type_ids=True)
# print(encode)
with torch.no_grad():
    out = model(encode['input_ids'], encode['attention_mask'], encode['token_type_ids'])
print('两个句子是否相关', out.argmax(dim=-1).item())
