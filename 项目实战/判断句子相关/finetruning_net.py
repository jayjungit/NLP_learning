import torch.nn as nn
import torch
from transformers import AutoModel
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = AutoModel.from_pretrained('bert-base-chinese')
        self.fc1 = nn.Linear(768, 8)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            x = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = self.dropout(x.last_hidden_state[:, 0])
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        return x


