from transformers import AutoTokenizer, AutoModel
import warnings
import datasets


warnings.filterwarnings('ignore')


tokenizer = AutoTokenizer.from_pretrained('distilgpt2', use_fast=True)
# tokens = tokenizer.batch_encode_plus(['The movie is great!', 'This warning can be disabled by setting the.'])
# print(tokens)

# 下载数据
# dataset = datasets.load_dataset(path='glue', name='sst2')
# dataset.save_to_disk(dataset_dict_path='../../data/sst2')

# 加载数据
dataset = datasets.load_from_disk('../../data/sst2')
train_data = dataset['train']
print(train_data['sentence'][:10])


def f1(data, tokenizer):
    return tokenizer.batch_encode_plus(data['sentence'])


def f2(data):
    return [len(i) >= 8 for i in data['input_ids']]


def f3(data):
    data['input_ids'] = [i[:8] for i in data['input_ids']]
    data['attention_mask'] = [[1] * 8] * len(data['attention_mask'])
    data['label'] = data['input_ids']
    return data


if __name__ == '__main__':

    dataset = dataset.map(f1, batched=True, batch_size=1000, num_proc=8,
                          remove_columns=['sentence', 'idx', 'label'],
                          fn_kwargs={'tokenizer': tokenizer})
    print(dataset)
    print("=+++++++++++++++++++++++++++++++++++")

    dataset = dataset.filter(f2, batched=True, batch_size=1000, num_proc=8)
    print(dataset)

    dataset = dataset.map(f3, batch_size=True, batched=1000, num_proc=8)
    print(dataset)
    dataset.save_to_disk(dataset_dict_path='../../data/sst')
    print(dataset['train'][0])
