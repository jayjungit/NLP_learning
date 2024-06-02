from transformers import AutoTokenizer
from transformers.data import default_data_collator
import datasets


tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
dataset = datasets.load_from_disk('../../data/sst2')


def f(data, tokenizer):
    return tokenizer.batch_encode_plus(data['sentence'])


def f2(data):
    return [len(i) >= 9 for i in data['input_ids']]


def f3(data):
    b = len(data['input_ids'])
    data['labels'] = data['attention_mask'].copy()
    for i in range(b):
        data['input_ids'][i] = data['input_ids'][i][:9]
        data['attention_mask'][i] = [1] * 9
        data['labels'][i] = [-100] * 9

        data['input_ids'][i][-1] = 2
        data['labels'][i][4] = data['input_ids'][i][4]
        data['input_ids'][i][4] = 50264
    return data


if __name__ == '__main__':
    dataset = dataset.map(f, batch_size=1000, batched=True, num_proc=4, remove_columns=['sentence', 'idx', 'label'],
                          fn_kwargs={'tokenizer': tokenizer})

    dataset = dataset.filter(f2, batched=True, batch_size=1000, num_proc=4)

    dataset = dataset.map(f3, batched=True, batch_size=1000, num_proc=4)

    dataset.save_to_disk('../../data/sst3')

    print(dataset)