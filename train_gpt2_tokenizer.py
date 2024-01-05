from datasets import load_from_disk
from transformers import AutoTokenizer
import torch

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = load_from_disk("dataset")
batch_size = 1000

print(dataset)


def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["target"]


print("training tokenizer")
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=30522)
new_tokenizer.save_pretrained("bert-tokenizer-bangla")
print("saved tokenizer")
