import json

import torch
import time
from tqdm import tqdm
import numpy as np
from random import choices
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from datasets import Dataset

tqdm.pandas()
sentiment_pipe_kwargs = {"top_k": None, "function_to_apply": "none"}

config = PPOConfig(
    model_name="gpt2", steps=51200, learning_rate=1.41e-5, remove_unused_columns=False,
    log_with="tensorboard", batch_size=8, ratio_threshold=100, project_kwargs={"logging_dir": "logs"}
)

txt_in_len = 5
txt_out_len = 256
seed = 42

np.random.seed(seed)

gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
gpt2_model_ref = create_reference_model(gpt2_model)
gpt2_tokenizer = AutoTokenizer.from_pretrained(config.model_name)

gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
dataset = load_dataset("imdb", split="train")
dataset = dataset.rename_columns({"text": "review", "label": "sentiment"})
# make sure the comments are at least 500 and trim to 1000
dataset = dataset.filter(lambda x: len(x["review"]) > 500, batched=False)
dataset = dataset.map(lambda x: {"review": x["review"][:100]}, batched=False)
dataset = dataset.map(
    lambda x: {"input_ids": gpt2_tokenizer.encode(" " + x["review"], return_tensors="pt")[0, :txt_in_len]},
    batched=False,
)
dataset = dataset.map(lambda x: {"query": gpt2_tokenizer.decode(x["input_ids"])}, batched=False)
dataset = dataset[:100]

dataset = Dataset.from_dict(dataset)
dataset.set_format("pytorch")
print("dataset processing done")


def collator(data):
    data = dict((key, [d[key] for d in data]) for key in data[0])
    # print("Collector:", data)
    # for key, value in data.items():
    #     print(key, len(value), value)
    return data


ppo_trainer = PPOTrainer(config, gpt2_model, gpt2_model_ref, gpt2_tokenizer, dataset, data_collator=collator)

if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
else:
    device = ppo_trainer.accelerator.device
sentiment_pipe = pipeline("sentiment-analysis", "lvwerra/distilbert-imdb", device=0)

text = "this movie was really bad!!"
output = sentiment_pipe(text, **sentiment_pipe_kwargs)
print(output)

text = "this movie was really good!!"
output = sentiment_pipe(text, **sentiment_pipe_kwargs)
print(output)

text = "this movie was a documentary"
output = sentiment_pipe(text, **sentiment_pipe_kwargs)
print(output)


def extract_pipe_output(outputs):
    positive_logits = []
    for out in outputs:
        for element in out:
            if element["label"] == "POSITIVE":
                positive_logits.append(torch.tensor(element["score"]))
    return positive_logits


print("output score", output[1]["score"])

ctrl_str = ["[negative]", "[neutral]", "[positive]"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # this should be handled by accelerate
ctrl_tokens = dict((s, gpt2_tokenizer.encode(s, return_tensors="pt").squeeze().to(device)) for s in ctrl_str)

print("ctrl_tokens", ctrl_tokens)


def pos_logit_to_reward(logit, task):
    """
    Take the positive sentiment logit and scale it for the task.
        task [negative]: reward = -logit
        task [neutral]: reward = -2*abs(logit)+4
        task [positive]: reward = logit
    """
    for i in range(len(logit)):
        if task[i] == "[negative]":
            logit[i] = -logit[i]
        elif task[i] == "[neutral]":
            logit[i] = -2 * torch.abs(logit[i]) + 4
        elif task[i] == "[positive]":
            pass
        else:
            raise ValueError("task has to be in [0, 1, 2]!")
    return logit


print("ctrl_str", ctrl_str)

print(pos_logit_to_reward(torch.Tensor([4, 4, 4]), ctrl_str))
print(pos_logit_to_reward(torch.Tensor([-4, -4, -4]), ctrl_str))
print(pos_logit_to_reward(torch.Tensor([0, 0, 0]), ctrl_str))

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": False,
    "pad_token_id": gpt2_tokenizer.eos_token_id,
    "max_new_tokens": txt_out_len,
    "eos_token_id": -1,
}


def test():
    input_str = "this movie was really"
    input_ids = gpt2_tokenizer.encode(" " + input_str, return_tensors="pt")
    o = gpt2_model.generate(input_ids.to(device), **generation_kwargs)
    o = gpt2_tokenizer.decode(o.squeeze())
    print("input_str", input_str)
    print("output", o)


for epoch in range(2):
    test()
    for batch in tqdm(ppo_trainer.dataloader):
        # print(batch)
        (logs, game_data,) = (
            dict(),
            dict(),
        )

        # prepend a random control token
        task_list = choices(ctrl_str, k=config.batch_size)
        game_data["query"] = [t + q for t, q in zip(task_list, batch["query"])]
        query_tensors = [torch.cat((ctrl_tokens[t], input_ids)) for t, input_ids in zip(task_list, batch["input_ids"])]

        # get response from gpt2
        response_tensors = []
        for query in query_tensors:
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-txt_out_len:])
        game_data["response"] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]

        # sentiment analysis
        texts = [q + r for q, r in zip(batch["query"], game_data["response"])]
        logits = extract_pipe_output(sentiment_pipe(texts, **sentiment_pipe_kwargs))
        rewards = pos_logit_to_reward(logits, task_list)

        # Run PPO training
        t = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # print(stats)
        for cs in ctrl_str:
            key = "env/reward_" + cs.strip("[]")
            stats[key] = np.mean([r.cpu().numpy() for r, t in zip(rewards, task_list) if t == cs])
        ppo_trainer.log_stats(stats, game_data, rewards)

        # print(logs)

test()

gpt2_model.save_pretrained("gpt2-imdb-ctrl")
gpt2_tokenizer.save_pretrained("gpt2-imdb-ctrl")
