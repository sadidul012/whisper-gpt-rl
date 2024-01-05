import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained("./gpt2-imdb-ctrl")
gpt2_tokenizer = AutoTokenizer.from_pretrained("./gpt2-imdb-ctrl")
txt_out_len = 256
gpt2_model.to(device)


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
    input_str = """
    Correct sentence: What is the
    """
    input_ids = gpt2_tokenizer.encode(" " + input_str, return_tensors="pt")
    o = gpt2_model.generate(input_ids.to(device), **generation_kwargs)
    o = gpt2_tokenizer.decode(o.squeeze())
    print("input_str", input_str)
    print("output", o)


if __name__ == '__main__':
    test()
