import json

import torch
import time
from tqdm import tqdm
import numpy as np
from random import choices
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead, create_reference_model, PreTrainedModelWrapper
from datasets import load_dataset, DatasetDict
from datasets import Audio
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration

tqdm.pandas()
DATASET_STT = "mozilla-foundation/common_voice_13_0"
WHISPER_MODEL = "sadidul012/whisper-small-bengali"
WHISPER_MODEL_LANGUAGE = "Bengali"
sentiment_pipe_kwargs = {"top_k": None, "function_to_apply": "none"}

config = PPOConfig(
    model_name="lvwerra/gpt2-imdb", steps=51200, learning_rate=1.41e-5, remove_unused_columns=False,
    log_with="tensorboard", batch_size=10, ratio_threshold=100, project_kwargs={"logging_dir": "logs"}
)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
common_voice = DatasetDict()
common_voice["train"] = load_dataset(DATASET_STT, "bn", split="train[50%:51%]+validation[50%:51%]", token=True)
common_voice["test"] = load_dataset(DATASET_STT, "bn", split="test[50%:51%]", token=True)

feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
tokenizer = WhisperTokenizer.from_pretrained(WHISPER_MODEL, language=WHISPER_MODEL_LANGUAGE, task="transcribe")
input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)


print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, language=WHISPER_MODEL_LANGUAGE, task="transcribe")
print(common_voice["train"][0])

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
print(common_voice["train"][0])


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)


class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __init__(self, proc):
        self.processor = proc

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        _labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (_labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            _labels = _labels[:, 1:]

        batch["labels"] = _labels

        return batch


print("running collator")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(proc=processor)

model = PreTrainedModelWrapper.from_pretrained(WHISPER_MODEL)
ref_model = create_reference_model(model)
# model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
# model.config.forced_decoder_ids = None
# model.config.suppress_tokens = []

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=common_voice, data_collator=data_collator)


print("Done")


