from transformers import WhisperForConditionalGeneration
import evaluate
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Audio
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
from datasets import load_dataset, DatasetDict
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from peft import prepare_model_for_kbit_training

from config import WHISPER_MODEL, WHISPER_MODEL_LANGUAGE, DATASET_STT, WHISPER_MODEL_OUTPUT
import os
import time
from datetime import timedelta
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


start = time.time()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if os.path.exists(WHISPER_MODEL_OUTPUT):
    print("trained model found")
    WHISPER_MODEL = WHISPER_MODEL_OUTPUT
else:
    print("trained model not found")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dataset = DatasetDict()
print(device)
dataset["train"] = load_dataset(DATASET_STT, "bn", split="train[50%:51%]+validation[50%:51%]", token=True)
dataset["test"] = load_dataset(DATASET_STT, "bn", split="test[50%:51%]", token=True)
# # common_voice["train"] = load_dataset(DATASET_STT, "bn", split="train+validation", use_auth_token=True)
# # common_voice["test"] = load_dataset(DATASET_STT, "bn", split="test", use_auth_token=True)

print(dataset)

feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model_ref = create_reference_model(model)
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, language=WHISPER_MODEL_LANGUAGE, task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(WHISPER_MODEL, language=WHISPER_MODEL_LANGUAGE, task="transcribe")

input_str = dataset["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

print(dataset["train"][0])

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
print(dataset["train"][0])


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


common_voice = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)


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


data_collator = DataCollatorSpeechSeq2SeqWithPadding(proc=processor)
metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


print("time taken: ", timedelta(seconds=time.time()-start))
