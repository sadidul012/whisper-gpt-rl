import numpy as np
import speech_recognition as sr
from config import USE_MICROPHONE, INPUT_AUDIO, TARGET_SAMPLING_RATE
from transformers import WhisperForConditionalGeneration
import torch
from datasets import Audio
from transformers import WhisperProcessor
from datasets import load_dataset, DatasetDict
from config import WHISPER_MODEL, WHISPER_MODEL_LANGUAGE, DATASET_STT, WHISPER_MODEL_OUTPUT
import sys


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


r = sr.Recognizer()
task = "transcribe"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# load model and processor
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, language=WHISPER_MODEL_LANGUAGE)
model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(device)
model.config.forced_decoder_ids = None

dataset = DatasetDict()
dataset["train"] = load_dataset(DATASET_STT, "bn", split="train", token=True)
dataset["validation"] = load_dataset(DATASET_STT, "bn", split="validation", token=True)
dataset["test"] = load_dataset(DATASET_STT, "bn", split="test", token=True)
# dataset["train"] = load_dataset(DATASET_STT, "bn", split="train[50%:51%]", token=True)
# dataset["validation"] = load_dataset(DATASET_STT, "bn", split="validation[50%:51%]", token=True)
# dataset["test"] = load_dataset(DATASET_STT, "bn", split="test[50%:51%]", token=True)
print(dataset)

dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLING_RATE))


def recognize(batch):
    array = [processor(b["array"], sampling_rate=TARGET_SAMPLING_RATE, return_tensors="pt").input_features for b in batch["audio"]]
    array = np.array(array).reshape((-1, array[0].shape[1], array[0].shape[2]))
    predicted_ids = model.generate(torch.from_numpy(array).to(device))
    batch["output"] = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return batch


columns_to_remove = ['path', 'audio', ]
dataset = dataset.map(recognize, remove_columns=columns_to_remove, batched=True, batch_size=32)
dataset.save_to_disk("./dataset/")
