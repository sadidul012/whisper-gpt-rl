import torch
from peft import PeftModel
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor
import speech_recognition as sr
import librosa
import numpy as np
from config import USE_MICROPHONE, INPUT_AUDIO, TARGET_SAMPLING_RATE

model_name = "openai/whisper-small"
adapters_name = "./whisper-small-bengali"
language = "Bengali"
language_abbr = "bn"
task = "transcribe"
path = "audio/input/ban_00737_00012222450.wav"

print(f"Starting to load the model {model_name} into memory")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
model.config.forced_decoder_ids = None

model = PeftModel.from_pretrained(model, adapters_name)

print(f"Successfully loaded the model {model_name} into memory")
r = sr.Recognizer()
with sr.WavFile(path) as source:
    audio = r.record(source)
    sampling_rate = audio.sample_rate

audio_data = audio.get_wav_data()
data_s16 = np.frombuffer(audio_data, dtype=np.int16, count=len(audio_data) // 2, offset=0)
float_data = data_s16.astype(np.float32, order='C') / 32768.0
array = librosa.resample(float_data, orig_sr=sampling_rate, target_sr=TARGET_SAMPLING_RATE)
# print(array.shape)
input_features = processor(array, sampling_rate=TARGET_SAMPLING_RATE, return_tensors="pt").input_features
predicted_ids = model.generate(input_features.to(device))
text = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(text)
