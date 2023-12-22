import numpy as np
import evaluate
from transformers import WhisperFeatureExtractor
import os

import torch
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig
import speech_recognition as sr
from config import USE_MICROPHONE, INPUT_AUDIO, TARGET_SAMPLING_RATE, WHISPER_MODEL_OUTPUT_PEFT, WHISPER_MODEL, WHISPER_MODEL_LANGUAGE
import librosa
metric = evaluate.load("wer")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
task = "transcribe"
feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, language=WHISPER_MODEL_LANGUAGE, task=task)
tokenizer = WhisperTokenizer.from_pretrained(WHISPER_MODEL, language=WHISPER_MODEL_LANGUAGE, task=task)

# model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")
model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
peft_config = PeftConfig.from_pretrained(WHISPER_MODEL_OUTPUT_PEFT)
model = PeftModel.from_pretrained(model, WHISPER_MODEL_OUTPUT_PEFT)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
forced_decoder_ids = processor.get_decoder_prompt_ids(language=WHISPER_MODEL_LANGUAGE, task=task)
pipe = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)
r = sr.Recognizer()


def transcribe(audio):
    with torch.cuda.amp.autocast():
        text = pipe(audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]
    return text


def recognize(path=None):
    if USE_MICROPHONE or path is None:
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)
    else:
        with sr.WavFile(path) as source:
            audio = r.record(source)

    sampling_rate = audio.sample_rate
    audio_data = audio.get_wav_data()
    data_s16 = np.frombuffer(audio_data, dtype=np.int16, count=len(audio_data) // 2, offset=0)
    float_data = data_s16.astype(np.float32, order='C') / 32768.0
    array = librosa.resample(float_data, orig_sr=sampling_rate, target_sr=TARGET_SAMPLING_RATE)
    input_features = processor(array, sampling_rate=TARGET_SAMPLING_RATE, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)


if __name__ == '__main__':
    import glob

    for i in glob.glob(INPUT_AUDIO + "*.wav"):
        print(i)
        output = recognize(i)
        print(output)
