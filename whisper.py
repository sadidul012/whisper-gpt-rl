from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from config import WHISPER_MODEL_LANGUAGE, WHISPER_MODEL_OUTPUT, WHISPER_MODEL
import speech_recognition as sr
from config import USE_MICROPHONE, INPUT_AUDIO, TARGET_SAMPLING_RATE
import librosa
import numpy as np


r = sr.Recognizer()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# load model and processor
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, language=WHISPER_MODEL_LANGUAGE)
model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(device)
# processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_OUTPUT, language=WHISPER_MODEL_LANGUAGE)
# model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_OUTPUT).to(device)
model.config.forced_decoder_ids = None


def recognize(path=None):
    if USE_MICROPHONE or path is None:
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)
    else:
        with sr.WavFile(path) as source:
            audio = r.record(source)

    sampling_rate = audio.sample_rate
    # print(sampling_rate)
    audio_data = audio.get_wav_data()
    data_s16 = np.frombuffer(audio_data, dtype=np.int16, count=len(audio_data) // 2, offset=0)
    float_data = data_s16.astype(np.float32, order='C') / 32768.0
    array = librosa.resample(float_data, orig_sr=sampling_rate, target_sr=TARGET_SAMPLING_RATE)
    # print(array.shape)
    input_features = processor(array, sampling_rate=TARGET_SAMPLING_RATE, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features.to(device))
    # print(processor.batch_decode(predicted_ids, skip_special_tokens=True))
    # ds = Dataset.from_dict({"audio": [path]}).cast_column("audio", Audio(sampling_rate=TARGET_SAMPLING_RATE))
    # sample = ds["audio"][0]
    # input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
    # predicted_ids = model.generate(input_features.to(device))
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)


if __name__ == '__main__':
    import glob

    for i in glob.glob(INPUT_AUDIO + "*.wav"):
        print(i)
        output = recognize(i)
        print(output)
