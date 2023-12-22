from dotenv import load_dotenv
import os

load_dotenv()
PORT = os.environ.get('PORT')
ENV_NAME = os.environ.get('ENV_NAME')

USE_MICROPHONE = os.environ.get("USE_MICROPHONE") in ["True", "true"]

SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE"))
VOICE = os.environ.get("VOICE")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL")
# WHISPER_MODEL = environ.get("WHISPER_MODEL")
WHISPER_MODEL_OUTPUT = os.environ.get("WHISPER_MODEL_OUTPUT")
WHISPER_MODEL_OUTPUT_PEFT = os.environ.get("WHISPER_MODEL_OUTPUT_PEFT")
WHISPER_MODEL_CHECKPOINT = os.environ.get("WHISPER_MODEL_CHECKPOINT")
WHISPER_MODEL_LANGUAGE = os.environ.get("WHISPER_MODEL_LANGUAGE")

INPUT_AUDIO = os.environ.get("INPUT_AUDIO")
TARGET_SAMPLING_RATE = int(os.environ.get("TARGET_SAMPLING_RATE"))
DATASET_STT = os.environ.get("DATASET_STT")
