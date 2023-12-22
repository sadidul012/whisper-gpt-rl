from datasets import Audio
from transformers import WhisperTokenizer
from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperProcessor
import os
import evaluate
from peft import prepare_model_for_int8_training
import torch
from transformers import WhisperForConditionalGeneration
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from config import WHISPER_MODEL_OUTPUT_PEFT, WHISPER_MODEL, DATASET_STT, WHISPER_MODEL_LANGUAGE
from transformers import Seq2SeqTrainingArguments
from transformers.trainer_callback import PrinterCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
language_abbr = "bn"
task = "transcribe"

common_voice = DatasetDict()
# amount = "[50%:51%]"
amount = ""
common_voice["train"] = load_dataset(DATASET_STT, language_abbr, split=f"train{amount}+validation{amount}", token=True)
common_voice["test"] = load_dataset(DATASET_STT, language_abbr, split=f"test{amount}", token=True)

print(common_voice)
common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
)
print(common_voice)

feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
tokenizer = WhisperTokenizer.from_pretrained(WHISPER_MODEL, language=WHISPER_MODEL_LANGUAGE, task=task)
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL, language=WHISPER_MODEL_LANGUAGE, task=task)
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


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

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
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
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


model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL, load_in_8bit=True)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

model = prepare_model_for_int8_training(model)

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


training_args = Seq2SeqTrainingArguments(
    output_dir="temp",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    report_to=["tensorboard"],
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=256,
    logging_steps=25,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
    save_total_limit=1,
    save_strategy="no",
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,

    train_dataset=common_voice["train"],
    callbacks=[SavePeftModelCallback]
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.remove_callback(PrinterCallback)

print("training...")
try:
    trainer.train()
except KeyboardInterrupt:
    print("KeyboardInterrupt")
    pass


processor.save_pretrained(WHISPER_MODEL_OUTPUT_PEFT)
trainer.save_model(WHISPER_MODEL_OUTPUT_PEFT)
# model.save_pretrained(WHISPER_MODEL_OUTPUT_PEFT)

