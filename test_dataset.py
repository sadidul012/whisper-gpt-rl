import numpy as np
import tqdm
from datasets import load_from_disk
from transformers import WhisperTokenizer
from sklearn.metrics import accuracy_score, mean_squared_error

from config import WHISPER_MODEL, WHISPER_MODEL_LANGUAGE

dataset = load_from_disk("./dataset")
tokenizer = WhisperTokenizer.from_pretrained(WHISPER_MODEL, language=WHISPER_MODEL_LANGUAGE, task="transcribe")


def test_results(d):
    accurate = 0
    accuracy = []
    mse = []
    progress = tqdm.tqdm(range(len(d)))
    for i in d:
        if i["sentence"] == i["output"]:
            accurate += 1

        sentence = tokenizer(i["sentence"]).input_ids
        output = tokenizer(i["output"].strip()).input_ids
        if len(sentence) > len(output):
            output = output + ([0] * (len(sentence) - len(output)))
        if len(sentence) < len(output):
            sentence = sentence + ([0] * abs(len(sentence) - len(output)))

        accuracy.append(accuracy_score(sentence, output))
        mse.append(mean_squared_error(sentence, output))
        progress.update()

    print("accurate:", accurate, "total:", len(d), "accuracy", accurate / len(d))
    print("average accuracy:", np.mean(accuracy))
    print("average mse:", np.mean(mse))


if __name__ == '__main__':
    dataset = dataset.remove_columns(
        ['client_id', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'variant']
    )
    print(dataset)
    print(dataset["train"][0])
    print(dataset["validation"][0])
    print(dataset["test"][0])

    print("test set")
    test_results(dataset["test"])
    print("validation set")
    test_results(dataset["validation"])
    print("train set")
    test_results(dataset["train"])
