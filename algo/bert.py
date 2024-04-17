from dataclasses import dataclass
from pathlib import Path

import ctranslate2
import numpy as np
from transformers import AutoTokenizer


@dataclass
class SentimentOutput:
    label: int
    score: float


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class CT2BertSentimentClassifier:
    def __init__(self, model_dir):
        self.model = ctranslate2.Encoder(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.classifier = np.load(f'{model_dir}/classifier.npy')

    def get_embedding(self, text):
        inputs = self.tokenizer([text])
        embedding = self.model.forward_batch(inputs.input_ids).pooler_output
        embedding = np.array(embedding)
        return embedding

    def get_logits(self, embedding):
        logits = softmax(embedding @ self.classifier.T)
        return logits

    def __call__(self, x):
        embedding = self.get_embedding(x)
        logits = self.get_logits(embedding)
        label = logits.argmax(axis=-1).item()
        score = logits.max()
        return SentimentOutput(label=label, score=score)


def get_bert_model():
    model_dir = Path(__file__).parent / 'ct2-bert-base-japanese-sentiment-cyberbullying'
    return CT2BertSentimentClassifier(str(model_dir))
