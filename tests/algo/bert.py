import unittest
from unittest.mock import MagicMock
from pathlib import Path
import numpy as np

# 上記で定義したクラスと関数をインポート
from algo.bert import SentimentOutput, softmax


class MockCT2BertSentimentClassifier:
    def __init__(self):
        self.model = MagicMock(return_value=np.random.rand(1, 768))
        self.tokenizer = MagicMock()
        self.classifier = MagicMock(return_value=np.random.rand(2, 768))
    def get_embedding(self, text):
        embedding = self.model(text)
        return embedding
    
    def get_logits(self, embedding):
        classifier_weight = self.classifier()
        logits = softmax(embedding @ classifier_weight.T)
        return logits
    
    def __call__(self, x):
        embedding = self.get_embedding(x)
        logits = self.get_logits(embedding)
        label = np.argmax(logits)
        score = np.max(logits)
        return SentimentOutput(label, score)    

class TestCT2BertSentimentClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = MockCT2BertSentimentClassifier()

    def test_get_embedding(self):
        # ダミーテキスト
        dummy_text = "これはテストです。"
        # 実行
        embedding = self.classifier.get_embedding(dummy_text)
        # 検証
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (1, 768))

    def test_call(self):
        # ダミーテキスト
        dummy_text = "これはテストです。"
        # 実行
        result = self.classifier(dummy_text)
        # 検証
        self.assertIsInstance(result, SentimentOutput)
        self.assertIn(result.label, [0, 1])
        self.assertTrue(0 <= result.score <= 1)

if __name__ == '__main__':
    unittest.main()
