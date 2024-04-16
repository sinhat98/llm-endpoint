import unittest
from unittest.mock import MagicMock
from pathlib import Path
import numpy as np

# 上記で定義したクラスと関数をインポート
from algo.bert import CT2BertSentimentClassifier, SentimentOutput

class TestCT2BertSentimentClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = CT2BertSentimentClassifier()
        # モックを利用して、外部依存関係を模擬化
        self.classifier.model = MagicMock()
        self.classifier.tokenizer = MagicMock()
        self.classifier.classifier = MagicMock()

    def test_get_embedding(self):
        # ダミーテキスト
        dummy_text = "これはテストです。"
        # モックの設定
        self.classifier.model.forward_batch.return_value.pooler_output = [[0.5, -0.5]]
        # 実行
        embedding = self.classifier.get_embedding(dummy_text)
        # 検証
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (1, 2))

    def test_call(self):
        # ダミーテキスト
        dummy_text = "これはテストです。"
        # モックの設定
        self.classifier.get_embedding = MagicMock(return_value=np.array([[0.5, -0.5]]))
        self.classifier.get_logits = MagicMock(return_value=np.array([0.7, 0.3]))
        # 実行
        result = self.classifier(dummy_text)
        # 検証
        self.assertIsInstance(result, SentimentOutput)
        self.assertIn(result.label, [0, 1])
        self.assertTrue(0 <= result.score <= 1)

if __name__ == '__main__':
    unittest.main()
