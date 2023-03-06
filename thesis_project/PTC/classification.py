from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from utils.data_processor import process_data
import numpy as np

class DecisionTree:
    def __init__(self, max_depth: int = 5, preprocess_data: bool = True):
        self.preprocess_data = preprocess_data
        self._scaler = StandardScaler()
        self.classifier = DecisionTreeClassifier(max_depth=max_depth)

    def train(self, data: np.ndarray, labels: np.ndarray, input_len: int = 30):
        _data = process_data(data, input_len) if self.preprocess_data else data
        _d = self._scaler.fit_transform(X=_data, y=labels)
        self.classifier.fit(X=_d, y=labels)

    def predict(self, data: np.ndarray, input_len: int = 30):
        _data = process_data(data, input_len) if self.preprocess_data else data
        _d = self._scaler.fit_transform(_data)
        return self.classifier.predict(_d)