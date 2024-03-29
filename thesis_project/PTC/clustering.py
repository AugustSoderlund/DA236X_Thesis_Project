from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.data_processor import process_data
import numpy as np


class Cluster:
    def __init__(self, n: int = 8, eps: float = 0.3, preprocess_data: bool = True):
        self.n_c, self.eps, self.preprocess_data = n, eps, preprocess_data
        self.classifier = KMeans(n_clusters=n, n_init=10)
        self._scaler = StandardScaler()

    def train(self, data: np.ndarray, labels: np.ndarray, input_len: int = 30):
        _data = process_data(data, input_len) if self.preprocess_data else data
        _d = self._scaler.fit_transform(X=_data, y=labels)
        self.classifier.fit(X=_d, y=labels)

    def predict(self, data: np.ndarray):
        _data = process_data(data, int(data.shape[1]/6)) if self.preprocess_data else data
        _d = self._scaler.fit_transform(_data)
        return self.classifier.predict(_d)
    
class AutoCluster(Cluster):
    def __init__(self, n: int = 8, eps: float = 0.3, preprocess_data: bool = True):
        super().__init__(n, eps, preprocess_data)

    def train(self, data: np.ndarray, input_len: int = 30):
        _data = process_data(data, input_len) if self.preprocess_data else data
        _d = self._scaler.fit_transform(_data)
        self.classifier.fit(_d)