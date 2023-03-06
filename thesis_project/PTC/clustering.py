from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


class AutoCluster:
    def __init__(self, n: int = 8, classifier: str = "kmeans", eps: float = 0.3):
        self.classifier = KMeans(n_clusters=n, n_init=10) if classifier == "kmeans" else DBSCAN(eps=eps)
        self._scaler = StandardScaler()

    def train(self, data):
        _d = self._scaler.fit_transform(data)
        self.classifier.fit(_d)

    def predict(self, data):
        _d = self._scaler.fit_transform(data)
        return self.classifier.predict(_d)
    

if __name__ == "__main__":
    classifier = AutoCluster()