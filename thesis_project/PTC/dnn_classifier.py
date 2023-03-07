import numpy as np

class DNN:
    def __init__(self, input_size: np.ndarray, output_size: np.ndarray, nodes: np.ndarray = [200, 100]):
        self.__inp_size, self.__out_size, self.__nodes = input_size, output_size, nodes

    def __create_model(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass