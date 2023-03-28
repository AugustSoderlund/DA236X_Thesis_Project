import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import matplotlib.pyplot as plt

if __package__ or "." in __name__:
    from utils.data_reader import LABELS
else:
    from ..utils.data_reader import LABELS


def one_hot_encode(Y: np.ndarray):
    """ One-hot encodes the labels

        Parameters:
        ------------
        Y : np.ndarray
            Ground truth labeling with size (N,)
    """
    _labels, dim = [0]*len(Y), len(LABELS.keys())
    for i,y in enumerate(Y):
        _l = np.zeros(shape=(dim,))
        _l[y] = 1
        _labels[i] = _l
    return np.array(_labels)


class DNN:
    def __init__(self, input_size: np.ndarray, output_size: np.ndarray, nodes: np.ndarray = [200, 100], batch_size: int = 100):
        self._inp_size, self._out_size, self._nodes, self._batch_size = input_size, output_size, nodes, batch_size
        self.__create_model()

    def __create_model(self):
        """ Creates the TensorFlow model """
        # Help understanding LSTM https://www.kaggle.com/code/shivajbd/input-and-output-shape-in-lstm-keras
        # LSTM layer must have 3 dims into it.
        #   * Currently the 3 dims are                  : (100, 1, 180) --> (batch_size, 1, input_length)
        #   * Can also reshape input to (x,y,...,ay)    : (100, 6, 30)  --> (batch_size, #dims in input, length of trajectory)
        self._model = keras.models.Sequential()
        self._model.add(tf.keras.layers.Input(shape=(1,self._inp_size), batch_size=self._batch_size))
        self._model.add(tf.keras.layers.Dense(self._inp_size, activation="relu"))
        _return_seqs = [True]*len(self._nodes)
        _return_seqs[-1] = False
        for i,n in enumerate(self._nodes):
            self._model.add(tf.keras.layers.LSTM(n, activation="relu", return_sequences=_return_seqs[i], dropout=0.1))
        self._model.add(tf.keras.layers.Dense(self._out_size, activation="softmax"))

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 20, val_size: float = 0.2):
        """ Train the classifier on data and labels

            Parameters:
            -----------
            x : np.ndarray
                Chunks of trajectories
            y : np.ndarray
                Labels of chunks
            epochs : int (default = 20)
                The number of epochs that the model
                should be trained for
            val_size : float (default = 0.2)
                Size of the validation dataset
                (will round to nearest batchsize)
        """
        def unison_shuffled_copies(x: np.ndarray, y: np.ndarray):
            assert x.shape[0] == y.shape[0]
            p = np.random.permutation(x.shape[0])
            return x[p,:,:], y[p,:]
        self.__compile()
        if len(x.shape) <= 2: x = x.reshape(x.shape[0], 1, x.shape[1])
        if len(y.shape) == 1: y = one_hot_encode(y)
        _ids = np.random.randint(0, x.shape[0], size=(self._batch_size * math.floor(x.shape[0]/self._batch_size),))
        x, y = x[_ids,:,:], y[_ids,:]
        x, y = unison_shuffled_copies(x, y)
        x = tf.keras.utils.normalize(x, axis=2)
        _v = math.floor(x.shape[0]*val_size/self._batch_size) * self._batch_size
        _x, _y, x_val, y_val = x[0:-_v,:,:], y[0:-_v,:], x[_v:,:,:], y[_v:,:]
        self.__history = self._model.fit(_x, _y, self._batch_size, epochs, validation_data=(x_val, y_val))

    def predict(self, x: np.ndarray):
        """ Predict the mode of trajectory x """
        if len(x.shape) == 1: x = x.reshape(1, 1, *x.shape)
        else: x = x.reshape(x.shape[0], 1, x.shape[1])
        x = tf.keras.utils.normalize(x)
        return self._model(x)

    def __compile(self):
        """ Function for compiling the model """
        self._model.compile(optimizer="adam", 
                            loss=tf.keras.losses.CategoricalCrossentropy(), 
                            metrics=[keras.metrics.TopKCategoricalAccuracy(k=1, name="acc")])

    def plot_training(self):
        """ Plot the training losses and accuracies """
        acc, val_acc = 100*np.array(self.__history.history["acc"]), 100*np.array(self.__history.history["val_acc"])
        loss, val_loss = self.__history.history["loss"], self.__history.history["val_loss"]
        epochs = list(range(1,len(loss)+1))
        fig, ax = plt.subplots(1,2)
        ax[0].plot(epochs, acc, c="g")
        ax[0].plot(epochs, val_acc, c="orange")
        ax[0].set_ylim(0,100), ax[0].set_ylabel("Accuracy [%]")
        ax[0].legend(["Accuracy", "Validation accuracy"])
        ax[1].plot(epochs, loss, c="g")
        ax[1].plot(epochs, val_loss, c="orange")
        ax[1].legend(["Loss", "Validation loss"])
        ax[1].set_ylabel("Categorical cross-entropy loss")
        ax[0].grid(), ax[1].grid()
        fig.suptitle("Loss/accuracy during training")
        fig.supxlabel("Epochs")
        plt.show()
    

if __name__ == "__main__":
    dnn = DNN(180, 7, [150, 100,50])
    x, y = [], []
    for i in range(0,100):
        x.append(tf.random.normal(shape=(180,)))
        y.append(tf.random.normal(shape=(7,)))
    x = np.array(x)
    y = np.array(y)
    print(x[0:3,:].shape, len(x[0:3,:].shape))
    dnn.train(x, y)
    x_pred = dnn.predict(x[0:3,:])
    print(x_pred)
    dnn._model.summary()