import numpy as np


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    return A


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y, num_classes):
    one_hot_Y = np.zeros((Y.size, num_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


class Layer:
    def __init__(self, input_size, output_size, activation):
        self.deactivated = None
        self.activated = None
        self.W = np.random.rand(output_size, input_size) - 0.5
        self.b = np.random.rand(output_size, 1) - 0.5
        self.activation_funct = activation

    def activate(self, X):
        self.deactivated = self.W.dot(X) + self.b
        self.activated = self.activation_funct(self.deactivated)

    def update(self, dW, db, learning_rate):
        self.W -= learning_rate * dW
        self.b -= learning_rate * db


class MLP:
    def __init__(self,
                 input_size,
                 output_size,
                 n_hidden_layers,
                 learning_rate=0.1):

        if n_hidden_layers < 1:
            raise ValueError("Must have at least 1 hidden layer !!")

        self.learning_rate = learning_rate

        self.layers = []
        hidden_size = input_size
        for i in range(n_hidden_layers):
            hidden_size = output_size * (n_hidden_layers-i)
            self.layers.append(Layer(input_size, hidden_size, ReLU))
            input_size = hidden_size

        self.layers.append(Layer(hidden_size, output_size, softmax))

    def forward(self, X):
        for layer in self.layers:
            layer.activate(X)
            X = layer.activated

    def backward(self, X, Y):
        dWs = []
        dbs = []
        m = X.shape[1]

        one_hot_Y = one_hot(Y, self.layers[-1].W.shape[0])
        dZ = self.layers[-1].activated - one_hot_Y
        dWs.append(1 / m * dZ.dot(self.layers[-2].activated.T))
        dbs.append(1 / m * np.sum(dZ))

        for i in range(len(self.layers) - 2):
            dZ = self.layers[-1 - i].W.T.dot(dZ) * ReLU_deriv(self.layers[-2 - i].deactivated)
            dWs.append(1 / m * dZ.dot(self.layers[-3 - i].activated.T))
            dbs.append(1 / m * np.sum(dZ))

        dZ = self.layers[1].W.T.dot(dZ) * ReLU_deriv(self.layers[0].deactivated)
        dWs.append(1 / m * dZ.dot(X.T))
        dbs.append(1 / m * np.sum(dZ))

        self.update(reversed(dWs), reversed(dbs))

    def update(self, dWs, dbs):
        for layer, dW, db in zip(self.layers, dWs, dbs):
            layer.update(dW, db, self.learning_rate)

    def train(self, x, target, epochs=500, verbose=False):
        i = 0
        while i <= epochs:
            self.forward(x)
            self.backward(x, target)
            if verbose and i % 100 == 0:
                print(f"Epoch: {i} | Accuracy: {round(get_accuracy(self.predict(x), target), 4)}")
            i += 1

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.layers[-1].activated, axis=0)
