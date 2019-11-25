import pickle
#import cupy as np
import numpy as np


class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot"
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            #Used to calculate the Boundaries of the uniform distribution
            uni_boundary = 1 / np.sqrt(all_dims[layer_n - 1])

            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))
            self.weights[f"W{layer_n}"] = np.random.uniform(-uni_boundary, uni_boundary, (1, all_dims[layer_n]))

    #Calculates the RELU function
    def relu(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            pass
        # WRITE CODE HERE
        #Checks if x is an array and applies the appropriate function
        if not isinstance(x, list):
            return x if x > 0 else 0
        else:
            x_a = np.array(x)
            return np.where(x_a > 0, x_a, 0)

    # Calculates the Sigmoid function
    def sigmoid(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            pass
        # WRITE CODE HERE
        # Checks if x is an array and applies the appropriate function
        if not isinstance(x, list):
            return 1/(1 + np.exp(-x))
        else:
            x_a = np.array(x)
            return 1/(1 + np.exp(-x_a))

    # Calculates the Tanh function
    def tanh(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            pass
        # WRITE CODE HERE
        # Checks if x is an array and applies the appropriate function
        if not isinstance(x, list):
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        else:
            x_a = np.array(x)
            return (np.exp(x_a) - np.exp(-x_a)) / (np.exp(x_a) + np.exp(-x_a))

    # Returns the appropriate activation (self.activation_str) function on input x
    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            # WRITE CODE HERE
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            # WRITE CODE HERE
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            # WRITE CODE HERE
            return self.tanh(x, grad)
        else:
            raise Exception("invalid")
        return 0

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        # WRITE CODE HERE
        x_a = x - max(x)
        exp_sum = np.sum(np.exp(x_a))
        return np.exp(x_a)/exp_sum

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        # WRITE CODE HERE
        pass
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        # WRITE CODE HERE
        pass
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            pass

    def one_hot(self, y):
        # WRITE CODE HERE
        pass
        return 0

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        # WRITE CODE HERE
        pass
        return 0

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                # WRITE CODE HERE
                pass

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        # WRITE CODE HERE
        pass
        return 0

if __name__ == "__main__":
    # execute only if run as a script
    network = NN()
    network.initialize_weights((500,250))
    nuts = network.relu(-12)
    arr = network.relu([0, -1, 15])

    nuts_sig = network.sigmoid(12)
    arr_sig = network.sigmoid([0, 2, 15])

    exit(0)