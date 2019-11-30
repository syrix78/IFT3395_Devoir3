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
            uni_boundary = 1.0 / np.sqrt(all_dims[layer_n - 1])

            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))
            self.weights[f"W{layer_n}"] = np.random.uniform(-uni_boundary, uni_boundary, (all_dims[layer_n - 1], all_dims[layer_n]))

    #Calculates the RELU function
    def relu(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            return 1 * (x > 0)

        # WRITE CODE HERE
        return np.maximum(x, 0)

    # Calculates the Sigmoid function
    def sigmoid(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            if not (isinstance(x, list) or isinstance(x, np.ndarray)):
                sigmoid = 1.0 / (1 + np.exp(-x))
                return sigmoid * (1.0 - sigmoid)
            else:
                x_a = np.array(x)
                sigmoid = 1.0 / (1 + np.exp(-x_a))
                return sigmoid * (1.0 - sigmoid)
        # WRITE CODE HERE
        # Checks if x is an array and applies the appropriate function
        if not (isinstance(x, list) or isinstance(x, np.ndarray)):
            return 1.0/(1 + np.exp(-x))
        else:
            x_a = np.array(x)
            return 1.0/(1 + np.exp(-x_a))

    # Calculates the Tanh function
    def tanh(self, x, grad=False):
        if grad:
            if not (isinstance(x, list) or isinstance(x, np.ndarray)):
                tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
                return 1.0 - tanh ** 2
            else:
                x_a = np.array(x)
                tanh = (np.exp(x_a) - np.exp(-x_a)) / (np.exp(x_a) + np.exp(-x_a))
                return 1.0 - tanh ** 2
        # WRITE CODE HERE
        # Checks if x is an array and applies the appropriate function
        if not (isinstance(x, list) or isinstance(x, np.ndarray)):
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

    # Calculates the Softmax function
    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        # WRITE CODE HERE
        if np.array(x).ndim == 1:
            x_a = x - np.max(x)
            exp_sum = np.sum(np.exp(x_a))
            return np.exp(x_a) / exp_sum
        else:
            y = np.array([[np.max(x_i) for x_i in x]])
            x_a = x - y.T
            exp_sum = np.sum(np.exp(x_a), axis=1)

        exp_sum = np.sum(np.exp(x_a), axis=1)
        return np.array([np.exp(x_a[i])/(exp_sum[i]) for i in range(exp_sum.shape[0])])

    def forward(self, x):
        cache = {"Z0": x}

        # cache is a dictionary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        # WRITE CODE HERE
        for layer_n in range(1, self.n_hidden + 2):
            cache[f"A{layer_n}"] = np.dot(np.array(cache[f"Z{layer_n - 1}"]), self.weights[f"W{layer_n}"]) + self.weights[f"b{layer_n}"]

            #If we are on the last layer (output) we use the softmax function
            if layer_n == self.n_hidden + 1:
                cache[f"Z{layer_n}"] = self.softmax(cache[f"A{layer_n}"])
            else:
                cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"])

        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]

        grads = {}
        # grads is a dictionary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1

        # WRITE CODE HERE
        for layer_n in range(self.n_hidden + 1, 0, -1):

            if layer_n == self.n_hidden + 1:
                delta = output - labels
                grads[f"dA{layer_n}"] = delta #batch_size x n_classes

            else:
                dZdA = self.activation(cache[f"A{layer_n}"], grad=True)

                grads[f"dZ{layer_n}"] = np.dot(grads[f"dA{layer_n + 1}"], self.weights[f"W{layer_n + 1}"].T)
                grads[f"dA{layer_n}"] = grads[f"dZ{layer_n}"] * dZdA

            grads[f"db{layer_n}"] = [np.mean(grads[f"dA{layer_n}"], axis=0)]
            grads[f"dW{layer_n}"] = np.dot(cache[f"Z{layer_n - 1}"].T, grads[f"dA{layer_n}"])/grads[f"dA{layer_n}"].shape[0]

        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            self.weights[f"b{layer}"] = self.weights[f"b{layer}"] - (np.array(grads[f"db{layer}"]) * self.lr)
            self.weights[f"W{layer}"] = self.weights[f"W{layer}"] - (np.array(grads[f"dW{layer}"]) * self.lr)

    def one_hot(self, y):
        # WRITE CODE HERE
        # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
        onehot = np.zeros((len(y), self.n_classes))
        onehot[np.arange(len(y)), y] = 1
        return onehot

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        # WRITE CODE HERE
        loss = - (np.sum(labels * np.log(prediction)))/labels.shape[0]
        return loss

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        print("Onehot done!")
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
                forward = self.forward(minibatchX)
                print("Calculating Backward...")
                backward = self.backward(forward, minibatchY)
                print("Updating Gradients....")
                self.update(backward)
                print("Done Updating")

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            print(train_loss)
            print(train_accuracy)
            print("Walooo")
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)
            print("Badoo")

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        # WRITE CODE HERE
        loss, accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return loss, accuracy

if __name__ == "__main__":
    # execute only if run as a script
    network = NN(hidden_dims=(101, 102, 300), seed=18580, activation="relu")
    network.initialize_weights((13,10))
    nuts = network.relu(-12)
    arr = network.relu(np.array( [[-4.20531609, -1.30181528,  0.3558277,  -0.45034355, -2.12831444,  4.60852173],
 [ 1.71268496,  2.08934569, -3.41514148, -4.81665524,  1.6761871 , -3.48868469],
 [-2.11352749, -0.12541742,  0.09494224, -0.40851653, -1.78112786,  2.9328691],
 [-1.78231487, -4.02503518, -0.18301219, -1.71247117, -4.74505586,  2.63550436],
 [-4.55586157,  4.2658225 ,  1.98140629, -4.98583587,  4.47600418, -3.86815993],
  [1.46006241,  2.00729018,  3.2818104 , -0.98721676, -2.78794563, -2.80903693],
 [-1.80911561, -2.62705898, -4.11489466, -2.12509617,  0.0132717 , -4.62143929],
 [-3.12344964,  3.16691651,  2.1467475  ,-1.07773678, -2.23058477, -2.25227978],
  [2.72399909 , 1.41372489,  0.19681336, -3.39173231, -3.90792354,  0.36038354],
 [-3.41011257, -0.89015505,  0.69868546,  1.24812195, -1.69962741, -2.64979953],
 [-0.53162188, -0.64714127,  4.07882306, -1.46101684, -3.88433414,  2.09574252],
  [4.33286902, -1.27940074,  3.21729937,  1.63976389, -0.20668857, -3.60910753],
  [2.84513832,  3.44506821, -0.26364691, -2.67997945,  1.32313253,  3.50294454],
  [4.886574  , -1.81100504, -4.65224475, -3.35953583, -4.41213762, -2.5915771],
  [2.65621983, -3.94882417, -0.75522953,  3.56661994, -2.75125862,  2.67649253],
  [4.18475065 , 3.97579457,  4.85450146, -2.19030447,  4.53508797 , 1.64092596],
 [-0.64480856, -4.87970984, -2.80634353,  2.79549102,  0.54572866, -0.22949278]]), grad=False)

    arr6 = network.relu(np.array([-0.64480856, -4.87970984, -2.80634353,  2.79549102,  0.54572866, -0.22949278]), grad=False)
    nuts_sig = network.sigmoid(12)
    arr_sig = network.sigmoid([0, 2, 15])


