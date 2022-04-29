import numpy as np
from PIL import Image
from scipy import signal
import matplotlib.pyplot as plt
class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        self.output = self.output.clip(-1000, 1000)
        #print(np.mean(self.output), "f", "dense")

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dinputs = self.dinputs.clip(-1000, 1000)
        #print(np.mean(self.dweights), "b", "dense")


class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.empty_like(inputs)
        self.output = np.maximum(0, inputs)
        #self.output.clip(-100, 100)
        #print(np.mean(self.output), "f", "relu")

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        #self.dinputs[self.inputs <= 0] = 0
        self.dinputs.clip(min=0)
        #self.dinputs.clip(-100, 100)
        #print(np.mean(self.inputs), "b", "relu")


class ActivationSoftmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
        self.output = self.output.clip(-100, 100)
        #print(np.mean(self.output), "f", "actSoftmax")

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        self.dinputs = self.dinputs.clip(-100, 100)
            #print(np.mean(self.dinputs), "b", "actSoftmax")


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
        #self.dinputs = self.dinputs.clip(-10, 10)


class Activation_softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        #print(np.mean(self.output), "f", "loss_cacteg")
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
        #self.dinputs = self.dinputs.clip(-10, 10)
        #print(np.mean(self.dinputs), "b", "loss_categ")

class SGD_Optimizer:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights

            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

class Convolution:
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_depth = input_depth
        self.input_shape = input_shape
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.weights = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.inputs[j], self.weights[i, j], "valid")
        self.output.clip(-100, 100)
        return self.output

    def backward(self, output_gradient):
        self.dweights = np.zeros(self.kernels_shape)
        self.dinputs = np.zeros(self.input_shape)
        self.dbiases = output_gradient
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.dweights[i, j] = signal.correlate2d(self.inputs[j], output_gradient[i], "valid")
                self.dinputs[j] += signal.convolve2d(output_gradient[i], self.weights[i, j], "full")
        self.dinputs = self.dinputs.clip(-100, 100)
        #self.weights = self.weights.clip(-10, 10)
        return self.dweights


class Max_pool:
    def forward(self, matrix):
        self.matrix = matrix
        self.output = np.zeros((matrix.shape[0], int(matrix.shape[1] / 2), int(matrix.shape[2] / 2)))
        self.mask = np.zeros_like(self.matrix)
        for dim in range(matrix.shape[0]):
            for i in range(0, matrix.shape[1], 2):
                for j in range(0, matrix.shape[2], 2):

                    if j >= matrix.shape[2] - 2 or i >= matrix.shape[1] - 2:
                        break
                    self.output[dim, i//2, j//2] = np.max(matrix[dim, i:i + 1, j:j + 1])
                    self.mask[dim, i:i + 1, j:j + 1] = np.max(matrix[dim, i:i + 1, j:j + 1])


            # plt.imshow(self.output)
            # plt.show()
        #print(np.mean(self.output), "f", "maxpool")
        self.output = self.output.clip(-10, 10)
        return self.output

    def backward(self, inputs):
        self.dinputs = np.copy(self.mask)
        for dim in range(inputs.shape[0]):
            for i in range(inputs.shape[1]):
                for j in range(inputs.shape[2]):
                    if self.dinputs[dim, i * 2:i* 2 + 1, j*2:j*2 + 1] > 0:
                        self.dinputs[dim, i * 2:i * 2 + 1, j * 2:j * 2 + 1] = inputs[dim, i, j]
        #print(np.mean(self.dinputs), "b", "max_pool")
        self.dinputs = self.dinputs.clip(-10, 10)
        return self.dinputs