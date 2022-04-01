from cbirlib.CBIR import DenseLayer, ActivationReLU, ActivationSoftmax, Loss_CategoricalCrossentropy
import nnfs
import matplotlib.pyplot as plt
import numpy as np
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data

nnfs.init()

X, y = vertical_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()
if __name__ == '__main__':

    dense1 = DenseLayer(2, 3)
    dense2 = DenseLayer(3, 3)

    activation1 = ActivationReLU()
    activation2 = ActivationSoftmax()
    loss_function = Loss_CategoricalCrossentropy()
    dense1.forward(X)

    lowest_loss = 9999999
    best_dense1_weights = dense1.weights.copy()
    nest_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    nest_dense2_biases = dense2.biases.copy()

    for iteration in range(10000):
        dense1.weights += 0.05 * np.random.randn(2, 3)
        dense1.biases += 0.05 * np.random.randn(1, 3)
        dense2.weights += 0.05 * np.random.randn(3, 3)
        dense2.biases += 0.05 * np.random.randn(1, 3)

        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        loss = loss_function.calculate(activation2.output, y)
        predictions = np.argmax(activation2.output, axis=1)
        accuracy = np.mean(predictions == y)

        if loss < lowest_loss:
            print("New set of weights found, iteration:", iteration, "loss:", loss, "acc:", accuracy)
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
