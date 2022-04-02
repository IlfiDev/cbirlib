from cbirlib.CBIR import *
import nnfs
import matplotlib.pyplot as plt
import numpy as np
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()
if __name__ == '__main__':
    optimizer = SGD_Optimizer(decay=1e-3, momentum=0.9)
    dense1 = DenseLayer(2, 64)
    activation1 = ActivationReLU()
    dense2 = DenseLayer(64, 3)
    loss_activation = Activation_softmax_Loss_CategoricalCrossentropy()

    for epoch in range(10001):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        #np.mean = среднее арифметическое
        accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(f'{epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}, ' +
                  f'lr: {optimizer.current_learning_rate}')

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
