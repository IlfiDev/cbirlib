

from cbirlib.CBIR import *
import nnfs
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
from nnfs.datasets import spiral_data
from PIL import Image
from keras.datasets import mnist
from nnfs.datasets import vertical_data

nnfs.init()

#X, y = spiral_data(samples=100, classes=3)
#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')'
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(np.shape(x_train))
if __name__ == '__main__':
    X, y = spiral_data(100, 3)
    print(X.shape)
    #print(np.mean(self.dweights), "b", "actSoftmax")
    optimizer = SGD_Optimizer(decay=1e-3, momentum=0.9, learning_rate=1)
    convolution1 = Convolution((1, 28, 28), 3, 3)
    activation1 = ActivationReLU()
    max_pool1 = Max_pool()
    convolution2 = Convolution((1, 13, 13), 3, 3)
    activation2 = ActivationReLU()
    max_pool2 = Max_pool()
    dense1 = DenseLayer(75, 75)
    activation3 = ActivationReLU()
    dense2 = DenseLayer(75, 10)
    loss_activation = Activation_softmax_Loss_CategoricalCrossentropy()

    for epoch in range(10001):
        convolution1.forward(x_train)
        activation1.forward(convolution1.output)
        max_pool1.forward(activation1.output)
        convolution2.forward(max_pool1.output)
        activation2.forward(convolution2.output)
        max_pool2.forward(activation2.output)
        dense1.forward(max_pool2.output.reshape(1, 75))
        activation3.forward(dense1.output)
        dense2.forward(activation3.output)
        lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        loss = loss_activation.forward(dense2.output, y_train)
        #print(loss_activation.output)
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y_train.shape) == 2:
            y_train = np.argmax(y_train, axis=1)
        accuracy = np.mean(predictions == y_train)
        if not epoch % 100:
            plt.imshow(convolution2.output[0, :, :])
            plt.show()
            plt.imshow(convolution2.output[1, :, :])
            plt.show()
            plt.imshow(convolution2.output[2, :, :])
            plt.show()
            print(f'{epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}, ' +
                  f'lr: {optimizer.current_learning_rate}')

        loss_activation.backward(loss_activation.output, y_train)
        dense2.backward(loss_activation.dinputs)
        activation3.backward(dense2.dinputs)
        dense1.backward(activation3.dinputs)
        max_pool2.backward(dense1.dinputs.reshape((3, 5, 5)))
        activation2.backward(max_pool2.dinputs)
        convolution2.backward(activation2.dinputs)
        max_pool1.backward(convolution2.dinputs)
        activation1.backward(max_pool1.dinputs)
        convolution1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(convolution1)
        optimizer.update_params(convolution2)
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

