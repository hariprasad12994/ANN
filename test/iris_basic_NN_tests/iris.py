from neuralNetwork import NeuralNetwork
import sys
import numpy as np
from sklearn import datasets
from neuralNetwork import Trainer
from matplotlib import pyplot as plt


def main():
    weight_decay = int(sys.argv[1])
    depth = int(sys.argv[2])
    hidden_layer_size = int(sys.argv[3])

    iris = datasets.load_iris()

    features = iris.data[:, :4]
    # features /= 10
    for i in list(range(0, features.shape[1])):
        # features[:, i] = (features[:, i] - np.min(features[:, i])) / (np.max(features[:, i]) - np.min(features[:, i]))
        features[:, i] = (features[:, i] - np.mean(features[:, i])) / np.std(features[:, i])

    target = iris.target

    y = []
    for i in target:
        if i == 0:
            y.append([1, 0, 0])
        elif i == 1:
            y.append([0, 1, 0])
        else:
            y.append([0, 0, 1])
    y = np.array(y, dtype=float)
    y /= 10

    neuralNetwork = NeuralNetwork(features, hidden_layer_size, depth, y, weight_decay)

    neuralNetwork.getArchitecture()

    print(features)

    print("Weights before training:\n", neuralNetwork.weights)

    trainer = Trainer(neuralNetwork, features, y)

    trainer.train(features, y)

    print("Cost function optimisation:\n", trainer.J)

    print("Weights after training:\n", neuralNetwork.weights)

    neuralNetwork.feedForward(neuralNetwork.weights, neuralNetwork.sum_of_weighted_inputs,
                              neuralNetwork.activated_outputs)

    print("Expected output:\n", y)

    print("Output after training\n", neuralNetwork.activated_outputs[-1])

    a = neuralNetwork.activated_outputs[-1]

    plt.subplot(221)
    plt.plot(y[:, 0])
    plt.plot(a[:, 0])
    plt.plot(np.abs(a[:, 0] - y[:, 0]))
    plt.grid(1)
    plt.ylabel('y_values')
    plt.xlabel('no_of_data')

    plt.subplot(222)
    plt.plot(y[:, 1])
    plt.plot(a[:, 1])
    plt.plot(np.abs(a[:, 1] - y[:, 1]))
    plt.grid(1)
    plt.ylabel('y_values')
    plt.xlabel('no_of_data')

    plt.subplot(223)
    plt.plot(y[:, 2])
    plt.plot(a[:, 2])
    plt.plot(np.abs(a[:, 2] - y[:, 2]))
    plt.grid(1)
    plt.ylabel('y_values')
    plt.xlabel('no_of_data')

    plt.subplot(224)
    plt.plot(trainer.J)
    plt.grid(1)
    plt.ylabel('cost function')
    plt.xlabel('iterations')

    plt.show()

    print("Output-Input\n", neuralNetwork.activated_outputs[-1] - y)

    # j = neuralNetwork.costFunction(y)

    # neuralNetwork.calcGradients(y)

    # neuralNetwork.calcNumericalGradients(y)


if __name__ == main():
    main()
