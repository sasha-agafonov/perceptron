import numpy as np


class MLP:
    " Multi-layer perceptron "

    def __init__(self, sizes, beta=1, momentum=0.9):

        """
        sizes is a list of length four. The first element is the number of features
                in each samples. In the MNIST dataset, this is 784 (28*28). The second
                and the third  elements are the number of neurons in the first
                and the second hidden layers, respectively. The fourth element is the
                number of neurons in the output layer which is determined by the number
                of classes. For example, if the sizes list is [784, 5, 7, 10], this means
                the first hidden layer has 5 neurons and the second layer has 7 neurons.

        beta is a scalar used in the sigmoid function
        momentum is a scalar used for the gradient descent with momentum
        """
        self.beta = beta
        self.momentum = momentum

        self.nin = sizes[0]  # number of features in each sample
        self.nhidden1 = sizes[1]  # number of neurons in the first hidden layer
        self.nhidden2 = sizes[2]  # number of neurons in the second hidden layer
        self.nout = sizes[3]  # number of classes / the number of neurons in the output layer

        # Initialise the network of two hidden layers
        self.weights1 = (np.random.rand(self.nin + 1, self.nhidden1) - 0.5) * 2 / np.sqrt(self.nin)  # hidden layer 1
        self.weights2 = (np.random.rand(self.nhidden1 + 1, self.nhidden2) - 0.5) * 2 / np.sqrt(
            self.nhidden1)  # hidden layer 2
        self.weights3 = (np.random.rand(self.nhidden2 + 1, self.nout) - 0.5) * 2 / np.sqrt(
            self.nhidden2)  # output layer

    def train(self, inputs, targets, eta, niterations):
        """
        inputs is a numpy array of shape (num_train, D) containing the training images
                    consisting of num_train samples each of dimension D.

        targets is a numpy array of shape (num_train, D) containing the training labels
                    consisting of num_train samples each of dimension D.

        eta is the learning rate for optimization
        niterations is the number of iterations for updating the weights

        """
        ndata = np.shape(inputs)[0]  # number of data samples
        # adding the bias
        inputs = np.concatenate((inputs, -np.ones((ndata, 1))), axis=1)

        # numpy array to store the update weights
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
        updatew3 = np.zeros((np.shape(self.weights3)))

        for n in range(niterations):

            # forward phase
            self.outputs = self.forwardPass(inputs)

            # Error using the sum-of-squares error function
            error = 0.5 * np.sum((self.outputs - targets) ** 2)

            if (np.mod(n, 100) == 0):
                print("Iteration: ", n, " Error: ", error)

            # deriviative of error with respect to input to the softmax function
            deltao = (self.outputs - targets) * self.outputs * (1 - self.outputs)

            # use average of gradients rather than their sum; we don't want to overfit
            deltao /= ndata

            # compute the derivative of the second hidden layer
            # this layer used sigmoid instead of softmax
            deltah2 = self.beta * self.hidden2 * (1 - self.hidden2) * np.dot(deltao, self.weights3.T)

            # compute the derivative of the first hidden layer
            # this layer also used sigmoid
            deltah1 = self.beta * self.hidden1 * (1 - self.hidden1)

            # do not consider derviatives for bias in the second hidden layer
            deltah1 *= np.dot(deltah2[:, : -1], self.weights2.T)

            # we need to know how the previous weighs were changed for momentum
            prevw1 = updatew1
            prevw2 = updatew2
            prevw3 = updatew3

            # bias deriviatives in the first hidden layer not needed
            updatew1 = eta * (np.dot(inputs.T, deltah1[:, :-1]))

            # bias deriviatives in the second hidden layer not needed
            updatew2 = eta * (np.dot(self.hidden1.T, deltah2[:, :-1]))

            updatew3 = eta * (np.dot(self.hidden2.T, deltao))

            # apply momentum
            updatew1 += prevw1 * self.momentum
            updatew2 += prevw2 * self.momentum
            updatew3 += prevw3 * self.momentum

            # apply changes to weights
            self.weights1 -= updatew1
            self.weights2 -= updatew2
            self.weights3 -= updatew3


    def forwardPass(self, inputs):
        """
            inputs is a numpy array of shape (num_train, D) containing the training images
                    consisting of num_train samples each of dimension D.
        """


        # layer 1
        # compute the forward pass on the first hidden layer with the sigmoid function
        self.hidden1 = np.dot(inputs, self.weights1)

        # sigmoid 1
        self.hidden1 = np.reciprocal(1 + np.exp(-self.beta * self.hidden1), dtype = "float")

        # bias for hidden 1
        self.hidden1 = np.concatenate((self.hidden1, -np.ones((np.shape(self.hidden1)[0], 1))), axis = 1)

        # layer 2
        # the forward pass on the second hidden layer with the sigmoid function
        self.hidden2 = np.dot(self.hidden1, self.weights2)

        # sigmoid 2
        self.hidden2 = np.reciprocal(1 + np.exp(-self.beta * self.hidden2), dtype = "float")

        # bias for hidden 2
        self.hidden2 = np.concatenate((self.hidden2, -np.ones((np.shape(self.hidden2)[0], 1))), axis = 1)

        # output layer
        # the forward pass on the output layer with softmax function
        outputs = np.dot(self.hidden2, self.weights3)

        # softmax
        outputs = np.exp(outputs) / np.vstack(np.sum(np.exp(outputs), axis = 1))


        return outputs

    def evaluate(self, X, y):
        """
            this method is to evaluate our model on unseen samples
            it computes the confusion matrix and the accuracy

            X is a numpy array of shape (num_train, D) containing the testing images
                    consisting of num_train samples each of dimension D.
            y is  a numpy array of shape (num_train, D) containing the testing labels
                    consisting of num_train samples each of dimension D.
        """

        inputs = np.concatenate((X, -np.ones((np.shape(X)[0], 1))), axis=1)
        outputs = self.forwardPass(inputs)
        nclasses = np.shape(y)[1]

        # 1-of-N encoding
        outputs = np.argmax(outputs, 1)
        targets = np.argmax(y, 1)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        print("The confusion matrix is:")
        print(cm)
        print("The accuracy is ", np.trace(cm) / np.sum(cm) * 100)

        return cm
