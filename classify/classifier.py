from abc import ABCMeta, abstractclassmethod
import numpy as np
from numpy.random.mtrand import sample


class Classifier:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class Perceptron(Classifier):
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # update the weight i = 1 to m
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi

                # update the weight i = 0
                self.w_[0] += update

                # count errors
                errors += int(update != 0.0)

            self.errors_.append(errors)
        return self


class AdalineGD(Classifier):
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)

            # calculate errors: yi - φ(zi)
            errors = y - output

            # w = w + Δw
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # Sum of squared error (SEE)
            cost = (errors ** 2).sum() / 2.0

            self.cost_.append(cost)
        return self

    def activation(self, X):
        """活線型活性化関数の出力を計算。ADALINEでは不要"""
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
