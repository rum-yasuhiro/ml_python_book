import numpy as np
from numpy.random.mtrand import sample
from classify.classifier import Classifier

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