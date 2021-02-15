
import os
import numpy as np
import pandas as pd
from .ridge_regression import RidgeRegression
from scipy import optimize


class CustomRegressor(RidgeRegression):

    def __init__(self, l2_penality=1, max_iter = 1000):
        self.l2_penality = l2_penality
        self.max_iter = max_iter
        self.W = None
        self.b = None
        self.X = None
        self.y = None

    @staticmethod
    def mse(y_pred, y_true):
        return (np.square(y_true - y_pred)).mean()

    @staticmethod
    def msle(y_pred, y_true):
        return np.mean(np.square(np.log(y_true + 1.) - np.log(y_pred + 1.)))

    @staticmethod
    def poission_loss(y_pred, y_true):
        """
        y must follow poission distrubution
        """
        return np.mean(y_pred - y_true * np.log(y_pred), axis=-1)

    @staticmethod
    def logcosh(y_pred, y_true):
        return np.sum(np.log(np.cosh(y_pred - y_true)))

    def loss(self):
        error = self.logcosh(self.predict(self.X), self.y)
        return error
    
    def l2_loss(self, W):
        self.W = W
        model_error = self.loss()
        model_error += sum(self.l2_penality * np.square(self.W))
        return model_error

    def optimize_weights(self):
        result = optimize.minimize(self.l2_loss, self.W, method='BFGS',
                            options={'maxiter': self.max_iter})
        self.W = result.x
        return