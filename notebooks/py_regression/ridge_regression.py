import os
import numpy as np
import pandas as pd


class RidgeRegression(object):

    def __init__(self, l2_penality=1, learning_rate=0.0001, max_iter = 1000):
        self.l2_penality = l2_penality
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.W = None
        self.b = None
        self.X = None
        self.y = None


    def initialize_weights(self):
        """
        intializes weights to zero
        """
        self.W = np.random.normal(size = self.X.shape[1])
        self.b = 0
        return 
      
    def optimize_weights(self):
        """
        based on gradient descent
        """
        for i in range(self.max_iter):
            dW, db = self.calc_gradients()
            self.W = self.W - self.learning_rate * dW     
            self.b = self.b - self.learning_rate * db

        return


    def calc_gradients(self):
        """
        calculate grad for mse
        dw = 1/N ∑−2xi(yi−yihat)
        dy = 1/N ∑−2(yi−yihat)
        """
        # make prediction with current weights
        y_pred = self.predict(self.X)
        # calc diff
        residuals = self.y - y_pred
        # calculate dW & db
        dW = -1/len(self.X) * (2*(self.X.T).dot(residuals) + 
                (2 * self.l2_penality * self.W))
        db = -2*np.sum(residuals)/len(self.X)
        return dW, db


    def fit(self, X, y):
        """
        train
        """
        self.X = X
        self.y = y
        self.initialize_weights()
        self.optimize_weights()
        return self

    
    def predict(self, X):
        """
        predict step 
        """
        return np.matmul(X, self.W) + self.b


    def calc_l2_loss(self, loss_function):
        y_pred = self.predict(self.X)
        loss = loss_function(y_pred, self.y)
        # l2_loss
        loss += sum(self.l2_penality * np.square(self.W))
        return loss





