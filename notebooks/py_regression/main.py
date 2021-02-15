from .ridge_regression import RidgeRegression
from .custom import CustomRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import os
import numpy as np
import pandas as pd


def main() : 
      
    # Generating data
    n_samples = 1000
    n_outliers = 100

    X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=2,
                        n_informative=1, noise=10, coef=True, random_state=0)

    # Splitting dataset into train and test set 
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, 
                                test_size=0.2, random_state=2020) 
      
    # Model training     
    model = RidgeRegression() 
    model.fit(X_train, Y_train) 
      
    # Prediction on test set 
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    print('MSE:', mse)
    print("Trained Coef:", model.W)     
    print("Original Coef:", coef)
    print("y_pred:", np.round(Y_pred[1:4], 2))      
    print("y_true:", Y_test[1:4])

    r = Ridge()
    r.fit(X_train, Y_train)
    print('sklearn coef_:', r.coef_)

    cr = CustomRegressor()
    cr.fit(X_train, Y_train) 
    Y_pred = cr.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    print('Loss: logcosh , MSE:', mse, 'coef:', cr.W)      
      
if __name__ == "__main__" :  
    main() 
