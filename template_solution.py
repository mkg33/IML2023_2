# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
# Polynomial
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)
    
    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    countries = train_df.columns.values.tolist()
    for counNum, counName in enumerate(countries[1:]):
        if counName == 'price_CHF':
            y_train = train_df[counName].fillna(train_df[counName].mean())  
        else:
            X_train[:,counNum] = train_df[counName].fillna(train_df[counName].mean())  
            X_test[:,counNum] = test_df[counName].fillna(test_df[counName].mean())  
    col = 'price_CHF'
    train_df[col] = train_df[col].fillna(train_df[col].mean()) 
    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions
    gpr = GaussianProcessRegressor(kernel=DotProduct())
    gpr = GaussianProcessRegressor(kernel=RBF())
    gpr = GaussianProcessRegressor(kernel=Matern())
    gpr = GaussianProcessRegressor(kernel=RationalQuadratic())

    gpr.fit(X_train, y_train)
    
    
    y_pred = gpr.predict(X_test)
    
    
    # kr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
    # kr.fit(X_train, y_train)
    # y_pred = kr.predict(X_test)

    
    
    
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

