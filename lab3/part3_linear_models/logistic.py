import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    X = np.vstack((np.ones((1,N)), X))
    w = np.zeros((P+1, 1))
    learning_rate = 0.001
    max_iter = 1000
    
    for i in range(max_iter):
        z = w.T @ X
        p = sigmoid(z)
        gradient = X @ (p - y).T
        w = w - learning_rate * gradient
    
    return w
