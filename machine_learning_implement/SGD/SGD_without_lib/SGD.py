import numpy as np
import pandas as pd
class SGD:
    def __init__(self, fit_intercept=True, copy_X=True,
                 alpha=0.0001, epochs=1000000, weight_decay=0.9,batch_size=1):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self._alpha = alpha
        self._epochs = epochs
        self._batch = batch_size

        self._cost_history = []

        self._coef = None
        self._intercept = None
        self._new_X = None
        self._w_history = None
        self._weight_decay = weight_decay

    def cost(self, h, y):
        m = y.size
        predictions = h
        sqErrors = (predictions - y)

        J = (1.0/(2*m)) * sqErrors.T.dot(sqErrors)
        return J

    def hypothesis_function(self, X, theta):
        return X.dot(theta)

    def gradient(self, X, theta):
        predictions = X.dot(theta)
        m = X.shape[0]
        
        hypo = (predictions-y).dot(X)
        theta = theta - self._alpha * (1.0/m) * hypo
        
        return theta
    
    def fit(self,X,y):
        if self.fit_intercept == True:
            make_ones = k = np.ones([X.shape[0],1])
            np.hstack((make_ones,X))
        
        cost_history = []
        theta_history = []

        theta = np.random.normal((X.shape[0],1))
        
        for _ in range(self._epochs):
            X_copy = np.copy(X)
            index = np.array(range(0,X.shape[0]))
            if self.is_SGD:
                np.random.shuffle(index)
                
                #np.random.shuffle(X_copy)
            batch  = len(X_copy) // self._batch
            
            for batch_count in range(batch):
                X_batch = X_copy[index][:batch]
                theta_batch = X_batch[index][:batch]
                theta_batch = self.gradient(X_batch,theta_batch)
                #np.concatenate((tt,X[index][:x*-1]),axis=0)
                theta = np.concatenate((theta_batch,theta_batch[index][:batch*-1]),axis=0)
                X_copy = X_copy[index]
            
            if _ % 1000 == 0:
                theta_history.append(theta)
                cost_history.append(self.cost(self.hypothesis_function(X,theta),y))
        
        self._coef = theta
        return theta,np.array(cost_history),np.array(theta_history)

        
        
    def predict(self, X):
        return X.dot(self.coef)

    @property
    def coef(self):
        return self._coef

    @property
    def intercept(self):
        return self._intercept

    @property
    def weights_history(self):
        return np.array(self._w_history)

    @property
    def cost_history(self):
        return self._cost_history


if __name__ == "__main__":
    X = np.array([[1,2],[1,4],[1,5]])
    y = np.array([4,8,12])
    my_sgd = SGD(fit_intercept=True)
    my_sgd.fit(X,y)