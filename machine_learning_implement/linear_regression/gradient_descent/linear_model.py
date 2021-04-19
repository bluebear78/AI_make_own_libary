import numpy as np
import pandas as pd
class linear_model_gd:
    def __init__(self, fit_intercept=True, copy_X=True,
                 alpha=0.0001, epochs=1000000, weight_decay=0.9):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self._alpha = alpha
        self._epochs = epochs

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
        
        #for i in range(theta.size):
        #    partial_marginal = X[:,i]
        #    errors_xi = (predictions - y) * partial_marginal
        #    theta[i] = theta[i] - self._alpha * (1.0/m) * errors_xi.sum()
        
        return theta
    
    def fit(self,X,y):
        if self.fit_intercept == True:
            make_ones = k = np.ones([X.shape[0],1])
            np.hstack((make_ones,X))
        
        cost_history = []
        theta_history = []

        theta = np.random.normal((X.shape[0],1))
        
        for _ in range(self._epochs):
            theta = self.gradient(X,theta)
            
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
    my_lgd = linear_model_gd(fit_intercept=True)
    my_lgd.fit(X,y)

    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(X,y)
    testX = np.array([[6,2],[2,4],[4,5]])
    regr.predict(testX)
    my_lgd.predict(testX)
    
