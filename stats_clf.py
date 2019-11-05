import numpy as np
from math import log,e,exp,pi
import copy
# from datetime import datetime
from util import to_categorical, max_count

class BiBetaNB:
    """Beta-binomial Classifier
    Input: numpy array"""

    def __init__(self,a,b=None):
        self.lamb = 0
        self.alpha = a
        self.beta = a if b is None else b

    def set_params(self,a,b=None):
        self.alpha = a
        self.beta = a if b is None else b
        return self.alpha, self.beta

    def fit(self,feature,target):
        """ Calculate feature posterior estimate and evaluate using training data """

        target = to_categorical(target)
        self.lamb = len(target[target[:,1]>0])/len(target)   # Class prior
        self.post_pred = []*target.shape[1]
        #Calculate feature posterior 
        for t in range(target.shape[1]):
            idx = np.nonzero(target[:,t])  #Find all class label 
            n = len(idx[0])
            for f in range(feature.shape[1]):
                n1 = len(np.nonzero(feature[idx,f][0])[0])  
                self.post_pred.append((n1+self.alpha)/(n+self.alpha+self.beta))
        self.post_pred = np.asarray(self.post_pred)
        self.post_pred = np.reshape(self.post_pred,(2,self.post_pred.shape[0]//2))
        return self.evaluate(feature,target,False)

    def evaluate(self,x,y,trans=True):
        if trans:
            y = to_categorical(y)
        label = [self.predict(x[f,:]) for f in range(x.shape[0])]
        label = np.asarray(label)
        label = np.argmax(label,axis=1)
        err = len(np.nonzero(label.T - y[:,1])[0])/y.shape[0]
        return err, 1-err

    def predict(self,x):
        """ Return a list of log probability for each class label
        in the order of list index """

        if (len(x.shape) == 1):
            post = copy.deepcopy(self.post_pred)
            pred_label = [0]*post.shape[0]
            for t in range(post.shape[0]):
                # Find features that are 0 and calculate feature likelihood
                post[t,np.argwhere(x<1)[:,0]] = 1 - post[t,np.argwhere(x<1)[:,0]]
                for idx in range(len(post[t])):
                    post[t,idx] = log(post[t,idx])    # Calculate log likelihood
                pred_label[t] = log(abs(1 - t - self.lamb)) + post[t].sum()
            return pred_label
        else:
            pred = [self.predict(x[idx]) for idx in range(x.shape[0])]
            return pred


class GaussianNB:
    """This classifier assumes ML estimate for class label 
    Input: numpy array"""

    def __init__(self,):
        self.lamb = 0

    def fit(self,feature,target):
        """ Calculate mu and sigma squared based on data
        and evaluate using training samples"""

        self.mu = []
        self.sigSq = []
        target = to_categorical(target)
        self.lamb = [len(target[target[:,i]>0])/len(target) for i in range(target.shape[1])]  #class prior
        for t in range(target.shape[1]):
            idx = np.nonzero(target[:,t])                   #Find all class label
            for f in range(feature.shape[1]):
                avg = feature[idx,f][0].mean()
                self.mu.append(avg)
                self.sigSq.append(np.mean((feature[idx,f][0]-avg)**2))
        self.mu = np.reshape(np.asarray(self.mu),(target.shape[1],len(self.mu)//target.shape[1]))
        self.sigSq = np.reshape(np.asarray(self.sigSq),(target.shape[1],len(self.sigSq)//target.shape[1]))
        return self.evaluate(feature,target,False)

    def evaluate(self,x,y,trans=True):
        if trans:
            y = to_categorical(y)
        label = [self.predict(x[f,:]) for f in range(x.shape[0])]
        label = np.asarray(label)
        label = np.argmax(label,axis=1)
        err = len(np.nonzero(label.T - y[:,1])[0])/y.shape[0]
        return err, 1-err

    def predict(self,x):
        """ Return a list of log probability for each class label
        in the order of list index """

        if (len(x.shape) == 1):
            pred_prob = [0]*self.mu.shape[0]
            for i in range(self.mu.shape[0]):
                gauss = [(-(val-self.mu[i,idx])**2/(2*self.sigSq[i,idx]) - log((2*pi*self.sigSq[i,idx])**0.5)) for idx,val in enumerate(x)]
                pred_prob[i] = log(abs(self.lamb[i])) + sum(gauss)
            return pred_prob
        else:
            pred = [self.predict(x[idx]) for idx in range(x.shape[0])]
            return pred


class BiLogReg:
    """A logistic regression classifier for binary target label
    Only accept numpy array as input"""

    def __init__(self,lamb):
        self.lamb = lamb
        self.w = None

    def set_params(self,lamb):
        self.lamb = lamb
        return self.lamb

    def fit(self,x,y,conv = 1e-6):
        """Calculate the weight vector and evaluate using training data\n
        Parameter conv indicates the convergence condition"""
        
        self.w = np.asarray([0.0]*x.shape[1])           # row vector
        self.w = np.reshape(self.w,(1,self.w.shape[0]))
        X = np.vstack((np.asarray([0]*x.shape[1]).T,x))
        Y = np.vstack((np.random.randint(2),y))
        update = [1] * 3                                #initializes random update vector
        while(max(update) > conv and abs(min(update)) > conv):
            mu = [1/(1+exp(-1*self.w @ X[row,:])) for row in range(Y.shape[0])]
            mu = np.asarray(mu)                         # row vector
            mu = np.reshape(mu,(mu.shape[0],1))

            # Calculate gradient 
            self.grad = X.T @ (mu - Y) + self.lamb * np.vstack((0,self.w[:,1:].T))

            # Calculate Hessian
            diag = [i*(1-i) for i in mu]
            S = np.diagflat(diag) 
            self.hess = X.T @ S @ X + self.lamb * np.diagflat([0]+[1]*(x.shape[1]-1))
            update = np.linalg.inv(self.hess) @ self.grad

            self.w -= update.T
        return self.evaluate(x,y)

    def evaluate(self,x,y):
        label_prob = [self.predict(x[row,:]) for row in range(y.shape[0])]
        label = np.argmax(np.asarray(label_prob),axis=1)
        err = len(np.nonzero(label - y.T)[0])/y.shape[0]
        return err, 1-err

    def predict(self,x):
        """ Return a list of log probability for each class label
        in the order of list index """

        if (len(x.shape) == 1):
            return [ 1 - 1/(1+exp(-(self.w @ x))),1/(1+exp(-(self.w @ x)))] 
        else:
            return [self.predict(x[idx]) for idx in range(x.shape[0])]


class KNNeighbors:
    """A simple k-nearest neighbors classifier"""

    def __init__(self,kneighbor):
        self.kneighbor = kneighbor

    def set_params(self,kneighbor):
        self.kneighbor = kneighbor
        return self.kneighbor

    def fit(self,x,y,order=2,evaluate=True):
        """ Store the training data into the classifier. If evaluate = True,
        return the evaluation using training data (error and accuracy)\n
        The order parameter indicate the order of Minkowski distance calculated:
        1 = city block, 2 = euclidean"""

        self.order = order
        self.x = x
        self.y = y
        if (evaluate):
            dist = np.zeros((x.shape[0],x.shape[0]))
            for idx in range(x.shape[0]-1):
                dist[idx,idx+1:] = ((((x[idx]-x[idx+1:])**self.order).sum(axis=1))**(1/self.order)) 
            dist_mat = dist + dist.T
            dist_mat_sorted = np.argsort(dist_mat,axis=1)[:,1:self.kneighbor+1]
            map_est = np.apply_along_axis(max_count,1,dist_mat_sorted, y)       #MAP estimate
            err = len(np.nonzero(map_est - y.T)[0])/y.shape[0]
            return err,1-err
        else:
            return self.x,self.y

    def evaluate(self,x,y):
        pred_list = [self.predict(x[idx]) for idx in range(x.shape[0])]
        err = len(np.nonzero(np.asarray(pred_list) - y.T)[0])/y.shape[0]
        return err,1-err

    def predict(self,x):
        """ Return the MAP estimate of class label of new data point """

        if (len(x.shape) <= 1):
            dist = ((((x-self.x[:])**self.order).sum(axis=1))**(1/self.order)) 
            kneighbors =  np.argsort(dist)[:self.kneighbor]
            return max_count(kneighbors,self.y)
        else:
            return [self.predict(x[idx]) for idx in range(x.shape[0])]
