import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from econml.dr import DRLearner
from econml.dml import LinearDML
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class MLP(nn.Module):
    '''
    Fully connected neural network with 3 hidden layers.
    '''
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x).flatten()
        return logits


class ConvNet(nn.Module):
    '''
    Convolutional neural network with 2 convolutional layers and 
    2 fully connected layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)#.flatten()
        return logits

def compute_effect(dataset, method="AD", pred=False, total=False, econml=True):
    if pred:
        Y = dataset.Y_hat
    else:
        Y = dataset.Y.numpy()
    T = dataset.T.numpy()
    W = dataset.W.numpy()
    U = dataset.U.numpy()
    N = len(Y)
    if total: 
        if len(W.shape) == 1:
            W = np.expand_dims(W, axis=1)
        if len(U.shape) == 1:
            U = np.expand_dims(U, axis=1)
        X = np.concatenate((W, U), axis=1)
    else:
        if len(W.shape) == 1:
            X = np.expand_dims(W, axis=1)
        else:
            X = W

    if method == "AD":
        return np.mean(Y[T == 1]) - np.mean(Y[T == 0])
    
    # if method == "AF":
    #         return np.mean(Y[(T == 1) & (W == 1)])*np.mean(W==1) + \
    #             np.mean(Y[(T == 1) & (W == 0)])*np.mean(W==0) - \
    #             np.mean(Y[(T == 0) & (W == 1)])*np.mean(W==1) - \
    #             np.mean(Y[(T == 0) & (W == 0)])*np.mean(W==0)
    # if method == "AF_total":
    #     PO_T0 = 0
    #     PO_T1 = 0
    #     for k in range(dataset.k+1):
    #         for w in range(2):
    #             PO_T0 += np.mean(Y[(T == 0) & (U == k) & (W == w)])*np.mean(W == w)*np.mean(U == k)
    #             PO_T1 += np.mean(Y[(T == 1) & (U == k) & (W == w)])*np.mean(W == w)*np.mean(U == k)
    #             # print(f"k={k}, w={w}, PO_T0={PO_T0}, PO_T1={PO_T1}")
    #     return PO_T1 - PO_T0

    if method == "AIPW":
        model_propensity = RandomForestClassifier()
        model_outcome = RandomForestClassifier()
        if econml:
            model = LinearDML(model_t=model_propensity, 
                              model_y=model_outcome,
                            discrete_treatment=True,
                            random_state=1)
            model.fit(Y=Y, T=T, X=X); 
            return model.ate(X=X)
        else:
            model_propensity.fit(X = X,   
                                 y = T)
            model_outcome.fit(X = np.concatenate((X, np.expand_dims(T, axis=1)), axis=1),
                              y = Y)
            mu0 = model_outcome.predict(np.concatenate((X, np.zeros((N, 1))), axis=1))
            mu1 = model_outcome.predict(np.concatenate((X, np.ones((N, 1))), axis=1))
            ps = model_propensity.predict_proba(X)[:, 1]
            norm_1 = np.mean(T/ps)
            norm_0 = np.mean((1-T)/(1-ps))
            ite = mu1-mu0 + T * (Y-mu1) / (ps*norm_1) - (1-T) * (Y-mu0) / ((1-ps)*norm_0)
            return np.mean(ite)