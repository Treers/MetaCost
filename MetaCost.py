# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.base import clone


class MetaCost(object):
    def __init__(self, S, L, C, q, m=50, n=1, p=True):
        if not isinstance(S, pd.DataFrame):
            raise ValueError('S must be a DataFrame object')
        new_index = list(range(len(S)))
        S.index = new_index
        self.S = S
        self.L = L
        self.C = C
        self.m = m
        self.n = len(S) * n
        self.p = p
        self.q = q

    def fit(self, flag, num_class):
        col = [col for col in self.S.columns if col != flag]
        S_ = {}
        M = []

        for i in range(self.m):
            # Let S_[i] be a resample of S with self.n examples
            S_[i] = self.S.sample(n=self.n, replace=True)

            X = S_[i][col].values
            y = S_[i][flag].values

            # Let M[i] = model produced by applying L to S_[i]
            model = clone(self.L)
            M.append(model.fit(X, y))

        label = []
        for i in range(len(self.S)):
            S_array = self.S[col].values

            if not self.q:
                k_th = [k for k, v in S_.items() if i not in v.index]
                M = list(np.array(M)[k_th])

            if self.p:
                P_j = [model.predict_proba(S_array[[i]]) for model in M]
            else:
                P_j = []
                vector = [0] * num_class
                for model in M:
                    vector[model.predict(S_array[[i]])] = 1
                    P_j.append(vector)

            # Calculate P(j|x)
            P = np.array(np.mean(P_j, 0)).T

            # Relabel
            label.append(np.argmin(self.C.dot(P)))



        # Model produced by applying L to S with relabeled y
        X_train = self.S[col].values
        y_train = np.array(label)
        model_new = clone(self.L)
        model_new.fit(X_train, y_train)

        return model_new


# test
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X = load_iris().data
X = pd.DataFrame(X)
y = load_iris().target
X['target'] = y

LR = LogisticRegression()
C = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
model = MetaCost(X, LR, C, q=True).fit('target', 3)

model.score(X[[0, 1, 2, 3]].values, X['target'])
