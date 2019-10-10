# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara

(Class)
> EnumLasso(modeltype='regression', maxitr=1000, tol=1e-4, rho=0.1, enumtype='k', r=1.1, k=10, delta=0, verbose=0, save=''):
    modeltype   : 'regression' or 'classification'
    maxitr      : maximum number of iterations for Lasso
    tol         : tolerance parameter to stop the iterative optimization
    rho         : regularization parameter
    enumtype    : 'r' or 'k'
    r           : stop enumeration when the objective function value is r-times larger than the optimal solution (valid only when enumtype='r')
    k           : stop enumeration when k solutions are enumerated (valid only when enumtype='k')
    delta       : coefficient threshold to skip unimportant features
    verbose     : print the feature search process when 'verbose=True'
    save        : save the model during the feature search into the specified file, e.g., save='model.npy'

(Method)
> EnumLasso.fit(X, y, featurename=[])
    X           : numpy array of size num x dim
    y           : numpy array of size num
    featurename : name of features
"""

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

class SupportedLasso(object):
    def __init__(self, modeltype='regression', maxitr=1000, tol=1e-4, rho=0.1):
        self.modeltype_ = modeltype
        self.maxitr_ = maxitr
        self.tol_ = tol
        self.rho_ = rho
        if self.modeltype_ == 'regression':
            self.lasso_ = Lasso(max_iter=self.maxitr_, tol=self.tol_, alpha=self.rho_)
        elif self.modeltype_ == 'classification':
            self.lasso_ = LogisticRegression(max_iter=self.maxitr_, tol=self.tol_, C=1/self.rho_, penalty='l1')
    
    def fit(self, x, y, supp, warm_start=False, a=[], b=[]):
        self.dim_ = x.shape[1]
        self.supp_ = supp
        if warm_start:
            self.lasso_.warm_start = warm_start
            if self.modeltype_ == 'regression':
                self.lasso_.coef_ = a
                self.lasso_.intercept_ = b
            elif self.modeltype_ == 'classification':
                self.lasso_.coef_ = a[np.newaxis, :]
                self.lasso_.intercept_ = np.array([b])
        if self.modeltype_ == 'classification':
            self.lasso_.C = 1 / (self.rho_ * x.shape[0])
        self.lasso_.fit(x[:, self.supp_], y)
        self.a_, self.b_ = self.__getA()
        self.obj_ = self.__getObj(x, y)
        
    def predict(self, x):
        return self.lasso_.predict(x[:, self.supp_])
    
    def __getA(self):
        a = np.zeros(self.dim_)
        if np.ndim(self.lasso_.coef_) == 2:
            a[self.supp_] = self.lasso_.coef_[0, :]
            return a, self.lasso_.intercept_[0]
        else:
            a[self.supp_] = self.lasso_.coef_
            return a, self.lasso_.intercept_
    
    def __getObj(self, x, y):
        if self.modeltype_ == 'regression':
            z = self.predict(x)
            return np.mean((y - z)**2) / 2 + self.rho_ * np.sum(np.abs(self.lasso_.coef_))
        elif self.modeltype_ == 'classification':
            z = x[:, self.supp_].dot(self.lasso_.coef_[0, :]) + self.lasso_.intercept_[0]
            f = np.mean(np.log(1 + np.exp(- (2 * y - 1) * z)))
            g = np.sum(np.abs(self.lasso_.coef_[0, :]))
            return f + self.rho_ * g

class EnumLasso(object):
    def __init__(self, modeltype='regression', maxitr=1000, tol=1e-4, rho=0.1, enumtype='k', r=1.1, k=10, delta=0, warm_start=True, check=0.99, verbose=0, save=''):
        self.modeltype_ = modeltype
        self.maxitr_ = maxitr
        self.tol_ = tol
        self.rho_ = rho
        self.r_ = r
        self.k_ = k
        self.enumtype_ = enumtype
        self.delta_ = delta
        self.warm_start_ = warm_start
        self.check_ = check
        self.verbose_ = verbose
        self.save_ = save
        self.lasso_ = SupportedLasso(maxitr=self.maxitr_, tol=self.tol_, rho=self.rho_, modeltype=self.modeltype_)
    
    def __str__(self):
        s = ''
        for i, obj in enumerate(self.obj_):
            s += 'Solution %3d: Obj. = %f\n' % (i+1, obj)
            for j, a in enumerate(self.a_[i]):
                if np.abs(a) < 1e-8:
                    continue
                s += '\t %s, Coef. = %f\n' % (self.featurename_[j], a)
            s += '\t %s, Coef. = %f\n' % ('intercept', self.b_[i])
        return s
    
    def setfeaturename(self, featurename):
        if len(featurename) > 0:
            self.featurename_ = featurename
        else:
            self.featurename_ = []
            for d in range(self.dim_):
                self.featurename_.append('x_%d' % (d+1,))
    
    def predict(self, x, idx=-1):
        if idx >= 0:
            return x.dot(self.a_[idx]) + self.b_[idx]
        else:
            K = len(self.obj_)
            y = np.zeros((x.shape[0], K))
            for i in range(K):
                y[:, i] = x.dot(self.a_[i]) + self.b_[i]
            return y
    
    def fit(self, x, y, featurename=[]):
        self.dim_ = x.shape[1]
        self.setfeaturename(featurename)
        
        # first lasso
        self.dim_ = x.shape[1]
        supp = list(range(self.dim_))
        self.lasso_.fit(x, y, supp, warm_start=False)
        nobranch = []
        zeros = set(np.where(np.abs(self.lasso_.a_) == 0)[0])
        if self.modeltype_ == 'regression':
            t = x.T.dot(y - self.lasso_.predict(x)) / (self.rho_ * y.size)
        elif self.modeltype_ == 'classification':
            z = (2 * y - 1) * (x.dot(self.lasso_.a_) + self.lasso_.b_)
            t = x.T.dot((2 * y - 1) * np.exp(-z) / (1 + np.exp(-z))) / (self.rho_ * y.size)
        searchlist = [(self.lasso_.obj_, self.lasso_.a_, self.lasso_.b_, [set(supp)], [set(nobranch)], zeros, t)]
        
        # enumeration
        self.obj_ = []
        self.a_ = []
        self.b_ = []
        self.const_ = []
        count = 0
        self.duplicate_ = []
        dup = 0
        while True:
            count += 1
            if len(searchlist) == 0:
                break
            i = np.argmin([v[0] for v in searchlist])
            obj = searchlist[i][0]
            a = searchlist[i][1]
            b = searchlist[i][2]
            supp = searchlist[i][3]
            nobranch = searchlist[i][4]
            self.obj_.append(obj)
            self.a_.append(a)
            self.b_.append(b)
            self.const_.append([list(np.setdiff1d(range(self.dim_), list(s))) for s in supp])
            self.duplicate_.append(dup)
            nonzeros = set(np.where(np.abs(a) > self.delta_)[0])
            searchlist.pop(i)
            if len(self.save_) > 0:
                joblib.dump(self, self.save_, compress=9)
            if self.enumtype_ == 'r':
                if obj > self.r_ * self.obj_[0]:
                    break
            if self.enumtype_ == 'k':
                if count >= self.k_:
                    break
            if self.verbose_:
                c = 0
                for n in enumerate(nobranch):
                    for d in nonzeros:
                        if d not in n:
                            c += 1
                print('\t Iteration %3d. Searching solution candidates: # of candidates = %d' % (count, c))
            for i, s in enumerate(supp):
                s = list(s)
                n = list(nobranch[i])
                for d in nonzeros:
                    if d in n:
                        continue
                    s.remove(d)
                    if len(s) == 0:
                        continue
                    
                    # pre-check
                    flg = False
                    for v in searchlist:
                        if len(set.union(set(s), v[5])) == self.dim_ and len(set.intersection(set(s), v[5])) == 0:
                            v[3].append(set(s[:]))
                            v[4].append(set(n[:]))
                            flg = True
                            break
                        elif len(set.union(set(s), v[5])) == self.dim_ and len(set.intersection(set(s), v[5])) > 0:
                            m = set(range(self.dim_)) - set(s)
                            j = list(set(v[5]) - m)
                            if np.max(np.abs(v[6][j])) < self.check_:
                                v[3].append(set(s[:]))
                                v[4].append(set(n[:]))
                                flg = True
                                break
                    if flg:
                        s.append(d)
                        n.append(d)
                        dup += 1
                        continue
                    
                    # lasso
                    if self.warm_start_:
                        self.lasso_.fit(x, y, s, warm_start=True, a=a[s], b=b)
                    else:
                        self.lasso_.fit(x, y, s, warm_start=False)
                    zeros = set(np.where(np.abs(self.lasso_.a_) == 0)[0])
                    if self.modeltype_ == 'regression':
                        t = x.T.dot(y - self.lasso_.predict(x)) / (self.rho_ * y.size)
                    elif self.modeltype_ == 'classification':
                        z = (2 * y - 1) * (x.dot(self.lasso_.a_) + self.lasso_.b_)
                        t = x.T.dot((2 * y - 1) * np.exp(-z) / (1 + np.exp(-z))) / (self.rho_ * y.size)
                    
                    # post-check
                    flg = False
                    for v in searchlist:
                        if zeros == v[5]:
                            v[3].append(set(s[:]))
                            v[4].append(set(n[:]))
                            flg = True
                            break
                    if flg:
                        s.append(d)
                        n.append(d)
                        dup += 1
                        continue
                    
                    # update list
                    searchlist.append((self.lasso_.obj_, self.lasso_.a_, self.lasso_.b_, [set(s[:])], [set(n[:])], zeros, t))
                    s.append(d)
                    n.append(d)

